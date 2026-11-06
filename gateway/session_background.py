"""Background work, as a session event a finite viewer can draw.

``terminal(background=true)`` and friends live in ``tools.process_registry`` inside the owner process.
The gateway already knows how to *wake* a session when one finishes, so the agent is fine — but a
client that only speaks the event stream (an ACP viewer, a phone) sees a turn "finished" while a build
is still running, and has no way to say so.

So the registry's rows for one session are published as ``background.tasks`` whenever they can have
changed: the end of a turn (a task was just started or read), a completion the gateway drained (the
doorbell rang), and a session attach (a viewer just reconnected). The payload is the whole list — the
state replaces the last one, so a lost frame costs a redraw, not a wrong row.

A task that exits WITHOUT ``notify_on_complete`` wakes nothing and so publishes nothing here: it is
the agent's own poll, invisible by design, and clients show only wake-capable rows anyway.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _rows(route: str | None) -> list[dict[str, Any]]:
    """This session's background processes, in the shape a client draws."""
    if not route:
        return []
    try:
        from tools.process_registry import process_registry
        raw = process_registry.list_sessions(session_key=route)
    except Exception:
        logger.debug('Could not read the process registry for %s', route, exc_info=True)
        return []
    rows: list[dict[str, Any]] = []
    for row in raw:
        task_id = str(row.get('session_id') or '')
        if not task_id:
            continue
        entry: dict[str, Any] = {
            'id': task_id,
            'command': str(row.get('command') or ''),
            'pid': row.get('pid'),
            'status': 'exited' if row.get('status') == 'exited' else 'running',
            'uptimeSeconds': int(row.get('uptime_seconds') or 0),
            # Explicit, never omitted: a client filters on this, and an older one must read "silent".
            'notifyOnComplete': row.get('notify_on_complete') is True,
            'kind': 'process',
        }
        if entry['status'] == 'exited':
            entry['exitCode'] = row.get('exit_code')
        rows.append(entry)
    return rows


def publish_snapshot(live, session_id: str) -> bool:
    """Publish this session's rows when they changed since the last snapshot. Returns whether a
    frame went out. Never raises: a status row is not worth failing a settled turn over."""
    try:
        rows = _rows(getattr(live, 'route', None))
        encoded = json.dumps(rows, sort_keys=True)
        if encoded == getattr(live, 'background_json', ''):
            return False
        live.background_json = encoded
        live.event_stream.publish(session_id, {'tasks': rows}, event_type='background.tasks')
        return True
    except Exception:
        logger.debug('Could not publish the background snapshot for %s', session_id, exc_info=True)
        return False


def publish_for_route(authority, route: str | None) -> int:
    """Publish for whichever live session owns ``route`` — the completion drain knows the session
    key, not the session id. Returns how many frames went out."""
    if not route:
        return 0
    published = 0
    for session_id, live in list(getattr(authority, 'sessions', {}).items()):
        if getattr(live, 'route', None) == route and publish_snapshot(live, session_id):
            published += 1
    return published
