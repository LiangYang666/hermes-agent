"""Background rows reach a viewer that cannot read the owner's process registry.

The gateway already wakes a session when a ``notify_on_complete`` process exits; what it never did was
*publish* the rows, so a finite viewer drew a finished turn over a still-running build (issue #22).
``gateway.session_background`` sends the whole list on change, at the three moments it can change:
turn settlement, a drained completion, and a session attach.
"""
from types import SimpleNamespace

import pytest


def _live(route='route-1'):
    published = []

    def publish(session_id, payload, *, event_type='message.complete'):
        published.append((session_id, payload, event_type))

    return SimpleNamespace(route=route, background_json='', event_stream=SimpleNamespace(publish=publish)), published


def test_the_snapshot_is_sent_once_per_change(monkeypatch):
    from gateway import session_background

    rows = [{'id': 'p1', 'command': 'npm run build', 'pid': 42, 'status': 'running', 'uptimeSeconds': 7,
             'notifyOnComplete': True, 'kind': 'process'}]
    monkeypatch.setattr(session_background, '_rows', lambda route: rows)
    live, published = _live()
    assert session_background.publish_snapshot(live, 's') is True
    assert published == [('s', {'tasks': rows}, 'background.tasks')]
    # Same rows again: the viewer already has them, so nothing goes on the wire.
    assert session_background.publish_snapshot(live, 's') is False
    assert len(published) == 1
    # The exit is a change, and it carries the code the row now has.
    rows = [{**rows[0], 'status': 'exited', 'exitCode': 0, 'uptimeSeconds': 31}]
    assert session_background.publish_snapshot(live, 's') is True
    assert published[-1][1]['tasks'][0]['status'] == 'exited'
    assert published[-1][1]['tasks'][0]['exitCode'] == 0


def test_registry_rows_are_shaped_for_the_client(monkeypatch):
    from gateway import session_background
    rows = [
        {'session_id': 'p1', 'command': 'sleep 60', 'pid': 7, 'uptime_seconds': 3, 'status': 'running',
         'notify_on_complete': True},
        {'session_id': 'p2', 'command': 'true', 'pid': 8, 'uptime_seconds': 9, 'status': 'exited',
         'exit_code': 0},
    ]

    class Registry:
        def list_sessions(self, *, session_key=None):
            assert session_key == 'route-1'
            return rows

    import tools.process_registry as registry_module
    monkeypatch.setattr(registry_module, 'process_registry', Registry(), raising=False)
    out = session_background._rows('route-1')
    assert [row['id'] for row in out] == ['p1', 'p2']
    # An explicit boolean, never omitted: a client filters on it and an absent key must read "silent".
    assert out[0]['notifyOnComplete'] is True and out[0]['kind'] == 'process'
    assert out[1]['notifyOnComplete'] is False
    # Only an exited row carries an exit code.
    assert 'exitCode' not in out[0] and out[1]['exitCode'] == 0
    # An unknown route publishes nothing at all.
    assert session_background._rows('') == []


def test_a_completion_publishes_for_whichever_session_owns_the_route(monkeypatch):
    from gateway import session_background
    first, first_published = _live('route-1')
    second, second_published = _live('route-2')
    monkeypatch.setattr(session_background, '_rows', lambda route: [{'id': route, 'command': 'x', 'pid': 1,
                                                                     'status': 'running', 'uptimeSeconds': 1,
                                                                     'notifyOnComplete': True, 'kind': 'process'}])
    authority = SimpleNamespace(sessions={'a': first, 'b': second})
    assert session_background.publish_for_route(authority, 'route-2') == 1
    assert second_published and not first_published
    # Nothing to say for a route no live session owns (a detached viewer's old key).
    assert session_background.publish_for_route(authority, 'route-gone') == 0
    assert session_background.publish_for_route(authority, None) == 0


def test_a_ticking_uptime_alone_does_not_send_a_frame(monkeypatch):
    """`uptimeSeconds` changes every second: if it counted as a change, "publish on change" would
    publish on every event and the badge would repaint the whole list for nothing."""
    from gateway import session_background
    live, published = _live('route-1')
    row = {'id': 'p1', 'command': 'sleep 20', 'pid': 1, 'status': 'running', 'uptimeSeconds': 1,
           'notifyOnComplete': True, 'kind': 'process'}
    monkeypatch.setattr(session_background, '_rows', lambda route: [dict(row)])
    assert session_background.publish_snapshot(live, 's1') is True
    row['uptimeSeconds'] = 7
    monkeypatch.setattr(session_background, '_rows', lambda route: [dict(row)])
    assert session_background.publish_snapshot(live, 's1') is False
    # A real change still goes out, and the frame carries the live uptime.
    row['status'] = 'exited'
    row['exitCode'] = 0
    monkeypatch.setattr(session_background, '_rows', lambda route: [dict(row)])
    assert session_background.publish_snapshot(live, 's1') is True
    assert published[-1][1]['tasks'][0]['uptimeSeconds'] == 7
