"""Background processes over ACP: completion wake-ups and a running snapshot.

WHY THIS MODULE EXISTS. ``terminal(background=true, notify_on_complete=true)`` is a closed loop on the
CLI and on the gateway: the child exits, ``ProcessRegistry`` publishes a completion event, the host
injects it into the session, and the agent opens a new turn to take the result. Over ACP the producer
side runs and **nobody consumes**: nothing under ``acp_adapter/`` ever touched
``process_registry.completion_queue``, so completions sat in an in-process queue until the process
died. Two consequences, both reported downstream (#22):

* the agent never wakes — the session reads "finished" while a build is still running;
* a host has no idea background work exists at all, because there is no frame carrying it.

WHAT ACP GIVES US TO CARRY IT. Nothing — there is no background/completion object, and inventing a
``sessionUpdate`` type would be invisible to hosts that validate the schema. So both carriers reuse
the door ``make_status_cb`` already proved: an ordinary ``agent_thought_chunk`` with EMPTY text and the
payload in ``_meta.hermes``::

    {"kind": "completion", "text": "…", "exitCode": 1, "processId": "…"}
    {"kind": "background", "tasks": [{"id", "command", "pid", "status", "uptimeSeconds", "exitCode", "notifyOnComplete"}]}

A host that does not know a kind renders nothing (an empty thought), which is the honest degradation;
a host that knows ``completion`` can synthesise the same follow-up prompt the gateway would have
injected, and one that knows ``background`` can draw a row. Neither side has to ship in lock-step.

SILENT EXITS. ``notify_on_complete`` is the only thing that publishes an event, so a process started
without it exits into silence and a snapshot that only read the registry would drop the row without
ever saying it ended. This module therefore remembers which ids it has seen running: an exit is
reported once (status ``exited`` + exit code) before the row disappears.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

import acp

logger = logging.getLogger(__name__)

# The gateway's own fallback cadence (``BACKGROUND_POLL_MS`` in the desktop status stack) — the point is
# a cheap reconcile, not a live progress feed: the agent already polls its own processes when it cares.
TICK_SECONDS = 5.0

# Truncation lives here, not in the host: a command line can be enormous, and the frame is a status
# row, not a transcript. 200 was too small to answer the question the row is opened for — the operator
# reading 「它到底在跑什么」 got a pipeline cut mid-flag. 4000 keeps the whole line for every realistic
# one-liner (and for a `bash -c '…'` heredoc) while still bounding a pathological paste.
COMMAND_CHARS = 4000
COMPLETION_TEXT_CHARS = 1200


def _summary_text(raw: dict[str, Any], formatted: str) -> str:
    """The sentence the host will inject as a follow-up prompt.

    ``drain_notifications`` already formats one (it is what the gateway's synthetic message uses), so
    prefer it and fall back to a plain one rather than sending an empty carrier.
    """
    text = str(formatted or "").strip()
    if text:
        return text[:COMPLETION_TEXT_CHARS]
    cmd = str(raw.get("command") or "").strip()[:COMMAND_CHARS]
    code = raw.get("exit_code")
    return f"后台进程已结束（exit {code}）：{cmd}"[:COMPLETION_TEXT_CHARS]


class BackgroundRelay:
    """One per ACP session: turn a per-process queue into frames a host can act on."""

    def __init__(self, conn: Any, session_id: str, loop: asyncio.AbstractEventLoop) -> None:
        # Public: the agent compares it on every pump, because a reconnect installs a new client object
        # and frames written to the previous socket reach nobody.
        self.conn = conn
        self.session_id = session_id
        self._loop = loop
        self._last_snapshot = ""
        # Last snapshot actually emitted, so the caller can ask "is anything still running?" without
        # re-reading the registry.
        self.last_tasks: list[dict[str, Any]] = []
        self._seen_running: dict[str, dict[str, Any]] = {}
        self._task: asyncio.Task | None = None

    # ---- wire -------------------------------------------------------------------------------
    async def _send(self, payload: dict[str, Any]) -> None:
        """One frame, awaited rather than fire-and-forget.

        ``events._send_update`` exists for callbacks that run on WORKER threads; here every caller is
        already on the loop (turn-end, session attach, the idle tick), and scheduling a coroutine onto the
        loop we are standing on would block ``future.result(timeout=5)`` for five seconds and then drop the
        frame. So the update is awaited directly.
        """
        update = acp.update_agent_thought_text("")
        # ``field_meta``, not ``_meta``: `_meta` is only the wire ALIAS of the model field, so assigning
        # it hangs an attribute the model does not hold and ``model_dump`` drops it silently (the empty
        # thought then says nothing at all — see ``make_status_cb``).
        update.field_meta = {"hermes": payload}
        try:
            await self.conn.session_update(session_id=self.session_id, update=update)
        except Exception:
            logger.debug("Could not send background frame for %s", self.session_id, exc_info=True)

    # ---- registry ----------------------------------------------------------------------------
    @staticmethod
    def _registry() -> Any:
        from tools.process_registry import process_registry

        return process_registry

    def _drain(self) -> list[tuple[dict[str, Any], str]]:
        """Pop this session's pending completions.

        ``skip_poll_observed=True`` mirrors the CLI/gateway default: if the agent already read the
        result with ``process_manage`` during the turn, it knows — re-injecting would make it read the
        same output twice and answer the same question twice.
        """
        try:
            return list(self._registry().drain_notifications(
                session_key=self.session_id, skip_poll_observed=True,
            ))
        except Exception:
            logger.debug("Could not drain process completions for %s", self.session_id, exc_info=True)
            return []

    def _rows(self) -> list[dict[str, Any]]:
        try:
            rows = self._registry().list_sessions(session_key=self.session_id)
        except Exception:
            logger.debug("Could not list background processes for %s", self.session_id, exc_info=True)
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            sid = str(row.get("session_id") or "")
            status = str(row.get("status") or "running")
            entry: dict[str, Any] = {
                "id": sid,
                "command": str(row.get("command") or "")[:COMMAND_CHARS],
                "pid": row.get("pid"),
                "status": status,
                "uptimeSeconds": int(row.get("uptime_seconds") or 0),
                # Does this process OWE the session a wake-up? `notify_on_complete` is the whole
                # difference between 「agent 已经收工，这个进程自己跑完就完事」 and 「agent 还在
                # 等这次唤醒才能继续」 — the host has no way to tell them apart from id/status
                # alone, and the two want different rendering (issue #35). Additive: a host that
                # does not know the field ignores it.
                "notifyOnComplete": bool(row.get("notify_on_complete")),
            }
            if status == "running":
                # Remember it so its exit is reported even when it was never tagged notify_on_complete.
                self._seen_running[sid] = entry
                out.append(entry)
                continue
            entry["exitCode"] = row.get("exit_code")
            if sid not in self._seen_running:
                # Finished before this relay ever saw it running: nothing to reconcile, and the
                # completion (if any) already carries the exit.
                continue
            # One-shot: the exit is news exactly once. Forget it now, so the NEXT snapshot drops the row
            # instead of leaving 「exited」 on screen for the rest of the session.
            self._seen_running.pop(sid, None)
            out.append(entry)
        # A row that vanished from the registry without an exit (cleared, reaped) must not linger either.
        for sid in list(self._seen_running):
            if not any(r.get("id") == sid for r in out):
                self._seen_running.pop(sid, None)
        # Subagents are the OTHER producer on this queue (issue #39). They never appear in
        # `list_sessions` — `delegate_task(background=true)` runs in-process — so a snapshot built
        # from the registry alone said 「没有后台任务」 while two children were reading files for eight
        # minutes, and `flush()`'s 「还有活的吗」 answered False on the first idle tick, which is what
        # killed the consumer before the completion ever arrived. Rows AND the verdict come from the
        # delegation ledger, so the host draws them and the relay keeps ticking until they finish.
        out.extend(self._delegation_rows())
        return out

    def _delegation_rows(self) -> list[dict[str, Any]]:
        """Live delegations, shaped like a task row so a host needs no second renderer.

        ``kind`` is additive: a host that predates it sees a row whose ``command`` is the goal and
        whose ``pid`` is null — legible, if plainer than it could be.
        """
        try:
            from tools.async_delegation import live_delegations_for_session
            live = live_delegations_for_session(session_key=self.session_id)
        except Exception:
            logger.debug("Could not list delegations for %s", self.session_id, exc_info=True)
            return []
        rows: list[dict[str, Any]] = []
        for row in live:
            goal = str(row.get("goal") or "")
            if not goal:
                goals = row.get("goals") or []
                goal = str(goals[0]) if goals else ""
            rows.append({
                "id": str(row.get("delegation_id") or ""),
                "kind": "delegation",
                # `command` carries the goal so an unaware host still shows something true.
                "command": goal[:COMMAND_CHARS],
                "pid": None,
                # Every state in the ledger's live set means 「still owes this session a wake-up」,
                # which is exactly what the registry calls `running`.
                "status": "running",
                "phase": str(row.get("status") or ""),
                "uptimeSeconds": int(row.get("uptime_seconds") or 0),
                "notifyOnComplete": True,
                "taskCount": int(row.get("task_count") or 1),
            })
        return rows

    # ---- pump --------------------------------------------------------------------------------
    async def flush_snapshot_only(self) -> list[dict[str, Any]]:
        """Emit the CURRENT snapshot without draining completions (the attach path).

        A session that is re-attached has no turn in flight, so a drained completion would be a prompt
        the host has to invent a turn for. State, however, must be correct immediately: the operator who
        reopens the cockpit should see 「2 个后台任务在跑」 without waiting for the first tick.
        """
        tasks = self._rows()
        blob = json.dumps(tasks, sort_keys=True, ensure_ascii=False)
        self.last_tasks = tasks
        if blob != self._last_snapshot:
            self._last_snapshot = blob
            await self._send({"kind": "background", "tasks": tasks})
        return tasks

    async def flush(self) -> bool:
        """Emit completions, then the snapshot if it changed. True when work still remains.

        Completions first: a host that reacts by waking a turn should see the finished row in the same
        breath, and a snapshot that arrives before its completion would show a process that has already
        ended as still running.
        """
        for raw, formatted in self._drain():
            await self._send({
                "kind": "completion",
                "text": _summary_text(raw, formatted),
                "exitCode": raw.get("exit_code"),
                "processId": raw.get("session_id"),
                "command": str(raw.get("command") or "")[:COMMAND_CHARS],
            })
        tasks = self._rows()
        blob = json.dumps(tasks, sort_keys=True, ensure_ascii=False)
        self.last_tasks = tasks
        if blob != self._last_snapshot:
            self._last_snapshot = blob
            await self._send({"kind": "background", "tasks": tasks})
        return any(t.get("status") == "running" for t in tasks)

    async def run(self, *, is_idle: Callable[[], bool]) -> None:
        """Reconcile until nothing is running. Started by the agent after a turn ends.

        Idle-only: during a turn the agent is polling its own processes and the frames would fight the
        turn's own status line. A turn that starts mid-tick simply pauses the loop, which is why the
        check is re-read every iteration instead of being a one-shot guard.
        """
        try:
            while True:
                await asyncio.sleep(TICK_SECONDS)
                if not is_idle():
                    continue
                if not await self.flush():
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Background relay for %s stopped on an error", self.session_id, exc_info=True)

    def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()

    def start(self, *, is_idle: Callable[[], bool]) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.ensure_future(self.run(is_idle=is_idle))
