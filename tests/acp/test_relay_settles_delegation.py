"""Settling a subagent result is the ACP relay's job — and only after the frame lands.

THE BUG. A subagent batch is a DURABLE row in the delegation ledger, and ``restore_completions()``
re-offers every row still marked ``pending`` on the next drain. The ACP relay delivered those rows
but never acknowledged them, so each idle window re-offered the same finished batch and woke the
session again: one 「子代理回来了」turn per /stop, forever.

THE FIX, AND WHY IT LIVES HERE. Acknowledging inside ``ProcessRegistry.drain_notifications`` "works"
for the ACP relay and breaks everyone else: the CLI and TUI consumers deliver the same row through a
claim → complete handshake, so a row settled during the drain leaves them nothing to claim and the
result is silently DROPPED instead of replayed. Settling therefore happens in the relay, after
``session_update`` returns — because a frame written to a dead socket reaches nobody, and
acknowledging it would turn "wakes twice" into "lost forever".
"""

from __future__ import annotations

import asyncio

import tools.async_delegation as ad
from acp_adapter.background import BackgroundRelay
from tools.process_registry import process_registry

SID = "acp-session-settle"


class FakeConn:
    def __init__(self, *, fail: bool = False) -> None:
        self.frames: list[dict] = []
        self.fail = fail

    async def session_update(self, *, session_id: str, update) -> None:  # noqa: ANN001
        if self.fail:
            raise RuntimeError("socket is gone")
        payload = getattr(update, "field_meta", None) or {}
        self.frames.append(payload.get("hermes") or {})


class FakeRegistry:
    def __init__(self, pending: list[tuple[dict, str]]) -> None:
        self.pending = pending

    def list_sessions(self, session_key: str = "") -> list[dict]:  # noqa: ARG002
        return []

    def drain_notifications(self, session_key: str = "", skip_poll_observed: bool = True):  # noqa: ARG002
        out, self.pending = self.pending, []
        return out


def _delegation_event(**over: object) -> dict:
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_settle",
        "session_key": SID,
        "results": [{"task_index": 0, "status": "completed", "summary": "done"}],
    }
    evt.update(over)
    return evt


def _relay(conn: FakeConn, pending: list[tuple[dict, str]], monkeypatch) -> BackgroundRelay:
    registry = FakeRegistry(pending)
    monkeypatch.setattr(BackgroundRelay, "_registry", staticmethod(lambda: registry))
    monkeypatch.setattr(BackgroundRelay, "_rows", lambda self: [])
    return BackgroundRelay(conn, SID, asyncio.new_event_loop())


def test_a_delivered_batch_is_acknowledged(monkeypatch):
    """The whole point: a row that reached the host must stop being re-offered."""
    settled: list[str] = []
    monkeypatch.setattr(ad, "mark_completion_delivered", lambda did: settled.append(did))

    conn = FakeConn()
    relay = _relay(conn, [(_delegation_event(), "子代理 batch 完成")], monkeypatch)
    assert asyncio.run(relay.flush()) is False       # nothing still running
    completions = [f for f in conn.frames if f.get("kind") == "completion"]
    assert len(completions) == 1, "the completion must still be delivered"
    assert settled == ["deleg_settle"], "delivered ≠ acknowledged; the row stays pending forever"


def test_a_frame_that_never_landed_leaves_the_row_pending(monkeypatch):
    """An unreachable socket must not acknowledge: the tick re-offers it instead of losing it."""
    settled: list[str] = []
    monkeypatch.setattr(ad, "mark_completion_delivered", lambda did: settled.append(did))

    relay = _relay(FakeConn(fail=True), [(_delegation_event(), "子代理 batch 完成")], monkeypatch)
    asyncio.run(relay.flush())
    assert settled == [], "acknowledging a frame nobody received turns a replay into a silent loss"


def test_an_interim_notice_does_not_acknowledge_the_batch(monkeypatch):
    """A per-task ``task_failure_notice`` shares the batch id but is not the result."""
    settled: list[str] = []
    monkeypatch.setattr(ad, "mark_completion_delivered", lambda did: settled.append(did))

    interim = _delegation_event(task_failure_notice=True)
    assert ad.is_interim_delegation_event(interim), "fixture must look like an interim notice"
    relay = _relay(FakeConn(), [(interim, "子代理 task 0 失败")], monkeypatch)
    asyncio.run(relay.flush())
    assert settled == [], "the final row would be gone before it exists"


def test_a_plain_process_completion_is_not_touched(monkeypatch):
    settled: list[str] = []
    monkeypatch.setattr(ad, "mark_completion_delivered", lambda did: settled.append(did))

    relay = _relay(FakeConn(), [({"type": "completion", "session_id": "proc_1"}, "exit 0")], monkeypatch)
    asyncio.run(relay.flush())
    assert settled == []


def test_the_registry_drain_does_not_settle_anything(monkeypatch):
    """Guard the contract the review caught: settling inside the drain starves CLI/TUI claims."""
    import queue as _queue

    settled: list[str] = []
    monkeypatch.setattr(ad, "mark_completion_delivered", lambda did: settled.append(did))

    q: "_queue.Queue[dict]" = _queue.Queue()
    q.put(_delegation_event())
    monkeypatch.setattr(process_registry, "completion_queue", q)
    monkeypatch.setattr(process_registry, "_completions_restored", True, raising=False)
    monkeypatch.setattr(process_registry, "restore_completions", lambda: None, raising=False)
    drained = process_registry.drain_notifications(session_key=SID, skip_poll_observed=True)
    assert len(drained) == 1
    assert settled == [], (
        "drain_notifications must not acknowledge: CLI/TUI deliver through claim_event_delivery, "
        "and a row settled here returns None from the claim and is skipped"
    )
