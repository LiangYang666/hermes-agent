"""Issue #39 — a live ``delegate_task(background=true)`` must keep the ACP relay alive.

THE BUG. ``BackgroundRelay`` was built from ``process_registry.list_sessions()`` only. Subagents run
in-process and never appear there, so a session that dispatched two of them and finished its turn
looked like it had no background work at all: ``flush()`` returned False on the first idle tick, the
consume loop exited, and the completion event (already sitting in ``completion_queue`` as
``type=async_delegation``) was never drained. The agent only learned the children had finished if a
later turn happened to drain the queue — with no further input, never. The host also had nothing to
draw: no row, no count, no hint that the session was still working.

Both halves are the same data source, so both are asserted here: rows come from the delegation
ledger, and 「还有活的吗」 includes them.
"""

from __future__ import annotations

import asyncio
import time

import tools.async_delegation as ad
from acp_adapter.background import BackgroundRelay

SID = "acp-session-39"


class FakeConn:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def session_update(self, *, session_id: str, update) -> None:  # noqa: ANN001
        payload = getattr(update, "field_meta", None) or {}
        self.frames.append(payload.get("hermes") or {})


class FakeRegistry:
    """Just enough of ``ProcessRegistry``: no processes, but a real completion queue."""

    def __init__(self) -> None:
        self.pending: list[tuple[dict, str]] = []

    def list_sessions(self, session_key: str = "") -> list[dict]:  # noqa: ARG002
        return []

    def drain_notifications(self, session_key: str = "", skip_poll_observed: bool = True):  # noqa: ARG002
        out, self.pending = self.pending, []
        return out


def _record(**over: object) -> dict:
    rec = {
        "delegation_id": "deleg_39",
        "session_key": SID,
        "status": "running",
        "goal": "read the repo and report",
        "is_batch": True,
        "goals": ["read the repo", "report"],
        "task_indexes": [0, 1],
        "dispatched_at": time.time() - 95,
    }
    rec.update(over)
    return rec


def _install(monkeypatch, registry: FakeRegistry) -> None:
    monkeypatch.setattr(BackgroundRelay, "_registry", staticmethod(lambda: registry))
    monkeypatch.setattr(ad, "_records", {}, raising=False)


def _relay() -> tuple[BackgroundRelay, FakeConn]:
    conn = FakeConn()
    loop = asyncio.new_event_loop()
    return BackgroundRelay(conn, SID, loop), conn


def test_delegation_row_is_visible_and_counts_children(monkeypatch):
    reg = FakeRegistry()
    _install(monkeypatch, reg)
    with ad._records_lock:
        ad._records["deleg_39"] = _record()

    relay, conn = _relay()
    tasks = asyncio.run(relay.flush_snapshot_only())

    assert len(tasks) == 1, tasks
    row = tasks[0]
    assert row["kind"] == "delegation"
    assert row["status"] == "running"          # 宿主只有 running/exited 两态
    assert row["phase"] == "running"
    assert row["taskCount"] == 2               # batch of two = two children
    assert row["command"] == "read the repo and report"
    assert row["pid"] is None
    assert row["notifyOnComplete"] is True     # 欠会话一次唤醒
    assert row["uptimeSeconds"] >= 90

    frame = conn.frames[-1]
    assert frame["kind"] == "background" and frame["tasks"][0]["kind"] == "delegation"


def test_flush_stays_alive_while_a_delegation_runs(monkeypatch):
    """The regression itself: with no registry process at all, remaining must be True."""
    reg = FakeRegistry()
    _install(monkeypatch, reg)
    with ad._records_lock:
        ad._records["deleg_39"] = _record(status="finalizing")

    relay, _conn = _relay()
    assert asyncio.run(relay.flush()) is True

    # ...and once the ledger no longer lists it, the loop is allowed to stop.
    with ad._records_lock:
        ad._records["deleg_39"]["status"] = "completed"
    assert asyncio.run(relay.flush()) is False


def test_completion_event_reaches_the_host_while_the_relay_ticks(monkeypatch):
    reg = FakeRegistry()
    _install(monkeypatch, reg)
    with ad._records_lock:
        ad._records["deleg_39"] = _record()

    relay, conn = _relay()
    assert asyncio.run(relay.flush()) is True
    conn.frames.clear()

    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_39",
        "session_key": SID,
        "goal": "read the repo and report",
        "status": "completed",
        "summary": "两个子任务都读完了",
        "dispatched_at": time.time() - 200,
        "completed_at": time.time(),
    }
    reg.pending.append((evt, "委派的子任务已完成：read the repo and report"))
    with ad._records_lock:
        ad._records["deleg_39"]["status"] = "completed"

    remaining = asyncio.run(relay.flush())
    assert remaining is False, "没有活干了，循环该退出"
    kinds = [f.get("kind") for f in conn.frames]
    assert "completion" in kinds, kinds
