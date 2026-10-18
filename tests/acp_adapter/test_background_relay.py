"""Tests for acp_adapter.background — the ACP side of background processes (#22).

Contract under test: completions are drained at a turn boundary and shipped as an EMPTY thought chunk
carrying ``_meta.hermes = {kind: "completion", …}``; the running snapshot rides the same door as
``kind: "background"``; and an exit is reported exactly once even when the process was never tagged
``notify_on_complete`` (no event is ever published for those, so a state-only view would drop the row
without ever saying it ended).
"""

import asyncio
from typing import Any

import pytest

from acp_adapter.background import BackgroundRelay


class FakeConn:
    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    async def session_update(self, session_id: str, update: Any) -> None:
        self.updates.append((session_id, update))


class FakeRegistry:
    def __init__(self, drain=None, rows=None) -> None:
        self._drain = list(drain or [])
        self.rows = list(rows or [])
        self.drain_kwargs: list[dict[str, Any]] = []

    def drain_notifications(self, **kwargs: Any):
        self.drain_kwargs.append(kwargs)
        out, self._drain = self._drain, []
        return out

    def list_sessions(self, **kwargs: Any):
        return list(self.rows)


def _payloads(conn: FakeConn) -> list[dict[str, Any]]:
    """The ``_meta.hermes`` payloads, read off the model the way the wire would serialise them."""
    out = []
    for _sid, update in conn.updates:
        dumped = update.model_dump(by_alias=True)
        out.append(dumped["_meta"]["hermes"])
        # The carrier must be an empty thought chunk: no prose, nothing a host would show as a thought.
        assert dumped["sessionUpdate"] == "agent_thought_chunk"
        assert dumped["content"]["text"] == ""
    return out


def _relay(monkeypatch, conn: FakeConn, registry: FakeRegistry) -> BackgroundRelay:
    monkeypatch.setattr(BackgroundRelay, "_registry", staticmethod(lambda: registry))
    return BackgroundRelay(conn, "sess-1", asyncio.get_event_loop())


def test_completion_frame_shape_and_drain_scope(monkeypatch):
    conn = FakeConn()
    registry = FakeRegistry(drain=[(
        {"type": "completion", "session_id": "p1", "exit_code": 0, "command": "npm run build"},
        "后台进程 p1 已结束（exit 0）：npm run build",
    )])
    relay = _relay(monkeypatch, conn, registry)

    asyncio.run(relay.flush())

    # Scope: this session only, and an output the agent already polled is not re-announced.
    assert registry.drain_kwargs == [{"session_key": "sess-1", "skip_poll_observed": True}]
    payloads = _payloads(conn)
    completions = [p for p in payloads if p["kind"] == "completion"]
    assert len(completions) == 1
    assert completions[0]["processId"] == "p1"
    assert completions[0]["exitCode"] == 0
    assert "npm run build" in completions[0]["text"]
    # The snapshot rides along and is empty here — the process was never seen running by this relay.
    assert [p for p in payloads if p["kind"] == "background"][-1]["tasks"] == []


def test_snapshot_is_emitted_once_per_change(monkeypatch):
    conn = FakeConn()
    running = [{
        "session_id": "p1", "command": "sleep 600", "pid": 4242,
        "status": "running", "uptime_seconds": 12,
    }]
    registry = FakeRegistry(rows=running)
    relay = _relay(monkeypatch, conn, registry)

    assert asyncio.run(relay.flush()) is True  # work remains
    snapshots = [p for p in _payloads(conn) if p["kind"] == "background"]
    assert len(snapshots) == 1
    assert snapshots[0]["tasks"][0]["id"] == "p1"
    assert snapshots[0]["tasks"][0]["uptimeSeconds"] == 12

    # Unchanged registry → no second frame (the tick runs every 5 s; a frame per tick would rewrite the
    # status row forever).
    assert asyncio.run(relay.flush()) is True
    assert len([p for p in _payloads(conn) if p["kind"] == "background"]) == 1


def test_silent_exit_is_reported_once_then_dropped(monkeypatch):
    """A process without notify_on_complete publishes no event — the snapshot is the only witness."""
    conn = FakeConn()
    registry = FakeRegistry(rows=[{
        "session_id": "p1", "command": "make", "pid": 1, "status": "running", "uptime_seconds": 3,
    }])
    relay = _relay(monkeypatch, conn, registry)
    asyncio.run(relay.flush())

    registry.rows = [{
        "session_id": "p1", "command": "make", "pid": 1, "status": "exited",
        "uptime_seconds": 9, "exit_code": 2,
    }]
    relay._last_snapshot = ""  # pretend the exit changed the row set; dedupe is asserted elsewhere
    assert asyncio.run(relay.flush()) is False  # nothing left running
    snapshots = [p for p in _payloads(conn) if p["kind"] == "background"]
    last = snapshots[-1]["tasks"]
    assert [(t["id"], t["status"], t["exitCode"]) for t in last] == [("p1", "exited", 2)]

    # Once reported, the row is forgotten: no zombie row forever.
    assert asyncio.run(relay.flush()) is False
    snapshots = [p for p in _payloads(conn) if p["kind"] == "background"]
    assert snapshots[-1]["tasks"] == []


def test_attach_path_reports_state_without_draining(monkeypatch):
    conn = FakeConn()
    registry = FakeRegistry(
        drain=[({"type": "completion", "session_id": "p9", "exit_code": 0}, "done")],
        rows=[{"session_id": "p1", "command": "x", "pid": 1, "status": "running", "uptime_seconds": 1}],
    )
    relay = _relay(monkeypatch, conn, registry)

    tasks = asyncio.run(relay.flush_snapshot_only())

    assert registry.drain_kwargs == []  # attach must not synthesise a turn
    assert [t["id"] for t in tasks] == ["p1"]
    assert [p["kind"] for p in _payloads(conn)] == ["background"]


@pytest.mark.asyncio
async def test_idle_tick_stops_when_nothing_runs(monkeypatch):
    conn = FakeConn()
    registry = FakeRegistry()
    monkeypatch.setattr(BackgroundRelay, "_registry", staticmethod(lambda: registry))
    relay = BackgroundRelay(conn, "sess-1", asyncio.get_running_loop())

    monkeypatch.setattr("acp_adapter.background.TICK_SECONDS", 0.01)
    await asyncio.wait_for(relay.run(is_idle=lambda: True), timeout=2)
    # One final empty snapshot, then the loop returned instead of spinning forever.
    assert [p["kind"] for p in _payloads(conn)] == ["background"]
