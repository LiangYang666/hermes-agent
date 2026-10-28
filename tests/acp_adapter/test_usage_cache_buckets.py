"""ACP prompt-response usage must carry BOTH cache buckets (issue #71).

The turn result already carries `cache_write_tokens` (its key set is `_SESSION_TOKEN_KEYS`
in agent/turn_finalizer.py), and `input_tokens` is filled from `prompt_tokens`, which the
agent's own accounting defines as input + cache_read + cache_write. The adapter used to
forward only the read half, so a client could neither reconcile the prompt total nor bill
the writes — and rendered an unreported write as a measured zero.

Pinned here because the omission was invisible: the field is optional in ACP's schema, so
nothing failed — the cockpit just divided by the wrong number.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionState


class _QuietConn:
    """Accepts session updates; the assertions are about the returned PromptResponse."""

    def __init__(self) -> None:
        self.updates = []

    async def session_update(self, session_id, update):
        self.updates.append(update)


async def _noop_coro(*args, **kwargs):
    return None


def _make_server():
    server = HermesACPAgent.__new__(HermesACPAgent)
    server.session_manager = SimpleNamespace(save_session=lambda sid: None)
    server._conn = None
    server._send_usage_update = _noop_coro  # type: ignore[method-assign]
    server._pump_background = _noop_coro  # type: ignore[method-assign]
    return server


def _state() -> SessionState:
    return SessionState(
        session_id="s1",
        agent=SimpleNamespace(session_id="h1"),
        history=[],
        cancel_event=None,
        is_running=True,
        queued_prompts=[],
        runtime_lock=threading.Lock(),
        current_prompt_text="hello",
    )


def _finish(server, result):
    return asyncio.run(server._finish_turn(_state(), "s1", _QuietConn(), result, "h1", False))


def test_prompt_response_carries_both_cache_buckets():
    server = _make_server()
    # hermes' shape: prompt_tokens is the FULL prompt (fresh 1000 + read 3000 + write 500).
    resp = _finish(server, {
        "final_response": "", "messages": [],
        "prompt_tokens": 4500, "completion_tokens": 120, "total_tokens": 4620,
        "cache_read_tokens": 3000, "cache_write_tokens": 500,
    })
    assert resp.usage is not None
    assert resp.usage.input_tokens == 4500
    assert resp.usage.cached_read_tokens == 3000
    assert resp.usage.cached_write_tokens == 500


def test_unreported_write_stays_none_not_zero():
    """An agent/provider that reports no write counter must not look like a measured zero."""
    server = _make_server()
    resp = _finish(server, {
        "final_response": "", "messages": [],
        "prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240,
        "cache_read_tokens": 800,
    })
    assert resp.usage is not None
    assert resp.usage.cached_read_tokens == 800
    assert resp.usage.cached_write_tokens is None
