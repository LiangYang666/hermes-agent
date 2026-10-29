"""A prompt that races a state-mutating command must wait for it, not queue behind it (issue #74).

The cockpit replays a new session's picker choices and the operator's first message back to back.
ACP dispatches each JSON-RPC message on its own, so ``session/prompt`` can reach the agent while
``session/set_model`` still holds ``command_op``. Queueing there answered "Queued for the next
turn." with ``stop_reason=end_turn``: a turn that never started, and the prompt's real turn (run by
the drain) then had no turn boundary — no stop button, no usage, no completion notice.
"""

import asyncio
from types import SimpleNamespace

import pytest
from acp.schema import TextContentBlock

from acp_adapter import server as server_mod
from tests.acp_adapter.test_acp_commands import (  # noqa: F401 — fixtures/harness reuse
    CaptureConn,
    FakeAgent,
    NoopDb,
    make_agent_and_state,
)
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


def _say(acp_agent, state, text):
    return acp_agent.prompt(session_id=state.session_id, prompt=[TextContentBlock(type="text", text=text)])


def _queued_texts(conn):
    return [
        u
        for _sid, u in conn.updates
        if "Queued for the next turn" in str(getattr(u, "content", u))
    ]


@pytest.mark.asyncio
async def test_prompt_waits_for_command_op_instead_of_queueing():
    acp_agent, state, fake, conn = make_agent_and_state()
    state.command_op = True  # the model switch is in flight

    task = asyncio.create_task(_say(acp_agent, state, "改成中文回答"))
    await asyncio.sleep(0.1)  # a queue-first implementation would have answered by now
    assert fake.runs == []
    assert state.queued_prompts == []
    assert _queued_texts(conn) == []

    state.command_op = False  # the switch lands
    response = await asyncio.wait_for(task, timeout=5)

    assert response.stop_reason == "end_turn"
    # It ran as a REAL turn, in the prompt itself — not drained later out of the queue.
    assert fake.runs == ["改成中文回答"]
    assert state.queued_prompts == []
    assert _queued_texts(conn) == []


@pytest.mark.asyncio
async def test_prompt_still_queues_when_a_turn_is_really_running():
    """The queue keeps its meaning: a genuine running turn is not "a command in flight"."""
    acp_agent, state, fake, conn = make_agent_and_state()
    # FakeAgent can redirect; a runtime that cannot is the case where a lost prompt is QUEUED
    # (a supported redirect steers the live turn instead, which is the better answer).
    fake._supports_active_turn_redirect = False
    state.is_running = True
    state.command_op = True

    response = await _say(acp_agent, state, "第二个问题")

    assert response.stop_reason == "end_turn"
    assert fake.runs == []
    assert state.queued_prompts == ["第二个问题"]
    assert len(_queued_texts(conn)) == 1


@pytest.mark.asyncio
async def test_a_slow_command_is_never_answered_as_a_finished_turn(monkeypatch):
    """No deadline: past the warn threshold the prompt KEEPS waiting.

    Every early answer re-creates issue #74 — "Queued for the next turn." with `end_turn` is a
    finished turn that never started, and a wedged command would park the sentence in a queue only
    the wedged path could drain. The escape hatch is the operator's Stop (next test).
    """
    monkeypatch.setattr(server_mod, "COMMAND_HANDOVER_WARN_S", 0.05)
    acp_agent, state, fake, conn = make_agent_and_state()
    state.command_op = True

    task = asyncio.create_task(_say(acp_agent, state, "先问一句"))
    await asyncio.sleep(0.3)  # well past the warn threshold
    assert not task.done(), "the prompt must still be waiting"
    assert state.queued_prompts == []
    assert _queued_texts(conn) == []

    state.command_op = False
    response = await asyncio.wait_for(task, timeout=5)
    assert response.stop_reason == "end_turn"
    assert fake.runs == ["先问一句"]
    assert state.queued_prompts == []


@pytest.mark.asyncio
async def test_stop_pressed_while_waiting_cancels_instead_of_running_the_turn():
    """`cancel()` is not gated on is_running — a Stop during the wait must win.

    Without this the turn starts after the operator stopped it, `prompt()` clears the flag on the
    way in, and the response says `end_turn` for a turn that was cancelled.
    """
    acp_agent, state, fake, conn = make_agent_and_state()
    state.command_op = True

    task = asyncio.create_task(_say(acp_agent, state, "别跑了"))
    await asyncio.sleep(0.1)
    await acp_agent.cancel(session_id=state.session_id)  # the Stop button
    state.command_op = False  # the command finally lands

    response = await asyncio.wait_for(task, timeout=5)

    assert response.stop_reason == "cancelled"
    assert fake.runs == [], "a turn the operator stopped must not run"
    assert state.queued_prompts == []


class _SlowAgent(FakeAgent):
    """A turn that OUTLIVES the poll interval — the configuration where ordering is decided."""

    def run_conversation(self, *, user_message, conversation_history, task_id, **kwargs):
        import time

        time.sleep(0.3)
        return super().run_conversation(
            user_message=user_message, conversation_history=conversation_history, task_id=task_id, **kwargs
        )


def _make_slow():
    fake = _SlowAgent()
    manager = SessionManager(agent_factory=lambda **kwargs: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    conn = CaptureConn()
    acp_agent.on_connect(conn)
    return acp_agent, state, fake, conn


@pytest.mark.asyncio
async def test_two_prompts_racing_one_command_keep_arrival_order():
    """Arrival order, even when the first turn outlives the poll interval.

    Ordering by "whichever timer woke first" reversed 1 run in 15 with a realistic turn; the loser
    was then `redirect()`ed into the winner's turn, i.e. the earlier sentence became a correction of
    a turn started by a sentence written after it.
    """
    for _ in range(8):
        acp_agent, state, fake, _conn = _make_slow()
        state.command_op = True

        # No gap: two frames the socket handed over in the same tick. Without arrival ordering they
        # are ~50/50 — the poll wakes whoever the timer heap likes — which is the inversion.
        first = asyncio.create_task(_say(acp_agent, state, "第一句"))
        second = asyncio.create_task(_say(acp_agent, state, "第二句"))
        await asyncio.sleep(0.05)
        state.command_op = False
        await asyncio.wait_for(asyncio.gather(first, second), timeout=10)

        assert fake.runs[:1] == ["第一句"], fake.runs
        # Steer the LIVE turn is a fine fate for the LATER sentence; the earlier one being treated as
        # a correction of a turn written after it is the bug.
        assert "第一句" not in fake.redirects, f"the earlier sentence was redirected: {fake.redirects}"
        if fake.redirects == ["第二句"]:
            assert fake.runs == ["第一句"], fake.runs
        else:
            # Queued behind the slow first turn, then drained after it — never before it.
            await asyncio.sleep(1.0)
            assert fake.runs == ["第一句", "第二句"], fake.runs
