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
async def test_prompt_gives_up_waiting_at_the_deadline_and_queues(monkeypatch):
    """Bounded: a wedged command must not wedge the operator's sentence forever."""
    monkeypatch.setattr(server_mod, "COMMAND_HANDOVER_TIMEOUT_S", 0.15)
    acp_agent, state, fake, conn = make_agent_and_state()
    state.command_op = True

    response = await asyncio.wait_for(_say(acp_agent, state, "先问一句"), timeout=5)

    assert response.stop_reason == "end_turn"
    assert fake.runs == []
    assert state.queued_prompts == ["先问一句"]
    assert len(_queued_texts(conn)) == 1


@pytest.mark.asyncio
async def test_two_prompts_racing_one_command_stay_ordered():
    """Two waiters must not merge their turns: each gets its own, in the order they arrived."""
    acp_agent, state, fake, conn = make_agent_and_state()
    fake._supports_active_turn_redirect = False
    state.command_op = True

    first = asyncio.create_task(_say(acp_agent, state, "第一句"))
    second = asyncio.create_task(_say(acp_agent, state, "第二句"))
    await asyncio.sleep(0.05)
    state.command_op = False
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)

    # The fake turn is instantaneous, so the second waiter legitimately finds the session idle
    # again and runs its own turn. What must never happen is the two sharing one turn, or the
    # later sentence overtaking the earlier one.
    assert fake.runs == ["第一句", "第二句"]
    assert state.queued_prompts == []
