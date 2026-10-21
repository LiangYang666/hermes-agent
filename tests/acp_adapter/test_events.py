"""Tests for acp_adapter.events — callback factories for ACP notifications."""

import asyncio
import gc
import uuid
import warnings
from concurrent.futures import Future
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import AgentPlanUpdate

from acp_adapter.events import (
    _send_update,
    make_message_cb,
    make_status_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)


@pytest.fixture()
def mock_conn():
    """Mock ACP Client connection."""
    conn = MagicMock(spec=acp.Client)
    conn.session_update = AsyncMock()
    return conn


@pytest.fixture()
def event_loop_fixture():
    """Create a real event loop for testing threadsafe coroutine submission."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tool progress callback
# ---------------------------------------------------------------------------


class TestToolProgressCallback:



    def test_duplicate_same_name_tool_calls_use_fifo_ids(self, mock_conn, event_loop_fixture):
        """Multiple same-name tool calls should be tracked independently in order."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        progress_cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        step_cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            progress_cb("tool.started", "terminal", "$ ls", {"command": "ls"})
            progress_cb("tool.started", "terminal", "$ pwd", {"command": "pwd"})
            assert len(tool_call_ids["terminal"]) == 2

            step_cb(1, [{"name": "terminal", "result": "ok-1"}])
            assert len(tool_call_ids["terminal"]) == 1

            step_cb(2, [{"name": "terminal", "result": "ok-2"}])
            assert "terminal" not in tool_call_ids


# ---------------------------------------------------------------------------
# Thinking callback
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Step callback
# ---------------------------------------------------------------------------


class TestStepCallback:



    @pytest.mark.parametrize("raw, expected", [("", ""), (0, "0"), (False, "False")])
    def test_falsey_result_reaches_client_unchanged(self, mock_conn, event_loop_fixture, raw, expected):
        """A present-but-falsey ``result`` is the tool's real output, not a missing key (#10845)."""
        from collections import deque

        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, {"terminal": deque(["tc-f"])}, {})
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            mock_rcts.return_value = MagicMock(spec=Future)
            cb(1, [{"name": "terminal", "result": raw}])
        mock_btc.assert_called_once_with("tc-f", "terminal", result=expected, function_args=None, snapshot=None)




    def test_tool_progress_captures_snapshot_metadata(self, mock_conn, event_loop_fixture):
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        with patch("acp_adapter.events.make_tool_call_id", return_value="tc-meta"), \
             patch("acp_adapter.events._send_update") as mock_send, \
             patch("agent.display.capture_local_edit_snapshot", return_value="snapshot"):
            cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
            cb("tool.started", "write_file", None, {"path": "diff-test.txt", "content": "hello"})

        assert list(tool_call_ids["write_file"]) == ["tc-meta"]
        assert tool_call_meta["tc-meta"] == {
            "args": {"path": "diff-test.txt", "content": "hello"},
            "snapshot": "snapshot",
        }
        mock_send.assert_called_once()

    @pytest.mark.parametrize("tool_name", ["todo", "todo_list"])
    def test_todo_completion_emits_native_plan_update_after_tool_completion(
        self, mock_conn, event_loop_fixture, tool_name
    ):
        from collections import deque

        tool_call_ids = {tool_name: deque(["tc-todo"])}
        loop = event_loop_fixture
        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})
        todo_result = (
            '{"todos":['
            '{"id":"inspect","content":"Inspect ACP","status":"completed"},'
            '{"id":"patch","content":"Patch renderer","status":"in_progress"},'
            '{"id":"old","content":"Drop stale task","status":"cancelled"}'
            '],"summary":{"total":3}}'
        )

        with patch("acp_adapter.events._send_update") as mock_send:
            cb(1, [{"name": tool_name, "result": todo_result}])

        updates = [call.args[3] for call in mock_send.call_args_list]
        assert [getattr(update, "session_update", None) for update in updates] == [
            "tool_call_update",
            "plan",
        ]
        plan = updates[1]
        assert isinstance(plan, AgentPlanUpdate)
        assert [entry.content for entry in plan.entries] == [
            "Inspect ACP",
            "Patch renderer",
            "[cancelled] Drop stale task",
        ]
        assert [entry.status for entry in plan.entries] == ["completed", "in_progress", "completed"]
        assert [entry.priority for entry in plan.entries] == ["medium", "medium", "medium"]
    def test_bridge_tool_call_of_todo_list_emits_plan_update(self, mock_conn, event_loop_fixture):
        # Deferred tools reach the transcript through the tool_search bridge:
        # name="tool_call", arguments={"calls": [{"name": "todo_list", ...}]}.
        # Plan emission must unwrap the bridge like display layers do.
        from collections import deque

        tool_call_ids = {"tool_call": deque(["tc-bridge"])}
        loop = event_loop_fixture
        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})
        todo_result = (
            '{"todos":[{"id":"a","content":"Say hi","status":"completed"}],'
            '"revision":1,"summary":{"total":1}}'
        )
        bridge_args = (
            '{"calls":[{"name":"todo_list","arguments":{"todos":'
            '[{"id":"a","content":"Say hi","status":"completed"}]}}]}'
        )

        with patch("acp_adapter.events._send_update") as mock_send:
            cb(1, [{"name": "tool_call", "result": todo_result, "arguments": bridge_args}])

        updates = [call.args[3] for call in mock_send.call_args_list]
        assert [getattr(u, "session_update", None) for u in updates] == ["tool_call_update", "plan"]
        assert [e.content for e in updates[1].entries] == ["Say hi"]





# ---------------------------------------------------------------------------
# Message callback
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Scheduler-failure regression
# ---------------------------------------------------------------------------

class TestSendUpdate:
    def test_scheduler_failure_closes_update_coroutine(self, event_loop_fixture):
        """If run_coroutine_threadsafe raises, _send_update must close the coro."""
        created = {"coro": None}

        async def _session_update(session_id, update):
            return None

        conn = MagicMock()

        def _capture_update(session_id, update):
            created["coro"] = _session_update(session_id, update)
            return created["coro"]

        conn.session_update = _capture_update

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch(
                "agent.async_utils.asyncio.run_coroutine_threadsafe",
                side_effect=RuntimeError("scheduler down"),
            ):
                _send_update(conn, "session-1", event_loop_fixture, {"type": "noop"})
            gc.collect()

        assert created["coro"] is not None
        assert created["coro"].cr_frame is None
        # Only count warnings about THIS test's coroutine; other tests
        #  may emit unrelated
        # "coroutine was never awaited" warnings that bleed through.
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_session_update" in str(w.message)
        ]
        assert runtime_warnings == []


class TestAssistantMessageIds:
    """Streamed chunks carry a per-message ACP messageId; the None flush sentinel starts a new one."""

    def test_deltas_share_one_uuid_until_flush(self, mock_conn, event_loop_fixture):
        from acp_adapter.events import AssistantMessageIdAllocator

        ids = AssistantMessageIdAllocator()
        cb = make_message_cb(mock_conn, "s", event_loop_fixture, ids)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            cb("Hello ")
            cb("")  # empty delta is ignored, not a flush
            cb("world")
            cb(None)  # flush sentinel — closes the message
            cb("next turn")
        assert sent[0].message_id == sent[1].message_id
        assert sent[2].message_id != sent[0].message_id
        # ACP requires UUID-format message ids.
        assert uuid.UUID(sent[0].message_id) and uuid.UUID(sent[2].message_id)

    def test_thought_chunks_carry_id(self, mock_conn, event_loop_fixture):
        from acp_adapter.events import AssistantMessageIdAllocator

        ids = AssistantMessageIdAllocator()
        think = make_thinking_cb(mock_conn, "s", event_loop_fixture, ids)
        msg = make_message_cb(mock_conn, "s", event_loop_fixture, ids)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            think("pondering")
            msg("answer")
        # Reasoning and answer of the same reply share one message id.
        assert sent[0].message_id == sent[1].message_id

    def test_no_allocator_keeps_legacy_shape(self, mock_conn, event_loop_fixture):
        cb = make_message_cb(mock_conn, "s", event_loop_fixture)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            cb("text")
        assert sent[0].message_id is None


class TestToolCallsAlwaysReachATerminalStatus:
    """A tool call left ``in_progress`` makes a finished turn look like it ran nothing.

    Two invariants: every call is closed exactly once from its own ``tool.completed``
    (the ``prev_tools`` step closer stands down once completions arrive), and whatever
    is still open at turn end is failed with BOTH per-turn dicts drained together."""

    def _patch(self):
        return patch("acp_adapter.events.asyncio.run_coroutine_threadsafe")

    def test_tool_completed_closes_the_call_once_and_the_step_closer_stands_down(self, mock_conn, event_loop_fixture):
        from collections import deque

        ids, meta, turn_state = {"read": deque(["tc-1", "tc-2"])}, {"tc-1": {"args": {"path": "a"}}}, {}
        progress = make_tool_progress_cb(mock_conn, "s", event_loop_fixture, ids, meta, turn_state=turn_state)
        step = make_step_cb(mock_conn, "s", event_loop_fixture, ids, meta, turn_state)
        with self._patch() as rcts, patch("acp_adapter.events.build_tool_complete") as btc:
            rcts.return_value = MagicMock(spec=Future)
            progress("tool.completed", "read", None, None, result="file body")
            step(2, [{"name": "read", "result": "file body", "arguments": '{"path": "a"}'}])
        btc.assert_called_once_with(
            "tc-1", "read", result="file body", function_args={"path": "a"}, snapshot=None, is_error=False,
        )
        assert list(ids["read"]) == ["tc-2"] and "tc-1" not in meta

    def test_step_fallback_coerces_wire_arguments_and_turn_end_flush_fails_what_is_still_open(
        self, mock_conn, event_loop_fixture,
    ):
        """No completion projected: the step closer must survive the JSON-string ``arguments``
        the wire carries (a real ``write_file`` close raised on ``.get`` and was swallowed);
        a denied edit is then failed at turn end, draining ``tool_call_ids`` AND ``tool_call_meta``."""
        from collections import deque

        from acp_adapter.events import flush_open_tool_calls

        ids = {"write_file": deque(["tc-1"]), "edit": deque(["tc-denied"])}
        meta = {"tc-1": {"args": {"path": "a"}, "snapshot": None}, "tc-denied": {"args": {}}}
        step = make_step_cb(mock_conn, "s", event_loop_fixture, ids, meta, {})
        with self._patch() as rcts:
            rcts.return_value = MagicMock(spec=Future)
            step(2, [{"name": "write_file", "result": "ok", "arguments": '{"path": "a", "content": "x"}'}])
            assert [c.args[1].status for c in mock_conn.session_update.call_args_list] == ["completed"]
            assert flush_open_tool_calls(mock_conn, "s", event_loop_fixture, ids, meta) == 1
            assert flush_open_tool_calls(mock_conn, "s", event_loop_fixture, ids, meta) == 0
        statuses = [c.args[1].status for c in mock_conn.session_update.call_args_list]
        assert statuses == ["completed", "failed"]
        assert ids == {} and meta == {}

    def test_tool_completed_is_error_flag_closes_the_call_as_failed(self, mock_conn, event_loop_fixture):
        """``tool.completed`` carries the executor's ``is_error``; a cancelled tool's plain-text
        result trips no heuristic, so dropping the flag showed an interrupted call green."""
        from collections import deque

        ids = {"terminal": deque(["tc-1"])}
        progress = make_tool_progress_cb(mock_conn, "s", event_loop_fixture, ids, {})
        with self._patch() as rcts:
            rcts.return_value = MagicMock(spec=Future)
            progress("tool.completed", "terminal", None, None, is_error=True,
                     result="[Tool execution cancelled — terminal was skipped. User sent a new message]")
        assert [c.args[1].status for c in mock_conn.session_update.call_args_list] == ["failed"]


# ---------------------------------------------------------------------------
# make_status_cb — the memory/compaction status line (ACP has no status object)
# ---------------------------------------------------------------------------


def _wire(update: Any) -> dict[str, Any]:
    """What the host actually receives for this update.

    Serialisation is `model_dump(by_alias=True)`, so a field is only on the wire if it is a real model
    field; an attribute set outside the schema (`update._meta`) vanishes here while still reading back
    fine in Python — which is why the assertions below go through this door and not the attribute.
    """
    return update.model_dump(by_alias=True, exclude_none=True)


class TestMakeStatusCb:
    """`👁️ Hindsight — recalled 8 memories` has to travel as an honest ACP frame.

    The assertions are about the CARRIER, because that is the part a host has to agree with: an empty
    thought chunk whose ``_meta.hermes`` holds the sentence. A client that does not know the extension
    renders nothing, rather than a fake line in the transcript nobody can honestly read.
    """

    def test_status_ships_as_a_meta_marked_thought_chunk(self, mock_conn, event_loop_fixture):
        with patch("acp_adapter.events._send_update") as mock_send:
            cb = make_status_cb(mock_conn, "session-1", event_loop_fixture)
            cb("lifecycle", "👁️ Hindsight — recalled 8 memories")

        mock_send.assert_called_once()
        _conn, session_id, _loop, update = mock_send.call_args.args
        assert session_id == "session-1"
        assert update.sessionUpdate == "agent_thought_chunk"
        assert update.content.text == ""  # nothing is added to the conversation
        # Assert on the WIRE, not on the attribute: `_meta` is only the alias of `field_meta`, so
        # reading back `update._meta` proves nothing about what a host actually receives. This is the
        # assertion that would have caught the status line never shipping (2026-10-17).
        assert _wire(update)["_meta"] == {
            "hermes": {
                "kind": "status",
                "statusKind": "lifecycle",
                "text": "👁️ Hindsight — recalled 8 memories",
            }
        }

    def test_blank_text_sends_nothing(self, mock_conn, event_loop_fixture):
        """An empty status is not an update: a client would otherwise see a new empty thought."""
        with patch("acp_adapter.events._send_update") as mock_send:
            cb = make_status_cb(mock_conn, "session-1", event_loop_fixture)
            cb("lifecycle", "   ")
            cb("lifecycle", "")

        mock_send.assert_not_called()

    def test_unknown_kind_defaults_to_lifecycle(self, mock_conn, event_loop_fixture):
        """Hermes routes warnings through the same channel; an unlabelled one is still a status."""
        with patch("acp_adapter.events._send_update") as mock_send:
            cb = make_status_cb(mock_conn, "session-1", event_loop_fixture)
            cb(None, "compaction starting")

        _conn, _sid, _loop, update = mock_send.call_args.args
        assert _wire(update)["_meta"]["hermes"]["statusKind"] == "lifecycle"

    def test_recall_items_ride_the_same_carrier(self, mock_conn, event_loop_fixture):
        """The items behind 「recalled 8 memories」 are `_meta.hermes.items` — same frame, same
        degradation: a client that never heard of the key draws the line exactly as before."""
        with patch("acp_adapter.events._send_update") as mock_send:
            cb = make_status_cb(mock_conn, "session-1", event_loop_fixture)
            cb(
                "lifecycle",
                "👁️ Hindsight — recalled 2 memories",
                extra={"items": [{"text": "prefers oolong"}, {"text": "works on agentus", "tags": ["work"]}],
                       "truncated": True},
            )

        _conn, _sid, _loop, update = mock_send.call_args.args
        hermes = _wire(update)["_meta"]["hermes"]
        assert hermes["text"] == "👁️ Hindsight — recalled 2 memories"
        assert hermes["items"] == [{"text": "prefers oolong"}, {"text": "works on agentus", "tags": ["work"]}]
        assert hermes["truncated"] is True

    def test_empty_items_add_no_keys(self, mock_conn, event_loop_fixture):
        """Count-only provider → byte-identical frame to before this change."""
        with patch("acp_adapter.events._send_update") as mock_send:
            cb = make_status_cb(mock_conn, "session-1", event_loop_fixture)
            cb("lifecycle", "👁️ Hindsight — recalled 2 memories", extra={"items": [], "truncated": False})

        _conn, _sid, _loop, update = mock_send.call_args.args
        assert _wire(update)["_meta"] == {
            "hermes": {"kind": "status", "statusKind": "lifecycle", "text": "👁️ Hindsight — recalled 2 memories"}
        }

    def test_recall_items_survive_the_real_callback_shape(self, mock_conn, event_loop_fixture):
        """Production does not pass `extra=` at all: `status_output._call_callback` spreads the
        detail dict into keyword args. The envelope tests above stayed green while every live
        frame shipped without items — this one pins the shape the driver actually uses."""
        with patch("acp_adapter.events._send_update") as mock_send:
            cb = make_status_cb(mock_conn, "session-1", event_loop_fixture)
            cb("lifecycle", "👁️ Hindsight — recalled 1 memory",
               items=[{"text": "prefers oolong"}], truncated=False, origin="_emit_status")

        _conn, _sid, _loop, update = mock_send.call_args.args
        hermes = _wire(update)["_meta"]["hermes"]
        assert hermes["items"] == [{"text": "prefers oolong"}]
        assert hermes["truncated"] is False
        assert "origin" not in hermes

    def test_kwargs_from_emit_status_do_not_leak(self, mock_conn, event_loop_fixture):
        """``_emit_status`` calls ``status_callback(kind, message, origin=...)``; origin is Hermes'
        own bookkeeping and must not reach the wire shape a host has to understand."""
        with patch("acp_adapter.events._send_update") as mock_send:
            cb = make_status_cb(mock_conn, "session-1", event_loop_fixture)
            cb("warn", "provider slow", origin="_emit_warning")

        _conn, _sid, _loop, update = mock_send.call_args.args
        assert _wire(update)["_meta"] == {"hermes": {"kind": "status", "statusKind": "warn", "text": "provider slow"}}
