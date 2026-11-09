"""How a gateway turn's reply reaches the editor: ``messageId`` grouping, exactly-once final text, tool and
permission bubbles closed as the in-process agent closed them. Mixed into ``GatewayACPAgent``, whose state
(``_conn``, ``_streamed``, ``_segments``, ``_tool_args``, ``_message_ids``, ``_cut``) it reads and updates."""
from __future__ import annotations

import logging

import acp

logger = logging.getLogger(__name__)


def _text(payload):
    """Reply text of a message frame; ``text`` may be present and null (a suppressed delivery)."""
    text = payload.get("text")
    return text if isinstance(text, str) else ""


class GatewayReplyMixin:
    async def _project_tool(self, kind, sid, editor_id, aid, payload):
        """``tool.start`` / ``tool.complete`` as the in-process callbacks projected them (#134889 ACP-15)."""
        from acp_adapter.tools import build_tool_complete, build_tool_start, coerce_tool_args
        # In-process turns publish ``tool_name``; managed workers publish ``name``.
        tool_name = payload.get("tool_name") or payload.get("name") or "tool"
        call_id = payload["tool_call_id"]
        if kind == "tool.start":
            # Deltas before a tool call are commentary the final reply does not repeat: close
            # that segment so ``message.complete`` measures only the final's own stream, and end
            # that assistant message so the text after the tool opens a new one (a new ``messageId``).
            self._close_segment(aid)
            self._end_message(sid)
            args = coerce_tool_args(payload.get("args"))
            self._tool_args[(sid, call_id)] = (tool_name, args, aid)
            if self._conn:
                await self._conn.session_update(session_id=editor_id, update=build_tool_start(call_id, tool_name, args))
            return
        name, args, _ = self._tool_args.pop((sid, call_id), (tool_name, {}, aid))
        result = payload.get("result")
        if self._conn:
            # ``is_error`` is the executor's verdict: a cancelled or failed tool may return plain text the
            # result heuristic reads as success.
            await self._conn.session_update(session_id=editor_id, update=build_tool_complete(
                call_id, name, result=result if isinstance(result, str) else None, function_args=args,
                is_error=payload.get("is_error") is True))

    async def _finish_reply(self, sid, editor_id, aid, payload):
        """Show the settled reply's unseen remainder, then end the turn: its reply's id closes, and a tool
        call it left open (blocked by a guard, an approval denied) ends ``failed``, as the in-process agent
        flushed them at turn end. Shared by ``message.complete`` and a retry's terminal receipt."""
        remainder = self._final_remainder(aid, payload)
        if remainder and self._conn:
            await self._conn.session_update(session_id=editor_id, update=self._chunk(sid, remainder))
        self._end_message(sid)
        await self._flush_open_tools(sid, editor_id, aid)

    async def _flush_open_tools(self, sid, editor_id, aid):
        """Close every call admission *aid* started and never completed (``failed``: it produced no result)."""
        from acp_adapter.tools import build_tool_abandoned
        for key in [key for key, held in self._tool_args.items() if key[0] == sid and held[2] == aid]:
            name, _, _ = self._tool_args.pop(key)
            if self._conn:
                await self._conn.session_update(session_id=editor_id, update=build_tool_abandoned(key[1], name))

    def _chunk(self, sid, text, *, thought=False):
        """An assistant (or thought) chunk carrying the session's current ``messageId``: ACP clients group
        chunks into one reply by it and start a new bubble when it changes."""
        from acp_adapter.events import AssistantMessageIdAllocator
        update = (acp.update_agent_thought_text if thought else acp.update_agent_message_text)(text)
        update.message_id = self._message_ids.setdefault(sid, AssistantMessageIdAllocator()).current()
        return update

    def _end_message(self, sid):
        if (ids := self._message_ids.get(sid)) is not None:
            ids.close()

    def _merged_background(self, sid):
        """这个会话要发的后台行：owner 的进程行 + 本适配层攒的子代理行（#97）。"""
        rows = list(self._bg_rows.get(sid) or [])
        rows.extend((self._subagents.get(sid) or {}).values())
        return rows

    async def _project_subagent(self, kind, sid, editor_id, payload):
        """`subagent.*` → 与 `background.tasks` 同形状的一行（issue #97）。

        上游把子代理的生命周期只发给自己的事件流（`gateway/local_viewer_events._publish_subagent`），
        ACP 侧原先一条都不投影：驾驶舱因此只能看见 shell 进程，看不见被派出去的子代理 —— 而它从前
        是有的。这里按 `subagent_id` 攒一张表，翻成 `kind: "delegation"` 的行（`command` 放目标，
        `exitCode` 由结束状态判定），每来一条就重发整张表，客户端照旧整表替换。
        """
        sub_id = payload.get("subagent_id")
        if not isinstance(sub_id, str) or not sub_id:
            return
        # Only the three lifecycle beats. `subagent.progress` / `.thinking` / `.text` are per-line
        # narration: one whole-table frame each would be a flood, and none of them changes a row.
        if kind not in {"subagent.spawn_requested", "subagent.start", "subagent.complete"}:
            return
        roster = self._subagents.setdefault(sid, {})
        prev = roster.get(sub_id) or {}
        goal = payload.get("goal") or prev.get("command") or sub_id
        count = payload.get("task_count")
        row = {
            "id": f"subagent:{sub_id}",
            "command": str(goal)[:200],
            "pid": None,
            "kind": "delegation",
            # A delegation always owes the session a wake-up: the child's report comes back as a turn.
            "notifyOnComplete": True,
            "status": "running",
            "uptimeSeconds": prev.get("uptimeSeconds") or 0,
            "exitCode": None,
            "taskCount": count if isinstance(count, int) and count > 0 else prev.get("taskCount") or 1,
            "phase": "running",
        }
        if kind == "subagent.complete":
            status = str(payload.get("status") or "")
            duration = payload.get("duration_seconds")
            row["status"] = "exited"
            # 失败状态就三个（SUBAGENT_FAILURE_STATUSES）；其余（completed 等）算成功。三态里的
            # 「没报」在这里不会出现 —— complete 一定带 status，缺了按成功处理会撒谎，所以给 1。
            row["exitCode"] = 1 if status in {"failed", "error", "timeout"} or not status else 0
            row["phase"] = status or "completed"
            if isinstance(duration, (int, float)) and duration >= 0:
                row["uptimeSeconds"] = int(duration)
        roster[sub_id] = row
        if self._conn:
            from acp_adapter.events import make_background_update
            await self._conn.session_update(
                session_id=editor_id, update=make_background_update(self._merged_background(sid)),
            )

    async def _project_side_update(self, kind, sid, editor_id, payload):
        """Turn events outside the reply text, as the in-process server projected them.

        ``reasoning.delta``: the model's reasoning as thought chunks. It never enters the reply's segments,
        so ``message.complete``'s final remainder is unchanged.
        ``todo.updated``: the owner's full todo snapshot (published for a LOCAL session's turn, which an ACP
        session is) as the native ``plan`` update Zed renders as its task panel (#134889 ACP-5).
        ``background.tasks``: the owner's background processes for this session. An ACP viewer cannot
        read the owner's registry, so this frame is the only way a cockpit draws 「N 个任务还在跑」;
        main's ``BackgroundRelay`` drew the same rows (issue #22)."""
        if kind == "reasoning.delta":
            text = _text(payload)
            update = self._chunk(sid, text, thought=True) if text else None
        elif kind == "background.tasks":
            from acp_adapter.events import make_background_update
            # 进程行换成 owner 这一帧的内容，子代理行留着（#97）：两套词汇合成同一个形状，否则后到的
            # 进程快照会把子代理那几行整片盖掉 —— 而它们本来是同一个问题的两个答案。
            self._bg_rows[sid] = payload.get("tasks") or []
            update = make_background_update(self._merged_background(sid))
        elif kind == "status.update":
            # The owner already publishes the status line it shows its own viewers
            # (``run_turn_progress._status_callback_sync``); ACP carries it in ``_meta.hermes``.
            from acp_adapter.events import make_status_update
            update = make_status_update(payload.get("kind"), payload.get("text"),
                                        payload.get("items"), payload.get("truncated"))
        else:
            import json
            from acp_adapter.events import _build_plan_update_from_todo_result
            todos = payload.get("todos")
            update = _build_plan_update_from_todo_result(json.dumps({"todos": todos})) \
                if isinstance(todos, list) else None
        if update is not None and self._conn:
            await self._conn.session_update(session_id=editor_id, update=update)

    def _close_segment(self, aid):
        streamed = self._streamed.pop(aid, "")
        if streamed.strip():
            self._segments.setdefault(aid, []).append(streamed)

    def _final_remainder(self, aid, payload):
        """The part of the settled reply the editor has not been shown yet (exactly-once text)."""
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        text = _text(payload)
        streamed = self._streamed.pop(aid, "")
        shown = self._segments.pop(aid, [])
        cut = self._cut.pop(aid, None)
        # Local interrupt status is metadata; ACP carries it in stop_reason.
        if not text or (payload.get("outcome") == "cancelled"
                        and text.startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX)):
            return ""
        segments = {s.strip() for s in shown}
        if payload.get("response_reused") is True and (
                text.strip() in segments | {streamed.strip()} if cut is not None or payload.get("receipt")
                else (streamed or shown)):
            # The owner names the final as a reply already painted (never inferred from text); after a
            # replay gap or from a receipt, only when this editor showed all of it.
            return ""
        if cut is not None:  # a replay gap stopped its live text: the rest once, or marked in full
            from hermes_cli.gateway_replay_gap import owed_reply
            # An interim after the gap closed the cut segment and may already have shown the whole reply.
            return "" if not cut and text.strip() in segments else owed_reply(cut, text)
        if text.startswith(streamed):
            remainder = text[len(streamed):]
            # An unstreamed final the owner already published as an interim segment.
            return "" if not streamed and text.strip() in {s.strip() for s in shown} else remainder
        if text.strip() == streamed.strip():
            return ""
        return "\n\n" + text

    async def _close_card(self, editor_id, tool_call, options, response):
        """The editor shows a permission request's ``tool_call`` as a pending bubble; once it is answered,
        close it as the in-process agent did (``permissions.await_permission``): ``completed`` for an allow,
        ``failed`` for a reject or a dismissal (#134889 ACP-15)."""
        outcome = getattr(response, "outcome", None)
        allowed = getattr(outcome, "outcome", None) == "selected" and any(
            option.option_id == getattr(outcome, "option_id", None) and option.kind.startswith("allow")
            for option in options)
        try:
            await self._conn.session_update(session_id=editor_id, update=acp.update_tool_call(
                tool_call.tool_call_id, status="completed" if allowed else "failed"))
        except Exception:
            # Boundary: display only; a failed close must never cost the user's answer its delivery.
            logger.debug("ACP permission card %s not closed", tool_call.tool_call_id, exc_info=True)
