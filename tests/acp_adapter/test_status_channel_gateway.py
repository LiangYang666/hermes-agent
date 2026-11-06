"""The status line reaches an ACP host as a ``_meta.hermes`` frame (fork feature).

The owner (gateway) publishes ``status.update`` for its local viewers — the same events the CLI
prints, e.g. ``👁️ Hindsight — recalled 8 memories`` or a compaction countdown. ACP has no status
object, so the projection carries the sentence in ``_meta`` on an EMPTY thought chunk: a client
that does not know the extension renders nothing instead of a fake line in the transcript.

The consumer side of this contract is the cockpit's `statusLineOf`/`statusItemsOf`; the shapes are
pinned here because ``_meta`` is optional in the schema — nothing fails if it goes wrong, the
status line (and the recall detail behind it) just silently never arrives.
"""

from __future__ import annotations

import asyncio

from acp_adapter.events import make_status_update
from acp_adapter.gateway_reply import GatewayReplyMixin


class _Conn:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update))


class _Agent(GatewayReplyMixin):
    """Only the status projection: that path touches nothing but ``_conn``."""

    def __init__(self, conn):
        self._conn = conn


def _project(payload, kind="status.update"):
    conn = _Conn()
    agent = _Agent(conn)
    asyncio.run(agent._project_side_update(kind, "s1", "e1", payload))
    return conn.updates


def test_status_line_rides_meta_on_an_empty_thought():
    updates = _project({"kind": "lifecycle", "text": "👁️ Hindsight — recalled 8 memories"})
    assert len(updates) == 1
    session_id, update = updates[0]
    assert session_id == "e1"
    wire = update.model_dump(by_alias=True)
    assert wire["sessionUpdate"] == "agent_thought_chunk"
    assert wire["content"]["text"] == "", "a status line must not paint text of its own"
    hermes = wire["_meta"]["hermes"]
    assert hermes["kind"] == "status"
    assert hermes["statusKind"] == "lifecycle"
    assert hermes["text"].startswith("👁️ Hindsight")
    assert "items" not in hermes, "no empty keys on the wire when the provider reports only a count"


def test_empty_text_is_not_sent():
    assert make_status_update("lifecycle", "   ") is None
    assert _project({"kind": "lifecycle", "text": ""}) == []


def test_recall_items_and_the_truncation_flag_travel_in_the_same_frame():
    items = [{"text": "user is on macOS 26.2", "tags": ["env"]}, {"text": "prefers Chinese"}]
    updates = _project({"kind": "memory", "text": "👁️ recalled 2 memories", "items": items,
                        "truncated": True})
    hermes = updates[0][1].model_dump(by_alias=True)["_meta"]["hermes"]
    assert hermes["items"] == items
    assert hermes["truncated"] is True
    assert hermes["statusKind"] == "memory"


def test_compaction_kind_survives():
    hermes = _project({"kind": "compacting", "text": "compacting context…"})[0][1].model_dump(
        by_alias=True)["_meta"]["hermes"]
    assert hermes["statusKind"] == "compacting"
