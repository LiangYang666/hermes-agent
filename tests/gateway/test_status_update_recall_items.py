"""The recall items behind 「recalled N memories」 reach a local viewer's status frame.

The sentence is deterministic but unfalsifiable ("recalled 3 memories" — which ones?); the items are
what a cockpit expands. The engine emits them beside the sentence (``_emit_status(..., extra=)``),
``status_output._call_callback`` spreads them as keyword args, and this callback must put them on the
``status.update`` payload — the one place the ACP adapter reads (``make_status_update``).
"""
from types import SimpleNamespace as NS
from unittest.mock import patch

from gateway.run_turn_runner import TurnRunner


def _holder(published):
    ctx = NS(source=NS(platform=None), mute_notification_reply=False, user_config={})
    return NS(_ctx=ctx, _status_live=lambda: True,
              _publish_execution=lambda kind, payload: published.append((kind, payload)))


def test_a_local_viewers_status_frame_carries_the_recall_items():
    published = []
    holder = _holder(published)
    items = [{"text": "prefers oolong"}, {"text": "works on agentus", "tags": ["work"]}]
    with patch("gateway.run_turn_progress.local_viewer_turn", lambda runner: True), \
         patch("gateway.run._prepare_gateway_status_message", lambda *a: "👁️ Hindsight — recalled 2 memories"), \
         patch("gateway.bot_peer_notices.automatic_notices_allowed", lambda source: True), \
         patch("gateway.warning_notifications.render_notification",
               lambda present, **kwargs: present()):
        TurnRunner._status_callback_sync(holder, "lifecycle", "👁️ Hindsight — recalled 2 memories",
                                         items=items, truncated=True)
    assert published == [("status.update", {"kind": "lifecycle",
                                            "text": "👁️ Hindsight — recalled 2 memories",
                                            "items": items, "truncated": True})]


def test_a_status_without_detail_adds_no_keys():
    """A count-only provider keeps the frame byte-identical to what it emitted before."""
    published = []
    holder = _holder(published)
    with patch("gateway.run_turn_progress.local_viewer_turn", lambda runner: True), \
         patch("gateway.run._prepare_gateway_status_message", lambda *a: "compaction progress"), \
         patch("gateway.bot_peer_notices.automatic_notices_allowed", lambda source: True), \
         patch("gateway.warning_notifications.render_notification",
               lambda present, **kwargs: present()):
        TurnRunner._status_callback_sync(holder, "lifecycle", "compaction progress")
    # Re-tagged to the TUI's own compacting kind, and nothing else added.
    assert published == [("status.update", {"kind": "compacting", "text": "compaction progress"})]
