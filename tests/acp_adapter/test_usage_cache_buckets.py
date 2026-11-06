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

from acp_adapter.gateway_session_meta import turn_usage


def test_prompt_response_carries_both_cache_buckets():
    # the owner's session counters: prompt_tokens is the FULL prompt (fresh 1000 + read 3000 + write 500).
    usage = turn_usage({
        "input_tokens": 4500, "output_tokens": 120, "total_tokens": 4620,
        "cache_read_tokens": 3000, "cache_write_tokens": 500,
    })
    assert usage is not None
    assert usage.input_tokens == 4500
    assert usage.cached_read_tokens == 3000
    assert usage.cached_write_tokens == 500


def test_unreported_write_stays_none_not_zero():
    """An agent/provider that reports no write counter must not look like a measured zero."""
    usage = turn_usage({
        "input_tokens": 1200, "output_tokens": 40, "total_tokens": 1240,
        "cache_read_tokens": 800,
    })
    assert usage is not None
    assert usage.cached_read_tokens == 800
    assert usage.cached_write_tokens is None


def test_no_stats_is_no_usage():
    assert turn_usage(None) is None
