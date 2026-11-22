"""ACP prompt-response usage must carry the FULL prompt plus BOTH cache buckets (issue #71).

The agent's session counters split the prompt: ``input_tokens`` is FRESH (normalize_usage
subtracts cache read + write) and ``prompt_tokens`` is the full sum. ACP's ``Usage.input_tokens``
is what the cockpit divides cached reads by for its hit rate — the cache buckets are a SUBSET of
the prompt, not peers beside it — so the adapter must forward ``prompt_tokens``. The 0.23 gateway
path forwarded the fresh counter instead, making read/fresh exceed 100% on every cache-heavy turn
and pinning the cockpit's hit rate at a clamped 100%. Pinned here because the field is typed `int`
in ACP's schema: nothing fails, the cockpit just divides by the wrong number.

Both cache buckets are forwarded too, and an unreported bucket stays ``None``: a client that is
told "read only" can neither reconcile the prompt total nor bill the writes (the 1.25x kind), and
an unreported write rendered as a measured zero reads as free.
"""

from __future__ import annotations

from acp_adapter.gateway_session_meta import turn_usage


def test_prompt_response_carries_full_prompt_and_both_cache_buckets():
    # REAL usage_stats shape: input_tokens FRESH 1000, prompt_tokens FULL 4500
    # (fresh 1000 + read 3000 + write 500). ACP input_tokens must be 4500, not 1000.
    usage = turn_usage({
        "input_tokens": 1000, "prompt_tokens": 4500, "output_tokens": 120, "total_tokens": 4620,
        "cache_read_tokens": 3000, "cache_write_tokens": 500,
    })
    assert usage is not None
    assert usage.input_tokens == 4500, "full prompt, not the fresh counter"
    assert usage.cached_read_tokens == 3000
    assert usage.cached_write_tokens == 500


def test_hit_rate_is_read_over_full_prompt_not_fresh():
    """The regression: read / input_tokens must be <= 100% with the full prompt."""
    usage = turn_usage({
        "input_tokens": 1000, "prompt_tokens": 4500, "output_tokens": 120, "total_tokens": 4620,
        "cache_read_tokens": 3000, "cache_write_tokens": 500,
    })
    assert usage is not None
    assert usage.cached_read_tokens is not None
    pct = round(usage.cached_read_tokens / usage.input_tokens * 100)
    assert pct == 67, f"read/full prompt should be 67%, got {pct}%"


def test_falls_back_to_input_when_prompt_tokens_absent():
    """Counters that predate the fresh/full split still report something sane."""
    usage = turn_usage({
        "input_tokens": 4500, "output_tokens": 120, "total_tokens": 4620,
        "cache_read_tokens": 3000, "cache_write_tokens": 500,
    })
    assert usage is not None
    assert usage.input_tokens == 4500


def test_numeric_prompt_tokens_normalized_but_bool_rejected():
    """A float prompt total is normalized to int; True (a bool) is not a prompt total."""
    u = turn_usage({
        "input_tokens": 1000, "prompt_tokens": 4500.0, "output_tokens": 120,
        "total_tokens": 4620, "cache_read_tokens": 3000,
    })
    assert u is not None and u.input_tokens == 4500
    b = turn_usage({
        "input_tokens": 900, "prompt_tokens": True, "output_tokens": 10,
        "total_tokens": 910, "cache_read_tokens": 50,
    })
    assert b is not None and b.input_tokens == 900, "True must fall back, not become 1"


def test_unreported_write_stays_none_not_zero():
    """An agent/provider that reports no write counter must not look like a measured zero."""
    usage = turn_usage({
        "input_tokens": 400, "prompt_tokens": 1200, "output_tokens": 40, "total_tokens": 1240,
        "cache_read_tokens": 800,
    })
    assert usage is not None
    assert usage.input_tokens == 1200
    assert usage.cached_read_tokens == 800
    assert usage.cached_write_tokens is None


def test_no_stats_is_no_usage():
    assert turn_usage(None) is None
