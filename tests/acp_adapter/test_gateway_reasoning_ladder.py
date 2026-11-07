"""The 0.23 config surface offers the ladder the ROUTE takes, not the whole vocabulary.

The picker used to list ``none``..``ultra`` for every model: on a DashScope route whose wire folds
``high``/``xhigh``/``max`` onto one value, the extra rows read as "the setting does nothing". The
ladder is per (provider, model) — the same host serves Qwen (toggle, then low/medium/xhigh on 3.8),
DeepSeek (none..max) and GLM 5.3 (low/high/max, thinking not switchable) — so the advertisement has
to come from ``route_offered_efforts``, on the gateway path as well as the in-process one.
"""

from __future__ import annotations

from acp.schema import ModelInfo, SessionModelState

from acp_adapter.gateway_session_meta import REASONING_CONFIG_ID, config_options


def _models(model_id: str) -> SessionModelState:
    return SessionModelState(current_model_id=model_id,
                             available_models=[ModelInfo(model_id=model_id, name=model_id)])


def _offered(info: dict, model: str) -> list[str]:
    for option in config_options(_models(model), info, "default"):
        if option.id == REASONING_CONFIG_ID:
            return [choice.value for choice in option.options]
    raise AssertionError("no reasoning effort option")


def test_qwen38_offers_the_graded_knob_only() -> None:
    assert _offered({"provider": "alibaba"}, "qwen3.8-flash") == [
        "default", "none", "low", "medium", "xhigh"]


def test_deepseek_route_offers_up_to_max() -> None:
    assert _offered({"provider": "alibaba"}, "deepseek-v4-flash") == [
        "default", "none", "low", "medium", "high", "max"]


def test_glm53_offers_its_narrow_set() -> None:
    assert _offered({"provider": "alibaba"}, "glm-5.3") == ["default", "low", "high", "max"]


def test_a_level_outside_the_route_reads_as_default() -> None:
    """A session pinned to ``high`` on a route that never offers it shows ``default``, not a level the
    picker no longer lists (a value absent from ``options`` is unselectable in every ACP client)."""
    assert _offered({"provider": "alibaba", "reasoning_effort": "high"}, "qwen3.8-flash")[0] == "default"
