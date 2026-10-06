"""Alibaba Cloud DashScope provider profiles (intl + CN, plus the Model Studio
Token Plan flat-token tier with its own key/endpoints — one module per vendor).

Profile names match models.dev catalog keys exactly so model metadata lines up
and ``model.provider: alibaba-cn`` resolves at runtime.

Thinking on this wire (checked against the 千问/百炼 docs, 2026-10-06): every DashScope
OpenAI-compatible profile answers to the SAME two ``extra_body`` parameters, and neither is a
standard OpenAI field:

``enable_thinking``
    ``true``/``false``. Qwen 3.8 (``qwen3.8-flash``/``-max``/``-omni-flash``) and DeepSeek V4 both
    default to thinking ON, so without this parameter the model always reasons — which is exactly
    the operator's "关掉它，它还是有思考". ``false`` is rejected by thinking-ONLY models (the
    glm-5.3 series, kimi-k3 on Alibaba), which is why an unknown family is left untouched below.

``reasoning_effort``
    The graded knob, with per-family values: Qwen 3.8 takes exactly ``low``/``medium``/``xhigh``
    (``high``/``max`` are documented aliases of ``xhigh``, ``minimal`` of ``low``); DeepSeek V4
    takes ``low``..``max``. Sending a level the family does not know is the silent no-op this
    module exists to prevent — the ladders live in ``agent.reasoning_effort`` (``QWEN38_EFFORTS`` /
    ``DEEPSEEK_V4_EFFORTS``) and a cockpit advertises ``route_offered_efforts`` straight from them.

``thinking_budget`` (never set here)
    A token cap that is mutually exclusive with ``reasoning_effort`` — the API errors when both
    arrive. We drive the knob by effort only, so the two can never collide.
"""

from typing import Any

from agent.reasoning_effort import (
    DEEPSEEK_V4_EFFORTS,
    DEEPSEEK_V4_OVERRIDES,
    QWEN38_EFFORTS,
    QWEN38_OVERRIDES,
    clamp_effort,
    requested_effort,
)
from providers import register_provider
from providers.base import ProviderProfile

#: Model prefix → which thinking ladder its wire speaks. One DashScope endpoint fronts several
#: vendors (Qwen, DeepSeek, GLM, Kimi) with different knobs, so the MODEL decides, never the host.
#: Anything absent sends no thinking parameter at all: a family we have not verified must not be
#: handed a ``false`` toggle it would reject outright.
_THINKING_FAMILIES: tuple[tuple[str, str], ...] = (
    ("qwen3", "qwen"),
    ("qwen-3", "qwen"),
    ("deepseek-v4", "deepseek"),
)


def _thinking_family(model: str) -> str | None:
    """``"qwen"`` / ``"deepseek"`` for a thinking-capable family we know, else None."""
    m = (model or "").strip().lower()
    if "/" in m:  # vendor-prefixed ids (``qwen/qwen3.8-flash``)
        m = m.rsplit("/", 1)[-1]
    for prefix, family in _THINKING_FAMILIES:
        if m.startswith(prefix):
            return family
    return None


def dashscope_thinking_extras(
    reasoning_config: dict | None,
    efforts: tuple[str, ...],
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Translate Hermes' reasoning config onto DashScope's ``extra_body`` thinking parameters.

    ``{"enabled": False}`` → ``enable_thinking: false``: the only way to actually stop the
    reasoning. A level → ``enable_thinking: true`` plus ``reasoning_effort`` clamped to what the
    family takes (``none`` is not an effort on this wire either — it is the toggle). An
    unrecognised effort still pins the toggle on explicitly, matching the API's own default while
    making the request self-describing, and never emits ``thinking_budget`` (mutually exclusive).
    """
    if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
        return {"enable_thinking": False}
    effort = requested_effort(reasoning_config)
    clamped = clamp_effort(None if effort == "none" else effort, efforts, overrides)
    extras: dict[str, Any] = {"enable_thinking": True}
    if clamped in efforts and clamped != "none":
        extras["reasoning_effort"] = clamped
    return extras


class DashScopeProfile(ProviderProfile):
    """One DashScope OpenAI-compatible endpoint: thinking parameters ride ``extra_body``."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        family = _thinking_family(model or "")
        if family is None:
            return {}, {}
        if family == "qwen":
            return dashscope_thinking_extras(reasoning_config, QWEN38_EFFORTS, QWEN38_OVERRIDES), {}
        return dashscope_thinking_extras(reasoning_config, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES), {}


alibaba = DashScopeProfile(
    name="alibaba", aliases=("dashscope", "alibaba-cloud", "qwen-dashscope", "aliyun"), env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

alibaba_cn = DashScopeProfile(
    name="alibaba-cn", aliases=("dashscope-cn", "alibaba-cloud-cn"),
    display_name="Alibaba Cloud DashScope (China)",
    description="Alibaba Cloud DashScope, mainland-China endpoint",
    env_vars=("DASHSCOPE_API_KEY", "DASHSCOPE_CN_BASE_URL"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

alibaba_token_plan = DashScopeProfile(
    name="alibaba-token-plan", aliases=("dashscope-token-plan",), display_name="Alibaba Cloud (Token Plan)",
    description="Alibaba Cloud Model Studio Token Plan (flat-token tier)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_TOKEN_PLAN_API_KEY", "ALIBABA_TOKEN_PLAN_BASE_URL"),
    base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", auth_type="api_key",
)

alibaba_token_plan_cn = DashScopeProfile(
    name="alibaba-token-plan-cn", aliases=("dashscope-token-plan-cn",),
    display_name="Alibaba Cloud (Token Plan, China)",
    description="Alibaba Cloud Model Studio Token Plan, mainland-China endpoint",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_TOKEN_PLAN_CN_API_KEY", "ALIBABA_TOKEN_PLAN_API_KEY", "ALIBABA_TOKEN_PLAN_CN_BASE_URL"),
    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", auth_type="api_key",
)

register_provider(alibaba)
register_provider(alibaba_cn)
register_provider(alibaba_token_plan)
register_provider(alibaba_token_plan_cn)
