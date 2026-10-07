"""Alibaba Cloud DashScope provider profiles (intl + CN, plus the Model Studio
Token Plan flat-token tier with its own key/endpoints — one module per vendor).

Profile names match models.dev catalog keys exactly so model metadata lines up
and ``model.provider: alibaba-cn`` resolves at runtime.

Thinking on this wire (千问/百炼 docs + live probes, 2026-10-06/07): every DashScope
OpenAI-compatible profile answers to the SAME two ``extra_body`` parameters, and neither is a
standard OpenAI field:

``enable_thinking``
    ``true``/``false``. Qwen 3.8 (``qwen3.8-flash``/``-max``/``-omni-flash``) and DeepSeek V4 both
    default to thinking ON, so without this parameter the model always reasons — which is exactly
    the operator's "关掉它，它还是有思考". ``false`` is rejected by thinking-only models (the
    glm-5.3 series), which is why an unknown family is left untouched below.

``reasoning_effort``
    The graded knob, with a per-FAMILY vocabulary: Qwen 3.8 takes exactly ``low``/``medium``/
    ``xhigh``, DeepSeek V4 ``low``..``max``, 阿里云直供 glm-5.2 the full ladder and glm-5.3 exactly
    ``low``/``high``/``max`` — while Qwen 3.7/3.6/3.5 have no graded knob at all. Sending a level
    the family does not know is the silent no-op this wiring exists to prevent, so the ladders and
    the translation (``dashscope_thinking_extras``, shared with the coding-plan tier) both live in
    ``agent.reasoning_effort``, and a cockpit advertises ``route_offered_efforts`` straight from
    them.

``thinking_budget`` (never set here)
    A token cap that is mutually exclusive with ``reasoning_effort`` — the API errors when both
    arrive. We drive the knob by effort only, so the two can never collide.
"""

from typing import Any

from agent.reasoning_effort import dashscope_thinking_extras
from providers import register_provider
from providers.base import ProviderProfile


class DashScopeProfile(ProviderProfile):
    """One DashScope OpenAI-compatible endpoint: thinking parameters ride ``extra_body``.

    The translation itself is shared (``agent.reasoning_effort.dashscope_thinking_extras``) because
    the token-plan and coding-plan tiers are the same wire under a different key/endpoint — one
    profile per module, one thinking contract for all of them.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return dashscope_thinking_extras(reasoning_config, model), {}


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
