"""Alibaba Cloud Coding Plan provider profiles (intl + CN): a dedicated endpoint
and key tier separate from ``alibaba``. Names match models.dev catalog keys.

The CN profile checks its own key first and keeps the shared vars as ordered
fallbacks so existing CN users configured with the shared key keep working.

Thinking: the Coding Plan endpoints are the SAME DashScope OpenAI-compatible wire as the
``alibaba`` profiles, so they answer to the same ``extra_body.enable_thinking`` /
``reasoning_effort`` translation — shared from ``agent.reasoning_effort``. Registering these as
plain ``ProviderProfile``s is what made every thinking-depth pick a no-op (and even "off" keep
reasoning) on this route: a bare profile emits no thinking parameter at all (reported 2026-10-07).
"""

from __future__ import annotations

from typing import Any

from agent.reasoning_effort import dashscope_thinking_extras
from providers import register_provider
from providers.base import ProviderProfile


class CodingPlanProfile(ProviderProfile):
    """One DashScope Coding Plan endpoint: thinking parameters ride ``extra_body``."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return dashscope_thinking_extras(reasoning_config, model), {}


alibaba_coding_plan = CodingPlanProfile(
    name="alibaba-coding-plan", aliases=("alibaba_coding", "alibaba-coding", "dashscope-coding"),
    display_name="Alibaba Cloud (Coding Plan)",
    description="Alibaba Cloud Coding Plan (Dedicated coding tier)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_CODING_PLAN_BASE_URL"),
    base_url="https://coding-intl.dashscope.aliyuncs.com/v1", auth_type="api_key",
)

alibaba_coding_plan_cn = CodingPlanProfile(
    name="alibaba-coding-plan-cn", aliases=("alibaba-coding-cn", "dashscope-coding-cn"),
    display_name="Alibaba Cloud (Coding Plan, China)",
    description="Alibaba Cloud Coding Plan, mainland-China endpoint",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_CODING_PLAN_CN_API_KEY", "ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_CODING_PLAN_CN_BASE_URL"),
    base_url="https://coding.dashscope.aliyuncs.com/v1", auth_type="api_key",
)

register_provider(alibaba_coding_plan)
register_provider(alibaba_coding_plan_cn)
