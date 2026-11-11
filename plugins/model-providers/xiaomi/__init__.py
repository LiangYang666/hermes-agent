"""Xiaomi MiMo provider profile.

MiMo turns thinking ON by default when the request carries no knob at all, so a session's
reasoning setting used to be silently ignored: ``minimal``/``none`` both burned tens of thousands
of reasoning tokens per turn (#100). The wire takes a graded top-level ``reasoning_effort``
(``none``/``low``/``medium``/``high`` only — ``minimal`` is an HTTP 400) and an
``extra_body.thinking`` toggle; this profile puts the operator's setting on both.
"""

from typing import Any

from agent.reasoning_effort import MIMO_EFFORTS, clamp_effort, explicit_off_reasoning, requested_effort
from providers import register_provider
from providers.base import ProviderProfile


class XiaomiProfile(ProviderProfile):
    """Xiaomi MiMo — graded ``reasoning_effort``, plus the toggle for the off state."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Off is a toggle, not a level: MiMo accepts ``reasoning_effort: "none"`` too, but
        # ``thinking: disabled`` is the form the vendor documents and the one that never depends
        # on the effort field being read at all. Surfaces report Off through the effort field
        # while ``enabled`` is still true (the folded picker state), so the effort spelling has to
        # count as off too — same read as the DeepSeek profile (#107238).
        effort_field = ""
        if isinstance(reasoning_config, dict):
            effort_field = str(reasoning_config.get("effort") or "").strip().lower()
        if explicit_off_reasoning(reasoning_config) or effort_field in {"none", "false", "disabled", "off"}:
            return {"thinking": {"type": "disabled"}}, {}
        effort = clamp_effort(requested_effort(reasoning_config), MIMO_EFFORTS)
        # Unset config → send nothing and keep MiMo's own default (thinking on), as before.
        # A requested level outside the MiMo vocabulary is clamped to the nearest WEAKER one
        # (``xhigh``/``max`` → ``high``); ``minimal`` has no weaker "on" level, so it lands on
        # ``low`` — ``clamp_effort`` never degrades an enabled ask into the off state.
        if effort and effort != "none":
            return {}, {"reasoning_effort": effort}
        return {}, {}


xiaomi = XiaomiProfile(
    name="xiaomi", aliases=("mimo", "xiaomi-mimo"), env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
)

register_provider(xiaomi)
