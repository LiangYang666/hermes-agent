"""Canonical reasoning-effort vocabulary and wire clamping.

Hermes' internal effort ladder (``VALID_REASONING_EFFORTS`` plus ``none``) is wider than any
single provider wire accepts; hand-rolled per-transport maps leaked new levels (``ultra``) to
wires that 400 and inverted the ladder (unknown → weak default). Single source of truth:
:data:`EFFORT_LADDER` (low→high), :func:`clamp_effort` (verbatim if supported, else the
nearest WEAKER level; only when nothing weaker exists the weakest supported), and named
wire-vocabulary constants so call sites declare data. Rules: wire shape stays local, only the
vocabulary math lives here; unset stays unset (never invent an effort); when a provider
rejects a level fix its declared set, never a predicate.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple, Optional, Sequence

#: Matches ``k3`` as a delimited token (``k3``, ``k3-256k``, ``kimi-k3-cot``), never K2-era names (``kimi-k2.6``).
# From #76427 by @ruizanthony.
_KIMI_K3_SLUG_RE = re.compile(r"(?:^|[^a-z0-9])k3(?:[^a-z0-9]|$)")

# Canonical low→high ordering for nearest-level clamping. Includes "none" so an explicit
# disable can be clamped when a provider publishes it as a level. ``ultra`` is Hermes-internal
# (the Codex product tier): no wire accepts it, every declared set stops at ``max``.
EFFORT_LADDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

#: Widest OpenAI-compatible wire vocabulary (OpenRouter, Nous Portal).
OPENAI_COMPAT_WIRE_EFFORTS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

#: OpenAI/Codex Responses per model generation (live-verified): ``minimal`` is rejected by
#: both (clamps to low); ``max`` is gpt-5.6 / gpt-6-tier only (legacy = 5.5 and older).
CODEX_GPT56_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh", "max")
CODEX_LEGACY_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh")
# GPT-6 Astra is account-gated and its Responses API accepts no disable/minimal
# wire level; callers normalize those requests to ``low`` at the transport boundary.
CODEX_ASTRA_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
ASTRA_MODEL_IDS: frozenset[str] = frozenset({"gpt-6-astra", "gpt-6-astra-900k"})
#: GPT-6 Sol/Terra/Luna (the 5.6 successors; ``-pro``/``-900k``/dated snapshots share the prefix).
GPT6_TIER_PREFIXES: tuple[str, ...] = ("gpt-6-sol", "gpt-6-luna")
#: GPT-6.1 Sol takes Astra's ``low..max`` ladder (``none`` 400s, live 2026-09-29) without Astra's
#: account gating, so it stays in the static catalogs.
NO_DISABLE_TIER_PREFIXES: tuple[str, ...] = ("gpt-6.1-sol",)
DAYBREAK_MODEL_IDS: frozenset[str] = frozenset(
    {"gpt-daybreak-blue-latest", "gpt-daybreak-blue-latest-900k"}
)

#: xAI Responses — Grok 4.6+ accepts xhigh; older Grok tops out at high.
XAI_GROK46_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")
XAI_LEGACY_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: Actual Computer relays (SGLang/vLLM).
ACTUAL_RELAY_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "max")

#: Moonshot/Kimi K3 (server default high) vs K2-era models. K3 quirks: ``high`` is K3's
#: positional middle AND server default, so ``medium`` rounds to it rather than down to
#: ``low``; ``xhigh`` rounds up to ``max`` (K3's top tier).
KIMI_K3_EFFORTS: tuple[str, ...] = ("low", "high", "max")
KIMI_K2_EFFORTS: tuple[str, ...] = ("low", "medium", "high")
KIMI_K3_OVERRIDES: dict[str, str] = {"medium": "high", "xhigh": "max"}

#: OpenCode "Ox Alpha" (x-preview-f-free): thinking cannot be disabled and the wire accepts
#: exactly low/high/max (medium/none/xhigh 400); xhigh rounds up.
OX_ALPHA_EFFORTS: tuple[str, ...] = ("low", "high", "max")
OX_ALPHA_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Tencent TokenHub / Nebius Token Factory / Upstage Solar: plain three-level knobs.
TOKENHUB_EFFORTS: tuple[str, ...] = ("low", "medium", "high")
NEBIUS_EFFORTS: tuple[str, ...] = ("low", "medium", "high")
SOLAR_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: GLM-5.2 native knob: exactly ``high`` (its minimum thinking level) and ``max``; GLM-5.3
#: widens it to a graded scale (live-verified, monotonic). ``xhigh`` requests the top tier.
GLM52_EFFORTS: tuple[str, ...] = ("high", "max")
GLM52_OVERRIDES: dict[str, str] = {"xhigh": "max"}
# : GLM-5.3 widens the knob to a graded low/medium/high/max scale — verified : live on
# api.z.ai/api/coding/paas/v4 (issue #91789, 2026-08-21): every : level accepted with monotonic
# reasoning-token scaling (low=4, medium=11, : high=98, max=125 on the probe prompt).
GLM53_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "max")
GLM53_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: DeepSeek V4 OpenAI-compat endpoint; ``xhigh`` requests the top tier.
DEEPSEEK_V4_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "max")
DEEPSEEK_V4_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Qwen 3.8 family on DashScope (千问/百炼 docs, checked 2026-10-06): thinking is ON by default and
#: ``reasoning_effort`` takes exactly ``low``/``medium``/``xhigh`` — ``high`` and ``max`` are
#: documented aliases of ``xhigh``, and ``minimal`` maps onto ``low``. Thinking is switched off
#: with ``extra_body.enable_thinking=false``, so ``none`` IS a level on this route (it is the
#: toggle), which is why the advertised ladder carries it. Never set ``thinking_budget`` alongside:
#: the docs reject the combination outright.
QWEN38_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "xhigh")
QWEN38_OVERRIDES: dict[str, str] = {"minimal": "low", "high": "xhigh", "max": "xhigh"}

#: Qwen 3.7 / 3.6 / 3.5 hybrids on DashScope: hybrid thinking (``enable_thinking`` switches it on
#: and off), but the GRADED knob ships with 3.8 only. Live-verified 2026-10-07 on the token-plan
#: endpoint: ``reasoning_effort: low`` reasons like no knob at all (qwen3.7-max 60 vs 61 reasoning
#: tokens, qwen3.6-flash 599 vs 576) while ``max`` is an HTTP 400 ("'reasoning_effort' must be one
#: of: 'none', 'minimal', 'low', 'medium', 'high', 'xhigh'"). Advertising 3.8's tiers here is the
#: very "the setting does nothing" bug this vocabulary exists to prevent, so the ladder is ``none``
#: (the off switch) plus one "on" level — which the profile deliberately never puts on the wire.
QWEN_TOGGLE_ONLY_EFFORTS: tuple[str, ...] = ("none", "medium")

#: 阿里云直供 GLM on DashScope. Host-scoped on purpose: the vendor's own API publishes narrower
#: sets for the same ids (``GLM52_EFFORTS``), so neither constant may stand in for the other.
#: glm-5.2/5.1/5 document the full OpenAI ladder (live: 300 reasoning tokens at ``low`` vs 398 at
#: ``max``); glm-5.3 takes exactly low/high/max and ALWAYS thinks — ``enable_thinking: false`` is
#: an HTTP 400 ("The value of the enable_thinking parameter is restricted to True", live-verified
#: 2026-10-07).
DASHSCOPE_GLM_EFFORTS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
DASHSCOPE_GLM53_EFFORTS: tuple[str, ...] = ("low", "high", "max")

#: Levels a wire takes when it also owns the thinking toggle (DeepSeek V4, Qwen 3.8 on DashScope):
#: ``none`` is not an effort, it is ``thinking: disabled`` — but a picker has to offer it as one
#: level or the operator cannot turn thinking off at all.
DEEPSEEK_ROUTE_EFFORTS: tuple[str, ...] = ("none",) + DEEPSEEK_V4_EFFORTS

#: Provider profile names (and their documented aliases) → the levels their wire really accepts.
#: Anything absent keeps the widest OpenAI-compatible vocabulary above: a route we have not
#: verified must not have choices taken away from it, and it clamps again downstream anyway.
_ROUTE_EFFORTS: dict[str, tuple[str, ...]] = {
    "deepseek": DEEPSEEK_ROUTE_EFFORTS,
    "deep-seek": DEEPSEEK_ROUTE_EFFORTS,
    "deepseek-chat": DEEPSEEK_ROUTE_EFFORTS,
    "alibaba": QWEN38_EFFORTS,
    "alibaba-cn": QWEN38_EFFORTS,
    "alibaba-cloud": QWEN38_EFFORTS,
    "alibaba-cloud-cn": QWEN38_EFFORTS,
    "alibaba-token-plan": QWEN38_EFFORTS,
    "alibaba-token-plan-cn": QWEN38_EFFORTS,
    "alibaba-coding-plan": QWEN38_EFFORTS,
    "alibaba-coding-plan-cn": QWEN38_EFFORTS,
    "dashscope": QWEN38_EFFORTS,
    "dashscope-cn": QWEN38_EFFORTS,
    "qwen-dashscope": QWEN38_EFFORTS,
    "aliyun": QWEN38_EFFORTS,
}

#: DashScope-hosted routes. The host itself decides nothing about thinking (one endpoint fronts
#: Qwen, DeepSeek, GLM and Kimi with different knobs), but it DOES own which GLM ladder applies —
#: the vendor's own API publishes a narrower one for the same ids — so the host-scoped table below
#: is consulted only for these routes, and vendor-prefixed ids (``qwen/qwen3.8-flash``) are
#: recognised as the family they name.
_DASHSCOPE_ROUTES: frozenset[str] = frozenset({
    "alibaba", "alibaba-cn", "alibaba-cloud", "alibaba-cloud-cn",
    "alibaba-token-plan", "alibaba-token-plan-cn",
    "alibaba-coding-plan", "alibaba-coding-plan-cn",
    "dashscope", "dashscope-cn", "qwen-dashscope", "aliyun",
})

#: Model ids whose OWN wire is narrower than their host's, so the ladder follows the model rather
#: than the host (a DashScope endpoint serves Qwen, DeepSeek and GLM with different ladders).
#: Longest prefix first: ``qwen3.8`` is the only Qwen generation with the graded knob, so it must
#: resolve before the ``qwen3`` catch-all.
_MODEL_ROUTE_EFFORTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("deepseek-v4", DEEPSEEK_ROUTE_EFFORTS),
    ("qwen3.8", QWEN38_EFFORTS),
    ("qwen-3.8", QWEN38_EFFORTS),
    ("qwen3", QWEN_TOGGLE_ONLY_EFFORTS),
    ("qwen-3", QWEN_TOGGLE_ONLY_EFFORTS),
)

#: DashScope-supplied families whose ladder the HOST decides: the same ``glm-5.3`` id takes
#: low/high/max from 阿里云直供 and a different set from the vendor's API, so only a DashScope route
#: may resolve through here. Ordered longest-prefix-first for the same reason as above.
_DASHSCOPE_MODEL_EFFORTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("glm-5.3", DASHSCOPE_GLM53_EFFORTS),
    ("glm-5", DASHSCOPE_GLM_EFFORTS),
)


def route_offered_efforts(provider: Optional[str], model: Optional[str]) -> tuple[str, ...]:
    """The effort levels to OFFER for one (provider, model) route — what a picker should show.

    Distinct from :func:`route_supported_efforts`, which is the ENTRY clamp and stays deliberately
    wide (the CLI/TUI already accept ``xhigh`` on routes that map it with their own overrides; a
    narrower entry clamp would reject choices those surfaces have always taken). This one is for
    advertising: a cockpit renders exactly these, so a level we offer but the wire folds away is
    the bug the operator reports as "the setting does nothing".

    Model first: the same DashScope host serves Qwen (``low``/``medium``/``xhigh`` on 3.8, toggle
    only before it), DeepSeek (``low``..``max``) and GLM (``low``/``high``/``max`` on 5.3, which
    cannot be switched off at all), so the model id — not the host — decides the ladder. Then the
    provider profile name (plus its aliases), then the widest vocabulary, so an unverified route
    never loses choices.
    """
    p = (provider or "").strip().lower()
    m = (model or "").strip().lower()
    on_dashscope = p in _DASHSCOPE_ROUTES
    # DashScope serves vendor-prefixed ids (``qwen/qwen3.8-flash``); off that host the full id is
    # the only thing we can key on, so the prefix is stripped for DashScope routes alone.
    key = m.rsplit("/", 1)[-1] if (on_dashscope and "/" in m) else m
    if key:
        for prefix, levels in _MODEL_ROUTE_EFFORTS:
            if key.startswith(prefix):
                return levels
    if on_dashscope and key:
        for prefix, levels in _DASHSCOPE_MODEL_EFFORTS:
            if key.startswith(prefix):
                return levels
    return _ROUTE_EFFORTS.get(p, OPENAI_COMPAT_WIRE_EFFORTS)

#: Ollama Cloud /v1/chat/completions: rejects ``minimal`` with HTTP 400.
OLLAMA_CLOUD_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "max")
OLLAMA_CLOUD_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Meta Model API (Muse): rejects ``none``.
META_AI_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")

#: Gemini 4 (``gemini-4-argon``) ``thinkingLevel``: thinking cannot be disabled, ``minimal`` and
#: ``thinkingBudget`` 400 (live on Vertex global, 2026-10-13).
GEMINI4_THINKING_LEVELS: tuple[str, ...] = ("low", "medium", "high")


def is_astra_model(model: Optional[str]) -> bool:
    """``gpt-6-astra`` or its Hermes-side ``-900k`` picker alias, with or without a ``vendor/`` prefix.
    The single home for the slug set: picker gating, effort vocabulary and the request sanitizer all
    key off it, so a new Astra alias is one edit."""
    return (model or "").strip().lower().rsplit("/", 1)[-1] in ASTRA_MODEL_IDS


def codex_supported_efforts(model: Optional[str]) -> tuple[str, ...]:
    """Supported effort set for an OpenAI/Codex Responses model."""
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    if is_astra_model(model) or bare.startswith(NO_DISABLE_TIER_PREFIXES):
        return CODEX_ASTRA_EFFORTS
    return (
        CODEX_GPT56_EFFORTS
        if "gpt-5.6" in bare or bare.startswith(GPT6_TIER_PREFIXES) or bare in DAYBREAK_MODEL_IDS
        else CODEX_LEGACY_EFFORTS
    )


def kimi_supported_efforts(model: Optional[str]) -> tuple[str, ...]:
    """Supported effort set for a Moonshot/Kimi slug (bare ``k3``, ``k3-256k``, ``kimi-k3*`` → K3).

    K3 is served as the bare slug ``k3``, plan variants like ``k3-256k``, and the ``kimi-k3*`` aliases; its
    documented set is low/high/max. Everything earlier speaks low/medium/high. Boundary-matched so K2-era
    names (``kimi-k2.6``) never match (detection regex from #76427 by @ruizanthony).
    """
    m = (model or "").strip().lower().split("/")[-1]
    return KIMI_K3_EFFORTS if _KIMI_K3_SLUG_RE.search(m) else KIMI_K2_EFFORTS


def clamp_effort(
    effort: Optional[str], supported: Optional[Sequence[str]], overrides: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Clamp a requested reasoning effort onto a wire's supported levels.

    ``overrides`` (a declared vendor mapping, e.g. Kimi K3 ``medium → high``) is consulted
    first. Otherwise the request passes through unchanged when it is supported, when the
    supported set is unknown/empty, or when it isn't a recognized ladder level (custom
    providers may use bespoke names). Else the **nearest weaker** supported level is returned
    so a clamp never escalates cost; when nothing weaker exists, the weakest supported level
    (the provider's floor is the closest honest match). Monotonic: a stronger request never
    resolves weaker than a weaker request would.
    """
    requested = str(effort or "").strip().lower()
    if not requested or not supported:
        return effort
    supported_norm = [lvl for lvl in (str(s).strip().lower() for s in supported) if lvl in EFFORT_LADDER]
    if not supported_norm or requested in supported_norm:
        return effort
    if overrides and overrides.get(requested) in supported_norm:
        return overrides[requested]
    if requested not in EFFORT_LADDER:
        return effort
    # "none" disables reasoning — never a degradation target for an enabled ask
    # (clamping "minimal" to "none" would silently switch thinking off).
    candidates = [level for level in supported_norm if level != "none"]
    if not candidates:
        return effort
    requested_idx = EFFORT_LADDER.index(requested)
    below = [level for level in candidates if EFFORT_LADDER.index(level) < requested_idx]
    return max(below, key=EFFORT_LADDER.index) if below else min(candidates, key=EFFORT_LADDER.index)


def route_supported_efforts(
    provider: Optional[str], model: Optional[str], api_mode: Optional[str] = None,
) -> tuple[str, ...]:
    """Levels the (provider, model) route's ENTRY clamp accepts: the Codex/OpenAI Responses set per
    model generation, else the widest OpenAI-compatible vocabulary (narrower providers clamp again
    downstream, never upward). On the Codex app-server runtime ``ultra`` is codex's own harness mode,
    accepted wherever the model's ladder reaches ``max`` (codex runs it at ``max``; a model without
    ``max`` rejects it), so it is sent verbatim there."""
    if (provider or "").strip().lower() == "openai-codex":
        supported = codex_supported_efforts(model)
        return (*supported, "ultra") if api_mode == "codex_app_server" and "max" in supported else supported
    return OPENAI_COMPAT_WIRE_EFFORTS


def effort_display_label(
    effort: Optional[str], provider: Optional[str] = None, model: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> str:
    """Picker / ``/reasoning`` status label for a ladder level: the level itself when the route sends
    it verbatim, else ``"<level> (sends <clamped> on this route)"`` so a Hermes-internal step such as
    ``ultra`` (#61634) is never presented as a distinct wire level the route does not have."""
    requested = str(effort or "").strip().lower()
    clamped = clamp_effort(requested, route_supported_efforts(provider, model, api_mode))
    return requested if not requested or clamped == requested else f"{requested} (sends {clamped} on this route)"


def requested_effort(reasoning_config: Optional[dict]) -> Optional[str]:
    """The user's explicit effort, or None (absent/malformed config, no effort, or reasoning
    disabled) — callers then omit the wire field."""
    if not isinstance(reasoning_config, dict) or reasoning_config.get("enabled") is False:
        return None
    return str(reasoning_config.get("effort") or "").strip().lower() or None


_OFF_REASONING_TOKENS = frozenset({"none", "false", "disabled", "off", "no", "0"})


def explicit_off_reasoning(reasoning_config: Optional[dict]) -> bool:
    """True when a reasoning config asks for thinking OFF, in any spelling Hermes accepts.

    ``enabled`` is the toggle, but operators write the off state in more than one shape: the
    folded form of ``auxiliary.<task>.reasoning_effort: none`` is ``{"enabled": False}``, while a
    hand-written ``extra_body.reasoning.effort: none`` carries no toggle at all, and YAML/JSON
    configs produce ``"false"``/``"off"`` strings. To a route whose profile defaults thinking ON
    (DashScope's ``enable_thinking``, DeepSeek's ``thinking``) those all mean the same thing, and
    reading only the boolean spelling ships the contradiction: an explicit-off config beside an
    explicit-on wire field, with the ON one winning (#52).
    """
    if not isinstance(reasoning_config, dict):
        return False
    if "enabled" in reasoning_config:
        value = reasoning_config.get("enabled")
        if isinstance(value, str):
            if value.strip().lower() not in _OFF_REASONING_TOKENS:
                return False          # an explicitly truthy toggle ("true") wins over any effort
            return True
        if value is not False and value != 0:
            return False
        return True
    for key in ("effort", "level", "thinking_effort"):
        value = reasoning_config.get(key)
        if isinstance(value, str) and value.strip().lower() in _OFF_REASONING_TOKENS:
            return True
    return False


def thinking_off_wire_overrides(extra_body: dict) -> dict:
    """Wire fields that must flip to OFF, for thinking controls already in ``extra_body``.

    The merge seam needs this when the caller/task said "thinking off" but the profile had been
    projected from a config that said nothing: the profile then emitted its route DEFAULT, and for
    DashScope (``enable_thinking``) and DeepSeek (``thinking``) that default is ON — the request
    shipped the operator's off beside the route's on, and the route obeys the on (#52).

    Only keys already on the wire are rewritten, so this can never invent a control the endpoint
    would reject; ``preserve_thinking`` is dropped because it only means something while thinking
    is on. Values already off pass through untouched.
    """
    overrides: dict[str, Any] = {}
    for key, value in (extra_body or {}).items():
        lower = str(key).strip().lower()
        if lower == "enable_thinking":
            if value is not False:
                overrides[key] = False
        elif lower == "thinking" and isinstance(value, dict):
            if str(value.get("type") or "").strip().lower() != "disabled":
                overrides[key] = {**value, "type": "disabled"}
        elif lower == "reasoning" and isinstance(value, dict):
            if value.get("enabled") is not False:
                overrides[key] = {**value, "enabled": False}
    if overrides:
        for key in list((extra_body or {})):
            if str(key).strip().lower() in {"preserve_thinking", "clear_thinking"}:
                overrides[key] = None          # removed by the caller
    return overrides


def tokenhub_effort(requested: Optional[str]) -> str:
    """TokenHub's top-level ``reasoning_effort``: ``high`` when no level was chosen, else the
    request clamped to low/medium/high. Shared by the tencent-tokenhub profile and the host-based
    branch base_url-only agents take; callers handle thinking-off themselves."""
    return "high" if requested is None else clamp_effort(requested, TOKENHUB_EFFORTS)


def generic_nested_reasoning(reasoning_config: Optional[dict]) -> dict:
    """The OpenAI-compatible ``extra_body`` fallback for a route whose profile declares no
    reasoning shape: ``{}`` with no config, ``{"reasoning": {"enabled": False}}`` when disabled,
    else ``{"reasoning": {"enabled": True, "effort": <effort or medium>}}``. Used by auxiliary
    calls, and by a profile hook that wants that fallback (``handles_reasoning`` skips it for any
    profile overriding ``build_api_kwargs_extras``)."""
    if not reasoning_config or not isinstance(reasoning_config, dict):
        return {}
    if reasoning_config.get("enabled") is False:
        return {"reasoning": {"enabled": False}}
    return {"reasoning": {"enabled": True, "effort": reasoning_config.get("effort") or "medium"}}


def clamp_reasoning_config(reasoning_config: Optional[dict], supported: Sequence[str] = OPENAI_COMPAT_WIRE_EFFORTS) -> Optional[dict]:
    """Return ``reasoning_config`` with its ``effort`` clamped onto ``supported`` (non-dicts and
    configs without an effort pass through untouched).

    The entry clamp for an OpenAI-compatible chat-completions request builder: Hermes-internal
    ``ultra`` never reaches a wire (#89503 main transport, #112010 aux/MoA), while provider
    profiles with narrower vocabularies clamp again downstream. Unset stays unset.
    """
    if not isinstance(reasoning_config, dict):
        return reasoning_config
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    clamped = clamp_effort(effort, supported) if effort else effort
    return {**reasoning_config, "effort": clamped} if clamped != effort else reasoning_config


def thinking_toggle_extras(
    reasoning_config: Optional[dict],
    efforts: Sequence[str],
    overrides: Optional[dict[str, str]] = None,
    *,
    always_emit_toggle: bool = False,
) -> tuple[dict, dict]:
    """Translate a reasoning config onto the Moonshot/DeepSeek chat_completions wire:
    ``extra_body.thinking`` toggle and top-level ``reasoning_effort``.

    Moonshot 400s when both are sent, so by default the effort (when it lands in
    ``efforts``) replaces the toggle. DeepSeek instead requires the toggle on every
    request (an omitted toggle defaults thinking on and then demands
    ``reasoning_content`` echoes), hence ``always_emit_toggle``. A requested effort of
    ``none`` is not a level on these wires; it falls back to the plain toggle.
    """
    if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
        return {"thinking": {"type": "disabled"}}, {}
    effort = requested_effort(reasoning_config)
    clamped = clamp_effort(None if effort == "none" else effort, efforts, overrides)
    if clamped in efforts:
        return ({"thinking": {"type": "enabled"}} if always_emit_toggle else {}), {"reasoning_effort": clamped}
    return {"thinking": {"type": "enabled"}}, {}


# DashScope documents ``preserve_thinking`` (Qwen3.6+ and the Kimi K2.6/K2.7 it hosts) and a
# top-level ``clear_thinking`` (its GLM-5.x) as the switches that make the server KEEP replayed
# prior-turn ``reasoning_content``; without them the replay is accepted and dropped.
# https://www.alibabacloud.com/help/en/model-studio/deep-thinking
_DASHSCOPE_PRESERVE_MODELS = ("qwen3.6", "qwen3.7", "qwen3.8", "kimi-k2.6", "kimi-k2.7")


def dashscope_preserve_thinking_extras(model: Optional[str]) -> dict:
    """``extra_body`` keys that keep replayed reasoning on a DashScope-hosted model, else ``{}``."""
    m = (model or "").strip().lower()
    if any(token in m for token in _DASHSCOPE_PRESERVE_MODELS):
        return {"preserve_thinking": True}
    if m.startswith("glm-5"):
        return {"clear_thinking": False}
    return {}


class _DashScopeThinking(NamedTuple):
    """What one DashScope-served family's thinking wire accepts."""

    efforts: Optional[tuple[str, ...]]  # None → the toggle IS the contract (Qwen 3.7/3.6/3.5)
    overrides: Optional[dict[str, str]] = None
    thinking_only: bool = False  # no off switch: ``enable_thinking: false`` is an HTTP 400


#: Model prefix → thinking contract, longest prefix first. One DashScope OpenAI-compatible endpoint
#: fronts several vendors whose knobs genuinely differ, so this is keyed by MODEL (``qwen3.8``
#: ahead of the ``qwen3`` catch-all, ``glm-5.3`` ahead of ``glm-5``). A family absent here gets NO
#: parameter at all: an unwarranted ``enable_thinking: false`` 400s on the thinking-only ids we have
#: not enumerated, and inventing an effort a family does not read is the silent no-op (an operator
#: picking a level that changes nothing) this module exists to prevent. Families deliberately left
#: out because their contract differs by SUPPLIER, not just by name — 阿里直供 kimi-k3 documents
#: low/high/max while 月之暗面直供 kimi-k3 documents ``max`` alone, MiniMax uses
#: ``thinking: adaptive|disabled`` instead of these two parameters, and Stepfun defaults to OFF —
#: get no parameter at all rather than a guess that could 400.
_DASHSCOPE_THINKING: tuple[tuple[str, _DashScopeThinking], ...] = (
    ("deepseek-v4", _DashScopeThinking(DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES)),
    ("qwen3.8", _DashScopeThinking(QWEN38_EFFORTS, QWEN38_OVERRIDES)),
    ("qwen-3.8", _DashScopeThinking(QWEN38_EFFORTS, QWEN38_OVERRIDES)),
    ("qwen3", _DashScopeThinking(None)),
    ("qwen-3", _DashScopeThinking(None)),
    ("glm-5.3", _DashScopeThinking(DASHSCOPE_GLM53_EFFORTS, thinking_only=True)),
    ("glm-5", _DashScopeThinking(DASHSCOPE_GLM_EFFORTS)),
)


def dashscope_model_thinking(model: Optional[str]) -> Optional[_DashScopeThinking]:
    """Thinking contract for a DashScope-served model id, or None for a family we have not verified."""
    m = (model or "").strip().lower()
    if "/" in m:  # vendor-prefixed ids (``qwen/qwen3.8-flash``)
        m = m.rsplit("/", 1)[-1]
    for prefix, spec in _DASHSCOPE_THINKING:
        if m.startswith(prefix):
            return spec
    return None


def dashscope_thinking_extras(reasoning_config: Optional[dict], model: Optional[str]) -> dict[str, Any]:
    """Translate a reasoning config onto DashScope's ``extra_body`` thinking parameters.

    The knob translation (``_dashscope_thinking_wire``) is the fork's; on top of it upstream's
    ``preserve_thinking`` switch is merged in, because the server silently DROPS replayed
    prior-turn ``reasoning_content`` without it. The two are orthogonal: the knob says how hard to
    think this turn, the switch says whether the earlier turns' thinking is readable at all.
    Upstream owns the model list, so the merge can never add the parameter to a family upstream
    has not verified.

    Two deviations from upstream, both deliberate:

    * when THIS turn explicitly turns thinking off we do not ask the server to preserve historical
      thinking — the request would contradict itself, and upstream never exercises the combination
      (its profile emits no ``enable_thinking`` at all);
    * we carry ONLY ``preserve_thinking``, never upstream's sibling ``clear_thinking: false``,
      which is already the server's documented default on the GLM family (``false`` = retain), so
      sending it changes nothing — while its support list (glm-5.2/5.1/5/4.7) excludes glm-5.3,
      which upstream's ``glm-5`` prefix matches anyway. An inert key is not worth a
      rejected-parameter risk.
    """
    extras = _dashscope_thinking_wire(reasoning_config, model)
    preserve = dashscope_preserve_thinking_extras(model)
    if extras.get("enable_thinking") is not False and preserve.get("preserve_thinking"):
        extras["preserve_thinking"] = True
    return extras


def _dashscope_thinking_wire(reasoning_config: Optional[dict], model: Optional[str]) -> dict[str, Any]:
    """The fork's per-family knob translation for DashScope's ``extra_body``.

    ``enable_thinking`` is the only real off switch and ``reasoning_effort`` carries the grade,
    clamped onto what the family's wire takes; ``thinking_budget`` is mutually exclusive with
    ``reasoning_effort`` and is never emitted. Two per-family rules live here:

    * a toggle-only family (Qwen 3.7/3.6/3.5) never sees an effort — its wire has none — so the
      picker's "on" level resolves to a bare ``enable_thinking: true``;
    * a thinking-only family (glm-5.3) never sees ``false`` (HTTP 400). An "off" ask lands on its
      lightest level, the closest honest match: live, glm-5.3 at ``low`` returns zero reasoning
      tokens.

    Shared by every DashScope profile (``alibaba``, its token-plan/coding-plan tiers), which is why
    it lives here next to the ladders it clamps onto rather than in one of them.
    """
    spec = dashscope_model_thinking(model)
    if spec is None:
        return {}
    disabled = isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False
    ask = requested_effort(reasoning_config)
    extras: dict[str, Any] = {"enable_thinking": True}
    if spec.thinking_only:
        efforts = spec.efforts or ()
        if disabled or ask == "none":
            ask = "low"
        clamped = clamp_effort(ask, efforts, spec.overrides)
        if clamped in efforts:
            extras["reasoning_effort"] = clamped
        return extras
    if disabled:
        return {"enable_thinking": False}
    if spec.efforts is None:  # toggle-only family: the picked "on" level rides nothing
        return extras
    clamped = clamp_effort(None if ask == "none" else ask, spec.efforts, spec.overrides)
    if clamped in spec.efforts and clamped != "none":
        extras["reasoning_effort"] = clamped
    return extras


def ox_alpha_reasoning_extras(reasoning_config: Optional[dict], model: Optional[str]) -> tuple[dict, dict]:
    """Ox Alpha (``x-preview-f-free``) ``reasoning_effort`` translation for the
    opencode-zen profile (low/high/max only; anything else 400s)."""
    if (model or "").strip().rsplit("/", 1)[-1].lower() != "x-preview-f-free":
        return {}, {}
    effort = requested_effort(reasoning_config)
    clamped = clamp_effort(None if effort == "none" else effort, OX_ALPHA_EFFORTS, OX_ALPHA_OVERRIDES)
    return ({}, {"reasoning_effort": clamped}) if clamped in OX_ALPHA_EFFORTS else ({}, {})
