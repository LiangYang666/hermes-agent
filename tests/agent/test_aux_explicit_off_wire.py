"""Explicit "thinking off" must reach the wire as OFF, on every route (#52).

Two shapes of the same operator instruction:

* ``auxiliary.<task>.reasoning_effort: none`` — folded by ``_get_task_extra_body`` into
  ``extra_body.reasoning = {"enabled": False}``;
* a hand-written ``auxiliary.<task>.extra_body.reasoning`` — which may say ``{"effort": "none"}``
  or ``{"enabled": "false"}`` and carry no boolean toggle at all.

Before this fix only the first spelling reached the projection; the others stayed in the body as
an inert ``reasoning`` key while the profile was projected from ``reasoning_config=None`` and
emitted its route default — ``enable_thinking: True`` on DashScope, ``thinking: {"type":
"enabled"}`` on DeepSeek. The route obeys the ON field, so the user's off did nothing.

The invariant asserted here is the one that matters and is provider-independent: when the config
says off, no field on the final ``extra_body`` may say ON.
"""

import types

import pytest

import agent.auxiliary_client as aux

DASHSCOPE = ("alibaba", "qwen3.8-flash", "https://dashscope.aliyuncs.com/compatible-mode/v1/")
DEEPSEEK = ("deepseek", "deepseek-v4-flash", "https://api.deepseek.com/v1")
OPENROUTER = ("openrouter", "anthropic/claude-sonnet-4.5", "https://openrouter.ai/api/v1")

# Every spelling an operator (or a config file) actually produces for "off".
OFF_SPELLINGS = [
    {"enabled": False},
    {"effort": "none"},
    {"enabled": "false"},
    {"enabled": "off"},
    {"level": "disabled"},
]


def _on_markers(body):
    """Fields that ask the route to think, as they would travel."""
    found = []
    if body.get("enable_thinking") is True:
        found.append("enable_thinking=True")
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and str(thinking.get("type") or "").lower() == "enabled":
        found.append("thinking.type=enabled")
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("enabled") is True:
        found.append("reasoning.enabled=True")
    return found


def _build(route, extra_body):
    provider, model, base = route
    return aux._build_call_kwargs(
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        extra_body=dict(extra_body),
        task="compression",
        base_url=base,
    )


@pytest.mark.parametrize("route", [DASHSCOPE, DEEPSEEK, OPENROUTER], ids=["dashscope", "deepseek", "openrouter"])
@pytest.mark.parametrize("spelling", OFF_SPELLINGS, ids=lambda s: ",".join(f"{k}={v}" for k, v in s.items()))
def test_folded_off_never_ships_a_thinking_on_field(route, spelling):
    kwargs = _build(route, {"reasoning": dict(spelling)})
    body = kwargs.get("extra_body") or {}

    assert _on_markers(body) == [], f"{route[0]} shipped ON beside an explicit OFF: {body}"
    assert kwargs.get("reasoning_effort") is None


def test_dashscope_off_is_the_toggle_it_reads():
    body = _build(DASHSCOPE, {"reasoning": {"effort": "none"}}).get("extra_body")
    assert body["enable_thinking"] is False
    # preserve_thinking only means something while thinking is on; it must not ride along.
    assert "preserve_thinking" not in body


def test_deepseek_off_is_the_native_disabled_toggle():
    """The provider+model pair we run day to day: DeepSeek's wire wants ``thinking.type``."""
    body = _build(DEEPSEEK, {"reasoning": {"effort": "none"}}).get("extra_body")
    assert body == {"thinking": {"type": "disabled"}}


def test_explicit_on_is_left_alone():
    """Over-normalising would be a regression: an 'on' config keeps its thinking field."""
    dashscope = _build(DASHSCOPE, {"reasoning": {"effort": "low"}}).get("extra_body")
    assert dashscope["enable_thinking"] is True
    assert dashscope["reasoning"] == {"effort": "low"}

    deepseek = _build(DEEPSEEK, {"reasoning": {"effort": "low"}}).get("extra_body")
    assert deepseek["thinking"] == {"type": "enabled"}


def test_merge_seam_treats_a_folded_off_as_disabled():
    """The issue's own repro: called directly, with a projection made from ``None``."""
    for route in (DASHSCOPE, DEEPSEEK):
        provider, model, base = route
        projection = aux._project_provider_profile(provider, provider, model, base, None)
        merged = aux._merge_aux_extra_body({"reasoning": {"enabled": False}}, projection, None, provider)
        assert _on_markers(merged) == [], f"{provider} still ships ON: {merged}"


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content="a summary", reasoning_content=None, tool_calls=None)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model=kwargs.get("model"),
        )


class _FakeClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.completions = _FakeCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)


@pytest.mark.parametrize("route", [DASHSCOPE, DEEPSEEK], ids=["dashscope", "deepseek"])
def test_compression_call_keeps_the_off_through_the_fast_lane(monkeypatch, route):
    """The compression task has its own lane stage after the build; it must not re-open thinking."""
    provider, model, base = route
    client = _FakeClient(base)
    monkeypatch.setattr(aux, "_resolve_call_client", lambda task, **kw: (client, model, provider, provider))

    aux.call_llm(
        task="compression",
        messages=[{"role": "user", "content": "summarise"}],
        extra_body={"reasoning": {"effort": "none"}},
        main_runtime={"model": model, "provider": provider, "base_url": base, "api_key": "test", "api_mode": None},
    )

    body = client.completions.calls[-1].get("extra_body") or {}
    assert _on_markers(body) == [], f"{provider} compression shipped ON: {body}"
    assert body.get("enable_thinking") is False if provider == "alibaba" else body == {"thinking": {"type": "disabled"}}


def test_explicit_off_reasoning_reads_every_spelling():
    from agent.reasoning_effort import explicit_off_reasoning

    for spelling in OFF_SPELLINGS:
        assert explicit_off_reasoning(dict(spelling)) is True, spelling
    assert explicit_off_reasoning({"enabled": True, "effort": "high"}) is False
    assert explicit_off_reasoning({"enabled": "true"}) is False
    assert explicit_off_reasoning({"effort": "low"}) is False
    assert explicit_off_reasoning(None) is False
    assert explicit_off_reasoning({}) is False
