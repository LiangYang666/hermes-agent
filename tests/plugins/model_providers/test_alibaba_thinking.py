"""Unit tests for the Alibaba/DashScope thinking wiring (Qwen 3.8 + DeepSeek V4 on 阿里云).

Why these exist (operator report, 2026-10-06): "用 DeepSeek 设置思考强度关闭它，它还是有思考；开弱，
它也是思考很多". On DashScope the reasoning is ON by default on both families, so a request that
carries no thinking parameter ALWAYS reasons, and an effort level the family does not know is
silently ignored — a cockpit picker that appears to work and changes nothing. The wire shape
checked here (千问/百炼 docs, 2026-10-06):

* ``extra_body.enable_thinking`` — the only real off switch (``false`` is rejected by thinking-only
  models such as the glm-5.3 series, so an unrecognised family gets no parameter at all);
* ``extra_body.reasoning_effort`` — Qwen 3.8 takes exactly low/medium/xhigh (high/max alias onto
  xhigh, minimal onto low), DeepSeek V4 takes low..max;
* ``thinking_budget`` is mutually exclusive with ``reasoning_effort`` and is never emitted here.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def alibaba_profile():
    """Resolve the REGISTERED profile (a plain ProviderProfile would collapse every assertion)."""
    import model_tools  # noqa: F401  (plugin discovery registers the profiles)
    import providers

    profile = providers.get_provider_profile("alibaba")
    assert profile is not None, "alibaba provider profile must be registered"
    return profile


@pytest.fixture
def alibaba_cn_profile():
    import model_tools  # noqa: F401
    import providers

    return providers.get_provider_profile("alibaba-cn")


class TestQwenThinkingWireShape:

    def test_unset_pins_the_toggle_on_and_no_effort(self, alibaba_profile):
        """Qwen 3.8 defaults to thinking ON; the request still says so explicitly, and never
        guesses an effort (the server default is xhigh)."""
        extra_body, top_level = alibaba_profile.build_api_kwargs_extras(
            reasoning_config=None, model="qwen3.8-flash")
        assert extra_body == {"enable_thinking": True}
        assert top_level == {}

    def test_off_sends_enable_thinking_false(self, alibaba_profile):
        """'Off' is ``{'enabled': False}`` downstream — the ONLY way to stop the reasoning."""
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="qwen3.8-flash")
        assert extra_body == {"enable_thinking": False}
        assert "reasoning_effort" not in extra_body

    @pytest.mark.parametrize("effort,expected", [
        ("low", "low"),
        ("medium", "medium"),
        ("xhigh", "xhigh"),
        # documented aliases: high/max are xhigh, minimal is low
        ("high", "xhigh"),
        ("max", "xhigh"),
        ("minimal", "low"),
    ])
    def test_effort_maps_onto_the_documented_levels(self, alibaba_profile, effort, expected):
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="qwen3.8-flash")
        assert extra_body == {"enable_thinking": True, "reasoning_effort": expected}

    def test_none_level_is_the_toggle_not_an_effort(self, alibaba_profile):
        """An effort of ``none`` that arrived as a level still means "do not think", never
        ``reasoning_effort: none`` (which the API would reject)."""
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}, model="qwen3.8-flash")
        assert extra_body == {"enable_thinking": True}

    def test_never_sends_thinking_budget(self, alibaba_profile):
        """``thinking_budget`` + ``reasoning_effort`` in one request is an API error."""
        for cfg in (None, {"enabled": False}, {"enabled": True, "effort": "high"}):
            extra_body, _ = alibaba_profile.build_api_kwargs_extras(
                reasoning_config=cfg, model="qwen3.8-flash")
            assert "thinking_budget" not in extra_body

    def test_vendor_prefixed_id_is_recognised(self, alibaba_profile):
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "low"}, model="qwen/qwen3.8-flash")
        assert extra_body == {"enable_thinking": True, "reasoning_effort": "low"}

    def test_cn_endpoint_behaves_identically(self, alibaba_cn_profile):
        extra_body, _ = alibaba_cn_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="qwen3.8-max")
        assert extra_body == {"enable_thinking": False}


class TestDeepSeekOnDashScope:

    def test_efforts_pass_through_and_xhigh_is_max(self, alibaba_profile):
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model="deepseek-v4-pro")
        assert extra_body == {"enable_thinking": True, "reasoning_effort": "high"}
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"}, model="deepseek-v4-flash")
        assert extra_body["reasoning_effort"] == "max"

    def test_off_uses_the_dashscope_toggle_not_deepseek_thinking_block(self, alibaba_profile):
        """DashScope's compatible mode takes ``enable_thinking``; the native-API
        ``thinking: {type: disabled}`` shape belongs to the deepseek profile only."""
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="deepseek-v4-pro")
        assert extra_body == {"enable_thinking": False}


class TestUnknownFamiliesAreLeftAlone:

    @pytest.mark.parametrize("model", [
        "glm-5.3",          # thinking-only on DashScope: enable_thinking=false is an API error
        "kimi-k3",
        "qwen-turbo",       # pre-3.x id whose thinking contract we have not verified
        "",
    ])
    def test_no_thinking_parameter_is_invented(self, alibaba_profile, model):
        assert alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model=model) == ({}, {})
