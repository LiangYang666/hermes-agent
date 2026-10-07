"""Unit tests for the Alibaba/DashScope thinking wiring (Qwen, DeepSeek and GLM on 阿里云).

Why these exist (operator reports, 2026-10-06 and 2026-10-07): "用 DeepSeek 设置思考强度关闭它，它还
是有思考；开弱，它也是思考很多", then "设置那个级别好像没什么区别" from an AgentSlot slot. On DashScope
the reasoning is ON by default on most families, so a request carrying no thinking parameter ALWAYS
reasons, and an effort level the family does not know is silently ignored — a cockpit picker that
appears to work and changes nothing. The wire shape checked here (千问/百炼 docs + live probes on the
token-plan endpoint, 2026-10-06/07):

* ``extra_body.enable_thinking`` — the only real off switch (``false`` is rejected by thinking-only
  models such as the glm-5.3 series, so an unrecognised family gets no parameter at all);
* ``extra_body.reasoning_effort`` — per FAMILY: Qwen 3.8 takes exactly low/medium/xhigh (high/max
  alias onto xhigh, minimal onto low), DeepSeek V4 low..max, 阿里云直供 glm-5.2 the full ladder and
  glm-5.3 exactly low/high/max, while Qwen 3.7/3.6/3.5 read no effort at all (only the toggle);
* ``thinking_budget`` is mutually exclusive with ``reasoning_effort`` and is never emitted here;
* every tier of the endpoint (alibaba, token-plan, coding-plan) answers to the same two parameters,
  which is why the translation is shared from ``agent.reasoning_effort`` rather than copied per
  profile — a profile that shipped without it dropped every pick, "Off" included.
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


class TestGlmOnDashScope:
    """阿里云直供 GLM. The same ids take different ladders from the vendor's own API, so the HOST
    decides here — and glm-5.3 cannot be switched off at all (``enable_thinking: false`` = 400)."""

    def test_glm_52_takes_the_full_ladder(self, alibaba_profile):
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "low"}, model="glm-5.2")
        assert extra_body == {"enable_thinking": True, "reasoning_effort": "low"}
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"}, model="glm-5.2")
        assert extra_body["reasoning_effort"] == "max"

    def test_glm_52_has_a_real_off_switch(self, alibaba_profile):
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2")
        assert extra_body == {"enable_thinking": False}

    def test_glm_53_never_sends_false(self, alibaba_profile):
        """An "off" ask lands on its lightest level instead — the closest honest match, and live
        at that level the model returns zero reasoning tokens."""
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.3")
        assert extra_body == {"enable_thinking": True, "reasoning_effort": "low"}

    @pytest.mark.parametrize("effort,expected", [
        ("minimal", "low"), ("low", "low"), ("medium", "low"),
        ("high", "high"), ("xhigh", "high"), ("max", "max"),
    ])
    def test_glm_53_clamps_onto_low_high_max(self, alibaba_profile, effort, expected):
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="glm-5.3")
        assert extra_body == {"enable_thinking": True, "reasoning_effort": expected}


class TestQwenToggleOnlyGenerations:
    """Qwen 3.7 / 3.6 / 3.5 have hybrid thinking but NO graded knob (live-verified 2026-10-07:
    ``low`` reasons like no knob, ``max`` is a 400). An effort on the wire here would be the
    "setting does nothing" bug all over again, so only the toggle travels."""

    @pytest.mark.parametrize("model", ["qwen3.7-max", "qwen3.6-flash", "qwen3.5-plus", "qwen3-max"])
    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
    def test_an_effort_never_rides_the_wire(self, alibaba_profile, model, effort):
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model=model)
        assert extra_body == {"enable_thinking": True}

    def test_the_off_switch_still_works(self, alibaba_profile):
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="qwen3.7-max")
        assert extra_body == {"enable_thinking": False}

    def test_38_still_gets_the_graded_knob(self, alibaba_profile):
        """The 3.8 prefix must beat the 3.x catch-all — same host, different contract."""
        extra_body, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model="qwen3.8-flash")
        assert extra_body == {"enable_thinking": True, "reasoning_effort": "xhigh"}


class TestCodingPlanTierSharesTheThinkingWire:
    """The coding-plan tier shipped as a bare ``ProviderProfile``, so it emitted NO thinking
    parameter: every thinking-depth pick — including "off" — was dropped on the floor (operator
    report 2026-10-07, on the AgentSlot slot that runs this route)."""

    @pytest.fixture
    def coding_plan_profile(self):
        import model_tools  # noqa: F401
        import providers

        profile = providers.get_provider_profile("alibaba-coding-plan")
        assert profile is not None, "alibaba-coding-plan provider profile must be registered"
        return profile

    def test_off_reaches_the_wire(self, coding_plan_profile):
        extra_body, top_level = coding_plan_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="deepseek-v4.1-flash")
        assert extra_body == {"enable_thinking": False}
        assert top_level == {}

    def test_effort_reaches_the_wire(self, coding_plan_profile):
        extra_body, _ = coding_plan_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "low"}, model="deepseek-v4.1-flash")
        assert extra_body == {"enable_thinking": True, "reasoning_effort": "low"}

    @pytest.mark.parametrize("model", [
        "qwen3.8-flash", "deepseek-v4-pro", "glm-5.2", "glm-5.3", "qwen3.6-flash", "kimi-k3",
    ])
    def test_one_contract_across_the_tiers(self, coding_plan_profile, alibaba_profile, model):
        """Token Plan and Coding Plan are the same wire under other keys — the two profiles must
        not drift apart, family by family."""
        for cfg in (None, {"enabled": False}, {"enabled": True, "effort": "high"}):
            assert coding_plan_profile.build_api_kwargs_extras(
                reasoning_config=cfg, model=model) == alibaba_profile.build_api_kwargs_extras(
                    reasoning_config=cfg, model=model), (model, cfg)


class TestUnknownFamiliesAreLeftAlone:

    @pytest.mark.parametrize("model", [
        "kimi-k3",          # low/high/max from 阿里直供 but `max` alone from 月之暗面 — supplier differs
        "minimax-m2.5",     # uses ``thinking: adaptive|disabled``, not these two parameters
        "stepfun/step-3.7-flash",  # defaults to thinking OFF; its own ladder is low/medium/high
        "qwen-turbo",       # pre-3.x id whose thinking contract we have not verified
        "",
    ])
    def test_no_thinking_parameter_is_invented(self, alibaba_profile, model):
        assert alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model=model) == ({}, {})
