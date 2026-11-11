"""Unit tests for the Xiaomi MiMo profile's reasoning wiring (#100).

MiMo thinks by default when the request carries no knob at all, so an unwired profile made
``minimal``/``none`` read as "the setting does nothing" while every turn burned tens of thousands
of reasoning tokens. Probed live against ``api.xiaomimimo.com`` on 2026-11-11: the wire takes
``none``/``low``/``medium``/``high`` for ``reasoning_effort`` (``minimal`` is an HTTP 400
naming exactly those four), ``none`` really switches thinking off, and an ``extra_body.thinking``
toggle is accepted. These tests pin that wire shape without going live.
"""

from __future__ import annotations

import pytest

from agent.reasoning_effort import MIMO_EFFORTS, route_offered_efforts
from agent.transports.chat_completions import ChatCompletionsTransport


@pytest.fixture
def xiaomi_profile():
    """Resolve the registered Xiaomi profile (a bare ``ProviderProfile`` is the bug)."""
    import model_tools  # noqa: F401  — triggers bundled plugin discovery
    import providers

    profile = providers.get_provider_profile("xiaomi")
    assert profile is not None, "xiaomi provider profile must be registered"
    assert type(profile).__name__ == "XiaomiProfile", (
        "a bare ProviderProfile returns ({}, {}) from build_api_kwargs_extras, which is exactly "
        "how the reasoning knob went missing on this route (#100)"
    )
    return profile


class TestMimoEffortVocabulary:
    @pytest.mark.parametrize(
        ("effort", "expected"),
        [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            # Above the MiMo ladder: nearest weaker level, never a 400.
            ("xhigh", "high"),
            ("max", "high"),
            # ``minimal`` is not a MiMo level; the nearest weaker *enabled* level is ``low``
            # (``clamp_effort`` refuses to degrade an enabled ask into the off state).
            ("minimal", "low"),
        ],
    )
    def test_requested_effort_lands_inside_the_mimo_vocabulary(self, xiaomi_profile, effort, expected):
        extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="mimo-v2.6-flash"
        )
        assert extra_body == {}, "the graded path must not add a toggle the wire did not ask for"
        assert top_level == {"reasoning_effort": expected}
        assert expected in MIMO_EFFORTS

    def test_off_is_the_toggle_in_every_spelling(self, xiaomi_profile):
        for reasoning_config in (
            {"enabled": True, "effort": "none"},
            {"enabled": False},
            {"effort": "off"},
        ):
            assert xiaomi_profile.build_api_kwargs_extras(
                reasoning_config=reasoning_config, model="mimo-v2.6-flash"
            ) == ({"thinking": {"type": "disabled"}}, {}), reasoning_config

    def test_unset_config_leaves_the_server_default_alone(self, xiaomi_profile):
        assert xiaomi_profile.build_api_kwargs_extras(
            reasoning_config=None, model="mimo-v2.6-flash"
        ) == ({}, {})

    def test_picker_offers_only_levels_the_wire_accepts(self):
        for provider in ("xiaomi", "mimo", "xiaomi-mimo"):
            assert route_offered_efforts(provider, "mimo-v2.6-flash") == MIMO_EFFORTS


class TestMimoWireShape:
    """The transport must actually carry the knob — the profile's return value alone proves nothing."""

    def test_session_effort_reaches_api_kwargs(self, xiaomi_profile):
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="mimo-v2.6-flash",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            provider_profile=xiaomi_profile,
            max_tokens=None,
            max_tokens_param_fn=lambda x: {"max_tokens": x} if x else {},
            timeout=300,
            reasoning_config={"enabled": True, "effort": "low"},
            request_overrides=None,
            session_id="test",
            ollama_num_ctx=None,
        )
        assert kwargs.get("reasoning_effort") == "low"
        assert "thinking" not in (kwargs.get("extra_body") or {})

    def test_off_reaches_extra_body_thinking_disabled(self, xiaomi_profile):
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="mimo-v2.6-flash",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            provider_profile=xiaomi_profile,
            max_tokens=None,
            max_tokens_param_fn=lambda x: {"max_tokens": x} if x else {},
            timeout=300,
            reasoning_config={"enabled": False},
            request_overrides=None,
            session_id="test",
            ollama_num_ctx=None,
        )
        assert (kwargs.get("extra_body") or {}).get("thinking") == {"type": "disabled"}
        assert "reasoning_effort" not in kwargs
