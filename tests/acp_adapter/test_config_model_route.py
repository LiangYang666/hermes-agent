"""ACP session building decodes a qualified ``model.default`` (#102).

``model.default: xiaomi:mimo-v2.6-flash`` beside ``model.provider: deepseek`` used to be handed to
DeepSeek as the literal model id ``xiaomi:mimo-v2.6-flash`` — the session was advertised as
``deepseek:xiaomi:mimo-v2.6-flash`` and its first turn died on the upstream model whitelist
("supported API model names are …, but you passed xiaomi:mimo-v2.6-flash"). The qualified form is
legal (``-m``/``/model`` accept it), so the read side has to decode it like every other startup
path does.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from acp_adapter.session import SessionManager, _decode_config_model


class NoopDb:
    def get_session(self, *_a, **_k):
        return None

    def save_session(self, *_a, **_k):
        return None

    def __getattr__(self, name):  # every other SessionDB call is a no-op
        def _noop(*_a, **_k):
            return None

        return _noop


class TestDecodeConfigModel:
    def test_qualified_default_wins_over_a_stale_config_provider(self):
        """The reported case: mimochi id with ``provider: deepseek`` beside it."""
        assert _decode_config_model(
            "xiaomi:mimo-v2.6-flash", config_provider="deepseek"
        ) == ("mimo-v2.6-flash", "xiaomi")

    @pytest.mark.parametrize(
        ("raw", "provider"),
        [
            ("mimo-v2.6-flash", "xiaomi"),
            ("deepseek-v4-pro", "deepseek"),
            ("", None),
            (None, None),
        ],
    )
    def test_plain_ids_pass_through_untouched(self, raw, provider):
        assert _decode_config_model(raw, config_provider=provider) == (raw or "", provider)

    def test_aggregator_native_slug_is_not_split(self):
        """``anthropic/claude-*`` on OpenRouter is the aggregator's own id, not provider/model."""
        raw = "anthropic/claude-opus-4.6"
        assert _decode_config_model(raw, config_provider="openrouter") == (raw, "openrouter")

    def test_configured_custom_provider_prefix_is_decoded(self):
        model, provider = _decode_config_model(
            "custom:lab:qwen3.8-27b",
            config_provider="deepseek",
            user_providers={"lab": {"base_url": "https://lab.invalid/v1"}},
        )
        assert model == "qwen3.8-27b"
        assert provider in {"custom:lab", "lab"}, provider


def test_make_agent_routes_a_qualified_default_to_its_own_provider(monkeypatch):
    """End of the chain: the agent is built with the decoded pair, not the raw string."""
    captured: dict = {}
    runtime_calls: list[dict] = []

    class CapturingAgent:
        model = "mimo-v2.6-flash"

        def __init__(self, **kwargs):
            captured.update(kwargs)

    def mod(name, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    def fake_resolve_runtime_provider(**kwargs):
        runtime_calls.append(kwargs)
        return {
            "provider": kwargs.get("requested") or "unknown",
            "api_mode": "chat_completions",
            "base_url": "u",
            "api_key": "k",
            "command": None,
            "args": [],
        }

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=CapturingAgent))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "xiaomi:mimo-v2.6-flash", "provider": "deepseek"}},
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod("hermes_cli.runtime_provider", resolve_runtime_provider=fake_resolve_runtime_provider),
    )

    manager = SessionManager(db=NoopDb())
    manager._make_agent(session_id="acp-session", cwd=".")

    assert captured["model"] == "mimo-v2.6-flash"
    assert captured["provider"] == "xiaomi"
    assert runtime_calls and runtime_calls[0]["requested"] == "xiaomi", runtime_calls
    assert runtime_calls[0]["target_model"] == "mimo-v2.6-flash"
