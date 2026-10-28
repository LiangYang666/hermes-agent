"""#72 — a named custom endpoint must keep its identity through the persisted-default round-trip.

Three links broke, each verified here:

1. ``_build_model_state`` handed the catalog the class ``custom`` instead of the entry's name, so the
   entry's own rows were never marked ``current`` and an invented ``Custom · <model>`` row took its
   place (a row whose "provider" is a CATEGORY).
2. ``model.default`` holding an ACP wire id (``custom:<name>:<model>``) was passed on as the model
   NAME, so ``agent_init`` opened a client with ``model=<name>:<model>`` and the upstream answered
   404/model_not_found.
3. The catalog could not recover the entry from a bare ``custom`` even though it already had the
   session's ``base_url`` — the only recovery branch was ollama-specific.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import acp_adapter.model_catalog as catalog_mod
from acp_adapter.model_catalog import build_model_state
from acp_adapter.session import _persisted_model_default

LAB_URL = "http://127.0.0.1:8000/v1"


def _row(slug, models, api_url=None, user_defined=False):
    return {"slug": slug, "name": slug, "api_url": api_url, "models": models,
            "is_user_defined": user_defined}


@pytest.fixture
def inventory(monkeypatch):
    """Two named entries + the legacy class bucket, without touching the real config/inventory."""
    rows = [
        _row("custom", ["shared-model"]),
        _row("lab", [{"id": "shared-model"}, {"id": "only-lab"}], api_url=LAB_URL, user_defined=True),
        _row("relay", [{"id": "relay-model"}, {"id": "dupe-model"}],
             api_url="https://relay.example/v1", user_defined=True),
        _row("lab", [{"id": "dupe-model"}], api_url=LAB_URL, user_defined=True),
    ]
    named = [
        ("custom:lab", "lab", [("shared-model", ""), ("only-lab", ""), ("dupe-model", "")]),
        ("custom:relay", "relay", [("relay-model", ""), ("dupe-model", "")]),
    ]

    class _Ctx:
        def with_overrides(self, **_kw):
            return self

    with patch("hermes_cli.inventory.load_picker_context", return_value=_Ctx()), patch(
        "hermes_cli.inventory.build_models_payload", return_value={"providers": rows}
    ), patch.object(catalog_mod, "_named_custom_provider_catalogs", return_value=named):
        yield

    # noqa: the fixture is a context manager on purpose — tests below rely on it only for patching


def _marked(state):
    return [m.model_id for m in state.available_models if "current" in (m.description or "")]


class TestIdentityRecovery:
    def test_base_url_recovers_the_named_entry(self, inventory):
        state = build_model_state("shared-model", "custom", LAB_URL)

        assert state.current_model_id == "custom:lab:shared-model"
        assert _marked(state) == ["custom:lab:shared-model"]

    def test_recovered_entry_hides_the_class_bucket(self, inventory):
        state = build_model_state("shared-model", "custom", LAB_URL)

        ids = [m.model_id for m in state.available_models]
        assert "custom:shared-model" not in ids          # the phantom / class row
        assert not any(m.model_id == "custom:custom:shared-model" for m in state.available_models)

    def test_unique_model_alone_recovers_the_entry(self, inventory):
        # No base_url to go on, but only one entry declares this model.
        state = build_model_state("only-lab", "custom", "")

        assert state.current_model_id == "custom:lab:only-lab"

    def test_ambiguous_model_marks_nothing_and_invents_nothing(self, inventory):
        # Two named entries declare "dupe-model" and the base_url matches neither: no honest answer.
        state = build_model_state("dupe-model", "custom", "http://10.255.255.1:9/v1")

        assert state.current_model_id == ""
        assert _marked(state) == []
        assert all(" • current" not in (m.description or "") for m in state.available_models)

    def test_bare_entry_name_still_resolves(self, inventory):
        state = build_model_state("shared-model", "lab", "")

        assert state.current_model_id == "custom:lab:shared-model"

    def test_canonical_provider_is_untouched(self, inventory):
        state = build_model_state("shared-model", "openrouter", "")

        assert state.current_model_id == "openrouter:shared-model"

    def test_class_bucket_never_becomes_a_double_prefixed_identity(self, inventory):
        state = build_model_state("shared-model", "custom", LAB_URL)

        assert not any(m.model_id.startswith("custom:custom") for m in state.available_models)


class TestBuildModelStateUsesRequestedProvider:
    def test_requested_provider_beats_the_billing_class(self):
        from acp_adapter.server import HermesACPAgent

        seen = {}

        def fake_build(model, provider, base_url):
            seen["provider"] = provider
            return SimpleNamespace(available_models=[], current_model_id="")

        agent = SimpleNamespace(model="shared-model", provider="custom",
                                requested_provider="custom:lab", base_url=LAB_URL)
        state = SimpleNamespace(model=None, agent=agent)

        with patch("acp_adapter.server.build_model_state", side_effect=fake_build):
            HermesACPAgent._build_model_state(object(), state)

        assert seen["provider"] == "custom:lab"

    def test_the_session_manager_attribute_is_read_first(self):
        from acp_adapter.server import HermesACPAgent

        seen = {}

        def fake_build(model, provider, base_url):
            seen["provider"] = provider
            return SimpleNamespace(available_models=[], current_model_id="")

        agent = SimpleNamespace(model="shared-model", provider="custom", requested_provider=None,
                                _acp_requested_provider="custom:lab", base_url=LAB_URL)
        state = SimpleNamespace(model=None, agent=agent)

        with patch("acp_adapter.server.build_model_state", side_effect=fake_build):
            HermesACPAgent._build_model_state(object(), state)

        assert seen["provider"] == "custom:lab"

    def test_falls_back_to_provider_when_no_requested_identity(self):
        from acp_adapter.server import HermesACPAgent

        seen = {}

        def fake_build(model, provider, base_url):
            seen["provider"] = provider
            return SimpleNamespace(available_models=[], current_model_id="")

        agent = SimpleNamespace(model="m", provider="openrouter", requested_provider=None, base_url="")
        state = SimpleNamespace(model=None, agent=agent)

        with patch("acp_adapter.server.build_model_state", side_effect=fake_build):
            HermesACPAgent._build_model_state(object(), state)

        assert seen["provider"] == "openrouter"


class TestRequestedIdentityIsWired:
    """The picker's identity must come from something the ACP build actually sets.

    ``agent.provider`` is the billing class for a named entry, and ``agent.requested_provider`` only
    carries a rebuild's argument — a fresh session asks for nothing, so ``SessionManager._make_agent``
    records the resolved intent on ``_acp_requested_provider`` (the attribute the persist path already
    reads). Without that wiring the picker's only clue is the base_url heuristic.
    """

    def test_make_agent_records_the_config_default_identity(self, monkeypatch, tmp_path):
        from acp_adapter.session import SessionManager
        from tests.acp_adapter.test_acp_reasoning_effort_config import CapturingAgent

        home = tmp_path / "hermes-home"
        home.mkdir()
        (home / "config.yaml").write_text(
            "model:\n"
            "  default: custom:lab:Fast\n"
            "  provider: lab\n"
            "providers:\n"
            "  lab:\n"
            "    name: lab\n"
            "    base_url: http://127.0.0.1:9000/v1\n"
            "    models:\n"
            "      - Fast\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr("run_agent.AIAgent", CapturingAgent)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None, **_kw: {
                "provider": "custom", "requested_provider": requested,
                "api_mode": "chat_completions", "base_url": "http://127.0.0.1:9000/v1", "api_key": "k",
            },
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda *_a, **_k: None)
        monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
                            lambda **_kw: None)

        manager = SessionManager(db=object())
        agent = manager._make_agent(session_id="s-72", cwd=".")

        assert agent._acp_requested_provider == "custom:lab"
        assert CapturingAgent.last_kwargs["model"] == "Fast"


class TestPersistedDefaultParsing:
    """The persisted default is split only when the split cannot destroy a working route.

    These drive the REAL ``parse_model_input`` (no stubs): the whole point of the guard is how that
    parser behaves on names like ``qwen:7b``, which stubbing would hide.
    """

    LAB_CONFIG = {"providers": {"lab": {"api_url": LAB_URL, "name": "lab"}}}

    def test_three_part_custom_id_is_split(self):
        model, provider = _persisted_model_default(
            {"default": "custom:lab:DeepSeek-Flash", "provider": "lab"}, self.LAB_CONFIG
        )

        assert (model, provider) == ("DeepSeek-Flash", "custom:lab")

    def test_canonical_id_is_split_only_when_the_provider_agrees(self):
        assert _persisted_model_default(
            {"default": "deepseek:deepseek-flash", "provider": "DeepSeek"}
        ) == ("deepseek-flash", "deepseek")

    def test_a_label_shaped_provider_still_agrees_after_normalisation(self):
        assert _persisted_model_default(
            {"default": "deepseek:deepseek-flash", "provider": "DEEPSEEK"}
        ) == ("deepseek-flash", "deepseek")

    @pytest.mark.parametrize("name,provider", [
        ("qwen:7b", "ollama"),
        ("kimi:latest", "ollama"),
        ("glm:latest", "custom"),
        ("moonshot:latest", ""),
        ("qwen:7b", ""),
    ])
    def test_a_colon_model_name_is_never_truncated(self, name, provider):
        # parse_model_input calls these left segments providers (alibaba / kimi-coding / zai), which
        # used to truncate the model AND re-route the session to another vendor.
        assert _persisted_model_default({"default": name, "provider": provider}) == (name, provider)

    def test_a_slash_prefixed_colon_name_is_untouched(self):
        name = "anthropic/claude-3.5-sonnet:beta"
        assert _persisted_model_default({"default": name, "provider": ""}) == (name, "")

    def test_a_custom_id_naming_no_configured_entry_is_left_verbatim(self):
        model, provider = _persisted_model_default(
            {"default": "custom:gone:Fast", "provider": "custom"}, self.LAB_CONFIG
        )

        assert (model, provider) == ("custom:gone:Fast", "custom")

    def test_string_form_and_empty_block(self):
        assert _persisted_model_default("qwen3.8-flash") == ("qwen3.8-flash", None)
        assert _persisted_model_default(None) == ("", None)
        assert _persisted_model_default({"default": ""}) == ("", None)

    def test_parser_failure_keeps_the_raw_value(self, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("inventory exploded")

        monkeypatch.setattr("hermes_cli.models.parse_model_input", boom)
        assert _persisted_model_default({"default": "a:b", "provider": "x"}) == ("a:b", "x")


class TestLegacyCustomEndpoint:
    """A box with no named entries: ``custom`` + ``model.base_url`` is a real, routable identity."""

    @pytest.fixture
    def legacy_only(self, monkeypatch):
        rows = [_row("custom", [{"id": "m2"}, {"id": "gpt-4o"}], api_url=LAB_URL)]

        class _Ctx:
            def with_overrides(self, **_kw):
                return self

        with patch("hermes_cli.inventory.load_picker_context", return_value=_Ctx()), patch(
            "hermes_cli.inventory.build_models_payload", return_value={"providers": rows}
        ), patch.object(catalog_mod, "_named_custom_provider_catalogs", return_value=[]):
            yield

    def test_the_class_row_stays_current(self, legacy_only):
        state = build_model_state("m2", "custom", LAB_URL)

        assert state.current_model_id == "custom:m2"
        assert _marked(state) == ["custom:m2"]

    def test_a_vendor_model_name_never_becomes_a_vendor_inference(self, monkeypatch):
        # A local relay serving vendor names, with OpenRouter configured and also listing gpt-4o.
        rows = [
            _row("custom", [{"id": "gpt-4o"}], api_url=LAB_URL),
            _row("openrouter", [{"id": "gpt-4o"}], api_url="https://openrouter.ai/api/v1"),
        ]

        class _Ctx:
            def with_overrides(self, **_kw):
                return self

        with patch("hermes_cli.inventory.load_picker_context", return_value=_Ctx()), patch(
            "hermes_cli.inventory.build_models_payload", return_value={"providers": rows}
        ), patch.object(catalog_mod, "_named_custom_provider_catalogs", return_value=[]):
            state = build_model_state("gpt-4o", "custom", LAB_URL)

        assert state.current_model_id == "custom:gpt-4o"
        assert "openrouter" not in state.current_model_id
