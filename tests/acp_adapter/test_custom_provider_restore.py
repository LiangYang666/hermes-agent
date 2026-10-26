"""#63 — restoring a named custom provider must not land the session on the config default.

The shape of the bug: ``_persist`` stored the agent's *billing class* (``custom``) instead of the
routable ``custom:<name>`` menu key, and the ACP recreate paths fed that class straight back into
``resolve_runtime_provider``. Two different failures follow, and both were silent:

* bare ``custom`` still resolves — to whichever compatible endpoint is configured (OpenRouter on a
  box that has an OpenRouter key) — so a session created for a local lab endpoint spent every later
  turn asking the cloud for a model it has never served, and died as that vendor's
  ``model_not_found`` while the cockpit still showed the custom model selected;
* a row whose entry is gone (``custom:gone``) raises instead, the exception was swallowed at
  ``logger.debug``, and the agent was built with no provider at all — again the config default.

These tests pin the four things that have to hold together, because fixing one leaves the bug alive:
  * reading: a persisted bare class is healed from the endpoint/model before resolution,
  * writing: rows store the *intent*, so a fallback is never written back as the session's provider,
  * routing: a recovered entry owns its endpoint, and a fork inherits the parent's route,
  * honesty: what cannot be recovered is logged, not silent.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acp_adapter.session import SessionManager, _heal_custom_identity

LAB_URL = "http://127.0.0.1:8000/v1"


def _config() -> dict:
    """Global default is a cloud vendor; the named entry points somewhere else entirely."""
    return {
        "model": {"default": "deepseek-v4-flash", "provider": "deepseek"},
        "providers": {"lab": {"base_url": LAB_URL, "model": "qwen38-27b", "api_key": "lab-key"}},
    }


class _FakeDB:
    """Just enough SessionDB for ``_persist``: one create branch and one update branch."""

    def __init__(self) -> None:
        self.created: dict[str, dict] = {}
        self.updated: dict[str, str] = {}

    def get_session(self, session_id):
        return self.created.get(session_id)

    def create_session(self, *, session_id, source, model, model_config, cwd):
        self.created[session_id] = {"source": source, "model": model, "model_config": model_config}
        return session_id

    def update_session_meta(self, session_id, meta_json, model):
        self.updated[session_id] = meta_json

    def replace_messages(self, session_id, history, active_only=False):
        return None


class _RestoreDB:
    """A stored ACP row, plus no-ops for everything ``_restore`` touches on the way to the agent."""

    def __init__(self, row: dict) -> None:
        self.row = row

    def get_session(self, session_id):
        return self.row if session_id == self.row["id"] else None

    def get_compression_continuation(self, session_id):
        return None

    def get_messages_as_conversation(self, session_id, repair_alternation=True):
        return [{"role": "user", "content": "hi"}]

    def __getattr__(self, name):
        return lambda *a, **k: None


def _state(**agent_attrs) -> SimpleNamespace:
    agent = SimpleNamespace(provider="custom", base_url=LAB_URL, api_mode="chat",
                            _session_db=None, _session_db_created=False)
    for key, value in agent_attrs.items():
        setattr(agent, key, value)
    return SimpleNamespace(session_id="s1", cwd="", model="qwen38-27b",
                           history=[{"role": "user", "content": "hi"}], agent=agent,
                           reasoning_config=None, mode="", context_budget=0)


class TestHealHelper:
    def test_bare_class_is_recovered_from_the_endpoint(self):
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            assert _heal_custom_identity("custom", base_url=LAB_URL, model="qwen38-27b") == "custom:lab"

    def test_endpoint_alone_is_enough(self):
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            assert _heal_custom_identity("custom", base_url=LAB_URL, model=None) == "custom:lab"

    def test_model_alone_is_enough(self):
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            assert _heal_custom_identity("custom", base_url=None, model="qwen38-27b") == "custom:lab"

    def test_a_routable_provider_is_never_rewritten(self):
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            assert _heal_custom_identity("deepseek", base_url=LAB_URL, model="qwen38-27b") == "deepseek"
            assert _heal_custom_identity("openrouter", base_url=None, model=None) == "openrouter"
            assert _heal_custom_identity("custom:lab", base_url=None, model=None) == "custom:lab"

    def test_a_stale_entry_key_is_treated_as_unroutable(self):
        """``custom:gone`` raises inside resolution; the TUI heals it, so must the ACP path."""
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            assert _heal_custom_identity("custom:gone", base_url=LAB_URL, model=None) == "custom:lab"

    def test_nothing_persisted_stays_nothing(self):
        assert _heal_custom_identity(None, base_url=None, model=None) is None
        assert _heal_custom_identity("", base_url=None, model=None) is None

    def test_unknown_endpoint_yields_no_identity(self):
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            assert _heal_custom_identity("custom", base_url="http://elsewhere:9/v1", model="nope") is None

    def test_the_healed_key_resolves_to_the_real_endpoint(self):
        """One real resolve — this is the #63 judgement itself, not a mock's opinion."""
        from hermes_cli.runtime_provider import resolve_runtime_provider

        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            healed = _heal_custom_identity("custom", base_url=LAB_URL, model="qwen38-27b")
            runtime = resolve_runtime_provider(requested=healed, target_model="qwen38-27b")
        assert runtime["base_url"].rstrip("/") == LAB_URL
        assert runtime.get("api_key")


class TestRestoreResolvesTheSessionsOwnProvider:
    """The regression proper: what identity reaches ``resolve_runtime_provider``."""

    def _make_agent(self, **kwargs):
        from acp_adapter import session as session_mod

        seen: dict = {}

        def _fake_resolve(*, requested=None, target_model=None):
            seen["requested"] = requested
            return {"provider": "custom", "api_mode": "chat", "base_url": LAB_URL,
                    "api_key": "lab-key", "credential_pool": None, "command": None, "args": []}

        with patch("hermes_cli.config.load_config", return_value=_config()), \
             patch("hermes_cli.runtime_provider.load_config", return_value=_config()), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_fake_resolve), \
             patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_: None), \
             patch.object(session_mod, "_register_task_cwd", lambda *a, **k: None), \
             patch("run_agent.AIAgent", lambda **kw: SimpleNamespace(**kw)):
            agent = SessionManager()._make_agent(session_id="s1", cwd="/tmp", **kwargs)
        return agent, seen

    def test_persisted_billing_class_is_healed_before_resolution(self):
        _agent, seen = self._make_agent(requested_provider="custom", base_url=LAB_URL, model="qwen38-27b")
        assert seen["requested"] == "custom:lab"

    def test_the_config_default_is_not_substituted_for_the_sessions_provider(self):
        _agent, seen = self._make_agent(requested_provider="custom", base_url=LAB_URL, model="qwen38-27b")
        assert seen["requested"] != "deepseek"
        assert seen["requested"] != "custom"

    def test_a_recovered_entry_owns_its_endpoint(self):
        """A stale persisted base_url must not be paired with the recovered entry's key."""
        agent, _seen = self._make_agent(requested_provider="custom", base_url="http://127.0.0.1:8001/v1",
                                        model="qwen38-27b")
        assert agent.base_url == LAB_URL

    def test_unrecoverable_class_falls_back_loudly(self, caplog):
        with caplog.at_level("WARNING"):
            _agent, seen = self._make_agent(requested_provider="custom", base_url="http://dead:1/v1",
                                            model="ghost-model")
        assert seen["requested"] == "deepseek"           # the documented fallback, unchanged
        assert any("could not recover a provider entry" in rec.message for rec in caplog.records)

    def test_a_healthy_identity_is_passed_through(self):
        _agent, seen = self._make_agent(requested_provider="deepseek", base_url=None, model="deepseek-v4-flash")
        assert seen["requested"] == "deepseek"


class TestRestoreFromAStoredRow:
    """End-to-end through ``_restore``: the legacy-row rescue the fix exists for."""

    def _restore(self, model_config: dict, billing_provider: str, billing_base_url=None):
        from acp_adapter import session as session_mod

        seen: dict = {}

        def _fake_resolve(*, requested=None, target_model=None):
            seen["requested"] = requested
            return {"provider": "custom", "api_mode": "chat", "base_url": LAB_URL,
                    "api_key": "lab-key", "credential_pool": None, "command": None, "args": []}

        row = {"id": "s1", "source": "acp", "model": "qwen38-27b", "model_config": json.dumps(model_config),
               "billing_provider": billing_provider, "billing_base_url": billing_base_url}
        with patch("hermes_cli.config.load_config", return_value=_config()), \
             patch("hermes_cli.runtime_provider.load_config", return_value=_config()), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_fake_resolve), \
             patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_: None), \
             patch.object(session_mod, "_register_task_cwd", lambda *a, **k: None), \
             patch("run_agent.AIAgent", lambda **kw: SimpleNamespace(**kw)):
            state = SessionManager(db=_RestoreDB(row)).get_session("s1")
        return state, seen

    def test_a_legacy_row_with_the_bare_class_is_rescued(self):
        """The rows that exist in the wild: model_config.provider = 'custom', billing = 'custom'."""
        state, seen = self._restore({"provider": "custom", "base_url": LAB_URL, "api_mode": "chat"}, "custom")
        assert state is not None
        assert seen["requested"] == "custom:lab"

    def test_the_billing_column_alone_is_enough(self):
        """Older rows may carry only the endpoint, in the billing columns."""
        state, seen = self._restore({}, "custom", LAB_URL)
        assert state is not None
        assert seen["requested"] == "custom:lab"


class TestPersistStoresARoutableKey:
    def test_create_branch_writes_the_menu_key_not_the_billing_class(self):
        db = _FakeDB()
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            SessionManager(db=db)._persist(_state())
        assert db.created["s1"]["model_config"]["provider"] == "custom:lab"

    def test_update_branch_writes_the_menu_key_too(self):
        db = _FakeDB()
        db.created["s1"] = {"stub": True}
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            SessionManager(db=db)._persist(_state())
        assert json.loads(db.updated["s1"])["provider"] == "custom:lab"

    def test_a_normal_provider_row_is_untouched(self):
        db = _FakeDB()
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            SessionManager(db=db)._persist(_state(provider="deepseek", base_url="https://api.deepseek.com/v1"))
        assert db.created["s1"]["model_config"]["provider"] == "deepseek"

    def test_a_fallback_is_never_written_back_as_the_sessions_provider(self):
        """One failed restore must not turn a salvageable ``custom`` row into the config default."""
        db = _FakeDB()
        state = _state(provider="deepseek", base_url="http://dead:1/v1", model="ghost-model")
        state.model = "ghost-model"
        state.agent._acp_requested_provider = "custom"
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            SessionManager(db=db)._persist(state)
        assert db.created["s1"]["model_config"]["provider"] == "custom"


class TestForkKeepsTheParentRoute:
    def test_a_forked_custom_session_still_targets_its_entry(self):
        from acp_adapter import session as session_mod

        seen: dict = {}

        def _fake_resolve(*, requested=None, target_model=None):
            seen["requested"] = requested
            return {"provider": "custom", "api_mode": "chat", "base_url": LAB_URL,
                    "api_key": "lab-key", "credential_pool": None, "command": None, "args": []}

        manager = SessionManager(db=_FakeDB())
        parent_agent = SimpleNamespace(provider="custom", base_url=LAB_URL, api_mode="chat",
                                       _session_db=None, _session_db_created=False)
        manager._install_state("parent", parent_agent, "/tmp", "qwen38-27b", [], mode="")

        with patch("hermes_cli.config.load_config", return_value=_config()), \
             patch("hermes_cli.runtime_provider.load_config", return_value=_config()), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_fake_resolve), \
             patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_: None), \
             patch.object(session_mod, "_register_task_cwd", lambda *a, **k: None), \
             patch("run_agent.AIAgent", lambda **kw: SimpleNamespace(**kw)):
            child = manager.fork_session("parent")

        assert child is not None
        assert seen["requested"] == "custom:lab"
