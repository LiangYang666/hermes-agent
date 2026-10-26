"""#63 — restoring a named custom provider must not land the session on the config default.

The shape of the bug: ``_persist`` stored the agent's *billing class* (``custom``) instead of the
routable ``custom:<name>`` menu key, and ``_make_agent`` fed that class straight back into
``resolve_runtime_provider``. Resolution raises for a bare class, the exception was swallowed at
``logger.debug``, and the agent was then built with **no** provider/base_url at all — so it used
whatever ``model.provider`` said. A session created for a local lab endpoint therefore spent every
later turn asking the global vendor's cloud for a model it has never heard of, and the only
symptom was that vendor's ``model_not_found``.

These tests pin three things, because fixing only one of them leaves the bug alive:
  * reading: a persisted bare class is healed from the endpoint/model before resolution,
  * writing: new rows store the routable key, so the heal is not needed for fresh sessions,
  * honesty: when nothing can be recovered the fallback is logged, not silent.
"""

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


def _state(**agent_attrs) -> SimpleNamespace:
    agent = SimpleNamespace(**{"provider": "custom", "base_url": LAB_URL, "api_mode": "chat",
                               "_session_db": None, "_session_db_created": False})
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

    def test_a_real_provider_is_never_rewritten(self):
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            assert _heal_custom_identity("deepseek", base_url=LAB_URL, model="qwen38-27b") == "deepseek"
            assert _heal_custom_identity("openrouter", base_url=None, model=None) == "openrouter"

    def test_nothing_persisted_stays_nothing(self):
        assert _heal_custom_identity(None, base_url=None, model=None) is None
        assert _heal_custom_identity("", base_url=None, model=None) is None

    def test_unknown_endpoint_yields_no_identity(self):
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            assert _heal_custom_identity("custom", base_url="http://elsewhere:9/v1", model="nope") is None


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
        agent, seen = self._make_agent(requested_provider="custom", base_url=LAB_URL, model="qwen38-27b")
        assert seen["requested"] == "custom:lab"
        assert agent.base_url == LAB_URL

    def test_the_config_default_is_not_substituted_for_the_sessions_provider(self):
        _agent, seen = self._make_agent(requested_provider="custom", base_url=LAB_URL, model="qwen38-27b")
        assert seen["requested"] != "deepseek"
        assert seen["requested"] != "custom"

    def test_unrecoverable_class_falls_back_loudly(self, caplog):
        with caplog.at_level("WARNING"):
            _agent, seen = self._make_agent(requested_provider="custom", base_url="http://dead:1/v1",
                                            model="ghost-model")
        assert seen["requested"] == "deepseek"           # the documented fallback, unchanged
        assert any("bare billing class" in rec.message for rec in caplog.records)

    def test_a_healthy_identity_is_passed_through(self):
        _agent, seen = self._make_agent(requested_provider="deepseek", base_url=None, model="deepseek-v4-flash")
        assert seen["requested"] == "deepseek"


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
        assert '"provider": "custom:lab"' in db.updated["s1"]

    def test_a_normal_provider_row_is_untouched(self):
        db = _FakeDB()
        with patch("hermes_cli.runtime_provider.load_config", return_value=_config()):
            SessionManager(db=db)._persist(_state(provider="deepseek", base_url="https://api.deepseek.com/v1"))
        assert db.created["s1"]["model_config"]["provider"] == "deepseek"
