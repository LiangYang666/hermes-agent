"""ACP must expose thinking depth (reasoning effort) as a session config option.

Hermes' ACP adapter accepted ``session/set_config_option`` but broadcast ``config_options=[]``
in every session response, so a cockpit (e.g. AgentSlot) had nothing to render and no way to
switch ``reasoning_effort`` per session — the only knobs were the config file or the CLI
``/reasoning`` command, invisible to ACP clients. Invariants:

1. ``_build_config_options`` advertises a ``reasoning_effort`` select whose options are exactly
   the levels the (provider, model) route accepts — never a wider vocabulary the wire would 400.
2. ``set_config_option(reasoning_effort)`` parses the level, pushes it onto the LIVE agent
   (same direct-assignment path as the TUI's methods_config_set), persists it, and returns the
   rebuilt option list with the new currentValue; off-route values are rejected, not snapped.
3. The pick round-trips through save/restore: the rebuilt agent is built with the session's
   reasoning_config instead of re-resolving the config default, and legacy rows (no key)
   restore cleanly to "follow the config default".
4. Session responses (new/load/resume) carry ``config_options``.
"""

import asyncio
import json

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from agent.reasoning_effort import OPENAI_COMPAT_WIRE_EFFORTS

from tests.acp_adapter.test_acp_commands import FakeAgent


class RecordingDb:
    """Minimal SessionDB stand-in that records what _persist writes and can
    hand rows back to _restore."""

    def __init__(self):
        self.rows = {}  # session_id -> row dict

    def get_session(self, session_id, *_args, **_kwargs):
        return self.rows.get(session_id)

    def create_session(self, session_id, source, model=None, model_config=None, **_kwargs):
        meta_json = json.dumps(model_config) if model_config is not None else None
        self.rows[session_id] = {"id": session_id, "source": source, "model": model,
                                 "model_config": meta_json}
        return session_id

    def update_session_meta(self, session_id, model_config_json, model=None):
        row = self.rows.get(session_id)
        if row is not None:
            row["model_config"] = model_config_json

    def replace_messages(self, *_args, **_kwargs):
        return None

    def get_messages_as_conversation(self, *_args, **_kwargs):
        return []


class CapturingAgent(FakeAgent):
    """FakeAgent whose construction kwargs the test can inspect (restore rebuilds through
    the real ``_make_agent`` with ``run_agent.AIAgent`` patched to this)."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        CapturingAgent.last_kwargs = dict(kwargs)
        super().__init__()
        self.model = kwargs.get("model") or self.model
        self.provider = kwargs.get("provider") or self.provider


def _make_manager(db, fake=None):
    return SessionManager(agent_factory=lambda **_kwargs: (fake or FakeAgent()), db=db)


def _live_session(manager):
    """_persist keeps contentless sessions ephemeral — give the row a message so it lands."""
    state = manager.create_session(cwd=".")
    state.history.append({"role": "user", "content": "hello"})
    return state


def _effort_option(acp_agent, state):
    options = acp_agent._build_config_options(state)
    return next((o for o in options if o.id == "reasoning_effort"), None)


@pytest.fixture
def real_make_agent(monkeypatch, tmp_path):
    """Point _make_agent's heavy path at CapturingAgent with an empty config home."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("run_agent.AIAgent", CapturingAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kw: {"provider": requested or "fake-provider",
                                       "api_mode": "chat_completions",
                                       "base_url": "https://example.invalid/v1",
                                       "api_key": "test-key"},
    )
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
                        lambda **_kw: None)
    return CapturingAgent


def test_advertises_route_supported_efforts():
    """FakeAgent is provider='fake-provider' → the widest OpenAI-compatible vocabulary,
    and nothing outside it (no invented 'ultra' the wire rejects)."""
    db = RecordingDb()
    manager = _make_manager(db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    opt = _effort_option(acp_agent, state)
    assert opt is not None, "reasoning_effort config option missing from session responses"
    assert opt.type == "select"
    assert [o.value for o in opt.options] == list(OPENAI_COMPAT_WIRE_EFFORTS)
    # Nothing configured anywhere → genuinely unset; must not invent an effort.
    assert opt.current_value == ""


def test_set_config_option_applies_persists_and_rebuilds_options():
    db = RecordingDb()
    fake = FakeAgent()
    manager = _make_manager(db, fake)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    resp = asyncio.run(acp_agent.set_config_option(
        config_id="reasoning_effort", session_id=state.session_id, value="high"))

    assert state.reasoning_config == {"enabled": True, "effort": "high"}
    assert fake.reasoning_config == {"enabled": True, "effort": "high"}, \
        "live agent not updated — next turn would still run at the old depth"
    opt = _effort_option(acp_agent, state)
    assert opt.current_value == "high"
    assert len(resp.config_options) >= 1, "set response must carry the rebuilt option list, not []"
    rebuilt = next(o for o in resp.config_options if o.id == "reasoning_effort")
    assert rebuilt.current_value == "high"

    meta = json.loads(db.get_session(state.session_id)["model_config"])
    assert meta.get("reasoning_config") == {"enabled": True, "effort": "high"}


def test_off_route_value_is_rejected_not_snapped():
    """The picker only ever advertises `supported`; anything else arriving is a client bug →
    -32602, never a silent snap to some other tier."""
    db = RecordingDb()
    manager = _make_manager(db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    import acp
    with pytest.raises(acp.RequestError):
        asyncio.run(acp_agent.set_config_option(
            config_id="reasoning_effort", session_id=state.session_id, value="ultra"))
    with pytest.raises(acp.RequestError):
        asyncio.run(acp_agent.set_config_option(
            config_id="reasoning_effort", session_id=state.session_id, value="hgih"))
    assert state.reasoning_config is None


def test_none_disables_and_round_trips_through_restore(real_make_agent):
    """'none' is a first-class choice ({'enabled': False}), survives a simulated restart, and
    reaches the rebuilt agent as an explicit override of the config default."""
    db = RecordingDb()
    manager = _make_manager(db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    asyncio.run(acp_agent.set_config_option(
        config_id="reasoning_effort", session_id=state.session_id, value="none"))
    assert state.reasoning_config == {"enabled": False}
    assert _effort_option(acp_agent, state).current_value == "none"

    # Simulated restart: drop memory, restore through the REAL _make_agent (factory removed).
    with manager._lock:
        manager._sessions.clear()
    manager._agent_factory = None
    manager._get_db = lambda: db

    restored = manager.get_session(state.session_id)
    assert restored is not None
    assert restored.reasoning_config == {"enabled": False}
    assert real_make_agent.last_kwargs.get("reasoning_config") == {"enabled": False}, \
        "rebuilt agent ignored the session's persisted effort"
    assert _effort_option(acp_agent, restored).current_value == "none"


def test_legacy_row_restores_to_config_default(real_make_agent, tmp_path):
    """A row persisted before this feature (no reasoning_config key) must restore cleanly and
    keep following resolve_reasoning_config — never None-shadow the default or crash."""
    import yaml
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(
        {"model": {"default": "gpt-5.6", "provider": "openai-api"},
         "agent": {"reasoning_effort": "medium"}}), encoding="utf-8")

    db = RecordingDb()
    db.rows["legacy-1"] = {
        "id": "legacy-1", "source": "acp", "model": "gpt-5.6",
        "model_config": json.dumps({"cwd": "."}),
    }

    manager = SessionManager(db=db)
    manager._get_db = lambda: db
    restored = manager.get_session("legacy-1")
    assert restored is not None
    assert restored.reasoning_config is None
    # _make_agent still resolved the config default for the agent itself:
    assert real_make_agent.last_kwargs.get("reasoning_config") == {"enabled": True, "effort": "medium"}


def test_session_response_fields_carry_config_options():
    db = RecordingDb()
    manager = _make_manager(db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    fields = asyncio.run(acp_agent._session_response_fields(state))
    assert "config_options" in fields
    assert any(o.id == "reasoning_effort" for o in fields["config_options"])

@pytest.mark.parametrize("provider,model,expected", [
    # 千问 3.8 Flash/Max (DashScope): reasoning_effort takes exactly low/medium/xhigh, and thinking
    # is switched off with enable_thinking=false — so "Off" is a real level and "High"/"Max" are
    # aliases of xhigh, not levels of their own. Offering the seven-level vocabulary here was the
    # bug: four picks folded away on the wire and the setting read as "does nothing" (2026-10-06).
    ("alibaba", "qwen3.8-flash", ("none", "low", "medium", "xhigh")),
    ("alibaba-cn", "qwen3.8-max", ("none", "low", "medium", "xhigh")),
    ("alibaba-token-plan", "qwen/qwen3.8-flash", ("none", "low", "medium", "xhigh")),
    # DeepSeek V4 on either host: low..max plus the thinking toggle.
    ("deepseek", "deepseek-v4-pro", ("none", "low", "medium", "high", "max")),
    ("alibaba", "deepseek-v4-pro", ("none", "low", "medium", "high", "max")),
    # Coding Plan — the AgentSlot route — shipped as a bare ProviderProfile, so NO level reached the
    # wire, "Off" included. Same ladders as the alibaba profiles now that it shares the translation.
    ("alibaba-coding-plan", "deepseek-v4.1-flash", ("none", "low", "medium", "high", "max")),
    ("alibaba-coding-plan", "qwen3.8-flash", ("none", "low", "medium", "xhigh")),
    # Qwen 3.7/3.6 have hybrid thinking but no graded knob (live-verified 2026-10-07): offering
    # 3.8's tiers would be three picks that change nothing.
    ("alibaba", "qwen3.7-max", ("none", "medium")),
    ("alibaba-coding-plan", "qwen3.6-flash", ("none", "medium")),
    # 阿里云直供 GLM: 5.2 has the full ladder, 5.3 always thinks and 400s on enable_thinking=false —
    # so "Off" is not a choice there, and low/high/max is the whole vocabulary.
    ("alibaba", "glm-5.2", ("none", "minimal", "low", "medium", "high", "xhigh", "max")),
    ("alibaba", "glm-5.3", ("low", "high", "max")),
    # A route we have not verified keeps every level: taking choices away is the worse error.
    ("openrouter", "some-unknown-model", OPENAI_COMPAT_WIRE_EFFORTS),
])
def test_advertises_only_the_levels_the_route_really_takes(provider, model, expected):
    """The picker shows what the wire takes — nothing it would silently fold away."""
    db = RecordingDb()
    fake = FakeAgent()
    fake.provider, fake.model = provider, model
    manager = _make_manager(db, fake)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    opt = _effort_option(acp_agent, state)
    assert opt is not None
    assert tuple(o.value for o in opt.options) == tuple(expected), \
        f"{provider}/{model} advertises a ladder its wire does not accept"
    labels = {o.value: o.name for o in opt.options}
    if "none" in expected:
        assert labels.get("none") == "Off", "the toggle must be reachable from the picker"
    else:
        assert "Off" not in labels.values(), \
            "an unswitchable route must not offer a pick that 400s the request"
    if "high" not in expected:
        assert "High" not in labels.values(), "an alias level must not be offered as its own choice"


def test_deepseek_keeps_a_level_the_entry_clamp_still_accepts():
    """Validation stays the (wide) entry clamp while the picker narrows: the CLI/TUI have always
    sent xhigh to DeepSeek, whose profile maps it onto max — that must keep working."""
    db = RecordingDb()
    fake = FakeAgent()
    fake.provider, fake.model = "deepseek", "deepseek-v4-pro"
    manager = _make_manager(db, fake)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    assert "xhigh" not in [o.value for o in _effort_option(acp_agent, state).options]
    asyncio.run(acp_agent.set_config_option(
        config_id="reasoning_effort", session_id=state.session_id, value="xhigh"))
    assert fake.reasoning_config == {"enabled": True, "effort": "xhigh"}

