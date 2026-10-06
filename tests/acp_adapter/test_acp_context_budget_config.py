"""ACP must expose the CONTEXT BUDGET as a session config option — and it must be a real knob.

Operator report (2026-10-06): "那个上下文的设置能够实现一下吗?" — AgentSlot had a per-model context
number that only relabelled the usage meter: ACP has no "set the context window" method, so nothing
about the request changed, and the setting read as broken. It does not have to be cosmetic. Hermes
budgets its own compression against a window (``ContextCompressor.context_length``; the trigger is a
percentage of it), and the TUI/gateway already change it at runtime through
``agent_init.set_config_context_length`` + a compressor re-derivation (#116467). This option is that
same knob, over ACP. Invariants:

1. ``_build_config_options`` advertises a ``context_budget`` select ("auto" = follow the model) whose
   values are the windows we offer — the picker cannot request a window we do not implement.
2. ``set_config_option(context_budget)`` records the pick on the session, pushes it onto the LIVE
   agent's compressor (both cached copies of the pin, then a re-derived trigger), and persists it:
   without persistence a restore silently reverts to the model default while the cockpit still shows
   the pick (the same failure ``reasoning_config`` exists to prevent).
3. ``auto`` clears the override instead of pinning some invented window.
4. An unparseable value is rejected (-32602), never snapped.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import yaml

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, apply_context_budget

from tests.acp_adapter.test_acp_commands import FakeAgent
from tests.acp_adapter.test_acp_reasoning_effort_config import (
    RecordingDb,
    _live_session,
    _make_manager,
    real_make_agent,
)


def _budget_option(acp_agent, state):
    options = acp_agent._build_config_options(state)
    return next((o for o in options if o.id == "context_budget"), None)


class Compressor:
    """Stand-in for ContextCompressor: only the attributes the pin actually touches."""

    def __init__(self, window: int = 200_000):
        self._config_context_length = None
        self._resolved_context_length = window
        self._threshold_tokens = 1
        self._tail_token_budget = 2

    @property
    def context_length(self) -> int:
        return self._resolved_context_length

    @context_length.setter
    def context_length(self, value: int) -> None:
        self._resolved_context_length = value


def _agent_with_compressor(window: int = 200_000):
    agent = FakeAgent()
    agent.context_compressor = Compressor(window)
    return agent


def test_advertises_the_windows_we_implement():
    db = RecordingDb()
    manager = _make_manager(db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    opt = _budget_option(acp_agent, state)
    assert opt is not None, "context_budget config option missing from session responses"
    assert opt.type == "select"
    assert [o.value for o in opt.options] == [v for v, _ in HermesACPAgent._CONTEXT_BUDGET_CHOICES]
    assert ["auto", "65536"][0] in [o.value for o in opt.options]
    # Nothing picked anywhere → follow the model, and never invent a window.
    assert opt.current_value == "auto"


def test_set_budget_records_persists_and_rebuilds_the_option():
    db = RecordingDb()
    manager = _make_manager(db, FakeAgent())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    resp = asyncio.run(acp_agent.set_config_option(
        config_id="context_budget", session_id=state.session_id, value="65536"))

    assert state.context_budget == 65536
    assert _budget_option(acp_agent, state).current_value == "65536"
    assert any(o.id == "context_budget" for o in resp.config_options), \
        "the set response must carry the rebuilt option list"
    stored = json.loads(db.rows[state.session_id]["model_config"])
    assert stored.get("context_budget") == 65536, \
        "an unpersisted pick reverts to the model window on the next restore"


def test_auto_clears_the_pick():
    db = RecordingDb()
    manager = _make_manager(db, FakeAgent())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    asyncio.run(acp_agent.set_config_option(
        config_id="context_budget", session_id=state.session_id, value="65536"))
    asyncio.run(acp_agent.set_config_option(
        config_id="context_budget", session_id=state.session_id, value="auto"))

    assert state.context_budget is None
    assert _budget_option(acp_agent, state).current_value == "auto"


def test_bogus_budget_is_rejected_not_snapped():
    db = RecordingDb()
    manager = _make_manager(db, FakeAgent())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _live_session(manager)

    with pytest.raises(Exception):
        asyncio.run(acp_agent.set_config_option(
            config_id="context_budget", session_id=state.session_id, value="as-big-as-you-can"))
    assert state.context_budget is None, "a rejected value must not leave a half-applied pick"


@pytest.mark.parametrize("value,expected", [("64k", 64_000), ("200000", 200_000), ("1m", 1_000_000)])
def test_budget_parsing_accepts_plain_counts_and_suffixes(value, expected):
    assert HermesACPAgent._parse_context_budget(value) == expected


class TestApplyContextBudget:
    """The knob is real: it repins BOTH cached copies of the window and drops the cached trigger."""

    def test_pin_sets_the_window_everywhere(self):
        agent = _agent_with_compressor(200_000)
        assert apply_context_budget(agent, 65_536) is True
        assert agent._config_context_length == 65_536
        assert agent.context_compressor._config_context_length == 65_536
        assert agent.context_compressor.context_length == 65_536
        assert agent.context_compressor._threshold_tokens is None, \
            "a stale trigger would compress against the old window"

    def test_auto_drops_the_override_and_re_infers(self):
        agent = _agent_with_compressor(65_536)
        apply_context_budget(agent, 65_536)
        apply_context_budget(agent, None)
        assert agent._config_context_length is None
        assert agent.context_compressor._config_context_length is None
        assert agent.context_compressor._resolved_context_length is None, \
            "clearing must force re-inference from model metadata, not keep the old number"

    def test_an_agent_without_a_compressor_is_reported_not_crashed(self):
        assert apply_context_budget(FakeAgent(), 65_536) is False


def test_budget_round_trips_through_restore(real_make_agent):
    """A restart must not silently put the session back on the model's full window."""
    db = RecordingDb()
    db.rows["budget-1"] = {
        "id": "budget-1", "source": "acp", "model": "qwen3.8-flash",
        "model_config": json.dumps({"cwd": ".", "context_budget": 131072}),
    }
    manager = SessionManager(db=db)
    manager._get_db = lambda: db

    restored = manager.get_session("budget-1")
    assert restored is not None
    assert restored.context_budget == 131072, "restore dropped the picked window"
    assert _budget_option(HermesACPAgent(session_manager=manager), restored).current_value == "131072"


def test_legacy_row_restores_to_auto(real_make_agent):
    db = RecordingDb()
    db.rows["legacy-2"] = {
        "id": "legacy-2", "source": "acp", "model": "qwen3.8-flash",
        "model_config": json.dumps({"cwd": "."}),
    }
    manager = SessionManager(db=db)
    manager._get_db = lambda: db

    restored = manager.get_session("legacy-2")
    assert restored is not None
    assert restored.context_budget is None
    assert _budget_option(HermesACPAgent(session_manager=manager), restored).current_value == "auto"


class TestFreeFormBudget:
    """The presets are shortcuts, not the allowed set: any window the operator types must work.

    A closed dropdown is the reason the raw protocol has a custom-type slot at all (ACP reserves
    ``_``-prefixed names for implementation-specific option kinds). Until the adapter speaks the
    version that carries that variant, the same intent is expressed inside a v1 select: ``_meta``
    describes the free-form shape for clients that read it, and the session's own pick is echoed
    back as an option so ``currentValue`` stays inside the advertised set for clients that do not.
    """

    def test_freeform_and_presets_are_advertised(self):
        acp_agent = HermesACPAgent(session_manager=_make_manager(RecordingDb()))
        opt = _budget_option(acp_agent, _live_session(_make_manager(RecordingDb())))

        assert opt.category == "_context_window", \
            "a `_`-prefixed category is the protocol's slot for a custom option kind"
        meta = opt.field_meta
        assert meta["freeform"] is True, "without this a client has no reason to offer an input"
        assert meta["unit"] == "tokens"
        assert meta["min"] == HermesACPAgent._CONTEXT_BUDGET_MIN_TOKENS
        assert meta["presets"] == [262144, 524288, 786432, 1048576], \
            "presets must stay available as shortcuts next to the input"

    def test_a_hand_typed_window_is_accepted_and_echoed_as_an_option(self):
        db = RecordingDb()
        manager = _make_manager(db, FakeAgent())
        acp_agent = HermesACPAgent(session_manager=manager)
        state = _live_session(manager)

        asyncio.run(acp_agent.set_config_option(
            config_id="context_budget", session_id=state.session_id, value="300000"))

        opt = _budget_option(acp_agent, state)
        assert state.context_budget == 300_000
        assert opt.current_value == "300000"
        values = [o.value for o in opt.options]
        assert "300000" in values, \
            "currentValue must be one of the advertised values or the list is lying"
        presets = [v for v, _ in HermesACPAgent._CONTEXT_BUDGET_CHOICES]
        assert all(p in values for p in presets), \
            "echoing the custom pick must not push the presets out"
        assert next(o.name for o in opt.options if o.value == "300000").endswith("(custom)")
        assert json.loads(db.rows[state.session_id]["model_config"])["context_budget"] == 300_000

    def test_preset_picks_do_not_grow_the_option_list(self):
        manager = _make_manager(RecordingDb(), FakeAgent())
        acp_agent = HermesACPAgent(session_manager=manager)
        state = _live_session(manager)

        asyncio.run(acp_agent.set_config_option(
            config_id="context_budget", session_id=state.session_id, value="262144"))

        opt = _budget_option(acp_agent, state)
        assert [o.value for o in opt.options] == [v for v, _ in HermesACPAgent._CONTEXT_BUDGET_CHOICES]

    def test_below_the_floor_is_refused(self):
        manager = _make_manager(RecordingDb(), FakeAgent())
        acp_agent = HermesACPAgent(session_manager=manager)
        state = _live_session(manager)

        with pytest.raises(Exception):
            asyncio.run(acp_agent.set_config_option(
                config_id="context_budget", session_id=state.session_id, value="4096"))
        assert state.context_budget is None, "a window the fixed prompt overflows must not be pinned"

    def test_hand_typed_window_round_trips_through_restore(self, real_make_agent):
        db = RecordingDb()
        db.rows["budget-9"] = {
            "id": "budget-9", "source": "acp", "model": "qwen3.8-flash",
            "model_config": json.dumps({"cwd": ".", "context_budget": 300_000}),
        }
        manager = SessionManager(db=db)
        manager._get_db = lambda: db

        restored = manager.get_session("budget-9")
        assert restored is not None and restored.context_budget == 300_000
        opt = _budget_option(HermesACPAgent(session_manager=manager), restored)
        assert opt.current_value == "300000" and "300000" in [o.value for o in opt.options]


class TestTheWindowBelongsToTheModel:
    """Operator: "改的话就改全局的…这个模型的全都全都改掉" — Studio keeps a context length per
    provider+model and so does Hermes: the pin is ``model_overrides.<provider>.<model>.context_window``
    in config.yaml, which every runtime reads at model_metadata resolution 0b (TUI, gateway, CLI,
    another slot). A pick therefore outlives the session that made it, and the sessions already
    running that model follow it immediately instead of at their next restart."""

    @pytest.fixture(autouse=True)
    def home(self, tmp_path, monkeypatch):
        """One HERMES_HOME per test: the pick now lands in config.yaml, so a shared home would hand
        the next test an already-pinned model."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        return tmp_path

    def _set(self, acp_agent, state, value):
        return asyncio.run(acp_agent.set_config_option(
            config_id="context_budget", session_id=state.session_id, value=value))

    def _manager_with_compressors(self, db):
        return SessionManager(agent_factory=lambda **_kw: _agent_with_compressor(), db=db)

    def test_the_pick_is_written_for_the_provider_and_model(self, home):
        manager = _make_manager(RecordingDb(), FakeAgent())
        acp_agent = HermesACPAgent(session_manager=manager)
        state = _live_session(manager)

        self._set(acp_agent, state, "300000")

        written = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert written["model_overrides"]["fake-provider"]["fake-model"]["context_window"] == 300_000, \
            "the window must be a property of the model, not of the session that set it"

    def test_the_runtime_resolves_the_pinned_window(self, home):
        """The proof that this is the real knob: the call a fresh session makes returns the pick."""
        manager = _make_manager(RecordingDb(), FakeAgent())
        acp_agent = HermesACPAgent(session_manager=manager)
        state = _live_session(manager)
        self._set(acp_agent, state, "300000")

        from agent.model_metadata import get_model_context_length

        assert get_model_context_length("fake-model", provider="fake-provider") == 300_000

    def test_a_session_that_never_picked_inherits_the_model_pin(self, home):
        db = RecordingDb()
        manager = _make_manager(db, FakeAgent())
        acp_agent = HermesACPAgent(session_manager=manager)
        first = _live_session(manager)
        self._set(acp_agent, first, "300000")

        second = _live_session(manager)  # nothing was ever set on this one

        assert second.context_budget is None, "the pick is not copied onto the session"
        assert _budget_option(acp_agent, second).current_value == "300000", \
            "a session on a pinned model must show that window, not 'auto'"

    def test_a_running_sibling_is_repinned_immediately(self, home):
        db = RecordingDb()
        manager = self._manager_with_compressors(db)
        acp_agent = HermesACPAgent(session_manager=manager)
        first = _live_session(manager)
        sibling = _live_session(manager)
        assert sibling.agent.context_compressor.context_length == 200_000

        self._set(acp_agent, first, "300000")

        assert sibling.context_budget == 300_000
        assert sibling.agent.context_compressor.context_length == 300_000, \
            "a sibling session would otherwise keep compressing against the old window"

    def test_auto_removes_the_pin_and_its_scaffolding(self, home):
        db = RecordingDb()
        manager = _make_manager(db, FakeAgent())
        acp_agent = HermesACPAgent(session_manager=manager)
        state = _live_session(manager)
        self._set(acp_agent, state, "300000")

        self._set(acp_agent, state, "auto")

        written = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
        assert "model_overrides" not in written, \
            "`auto` must read back as 'never set', not as empty provider scaffolding"
        assert _budget_option(acp_agent, state).current_value == "auto"

    def test_other_models_of_the_same_provider_are_untouched(self, home):
        db = RecordingDb()
        manager = _make_manager(db, FakeAgent())
        acp_agent = HermesACPAgent(session_manager=manager)
        state = _live_session(manager)

        self._set(acp_agent, state, "300000")

        written = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert list(written["model_overrides"]["fake-provider"]) == ["fake-model"], \
            "pinning one model must not write a section the runtime would read for its neighbours"
