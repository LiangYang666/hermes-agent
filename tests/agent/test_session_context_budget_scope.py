"""A session's context budget must survive the "configured default" scoping check.

``model.context_length`` in config.yaml describes the PROFILE's default model, so
``agent_init._scope_context_length_to_default_runtime`` drops it when the startup runtime is a
different model/route (a ``--model`` launch must not inherit the default's window).

A cockpit slot's budget is a different thing: the picker writes a ``model.context_length`` pin into
the SESSION's frozen policy (``gateway/session_mutation_policy.py``), read back by
``gateway.session_policy.session_context_pin``. It describes the route THIS session runs, which is
routinely not the profile's default (slot picks ``xiaomi:mimo-v2.6-flash`` while the profile's
default is ``deepseek-flash``). Running the pin through the configured-default scoping check
therefore threw it away as "stale", and the turn ran at the model's whole window (#101): a 512K
budget × threshold 0.5 never fired at 417K of context because the compressor was still at 1M.
"""

from types import SimpleNamespace

from run_agent import AIAgent

PIN = 524_288
ROUTE = "https://api.xiaomimimo.com/v1"

# The owning profile's config: its default model is what ``model.context_length`` describes.
PROFILE_CFG = {"model": {"default": "deepseek-flash", "provider": "deepseek"}}


def _stub_agent(model="mimo-v2.6-flash", provider="xiaomi", base_url=ROUTE):
    agent = AIAgent.__new__(AIAgent)
    agent.model = model
    agent.provider = provider
    agent.base_url = base_url
    agent.api_key = "sk-test"
    agent.api_mode = "chat_completions"
    agent.quiet_mode = True
    agent._ensure_lmstudio_runtime_loaded = lambda *_a, **_k: None
    agent._lmstudio_load_was_unverified = lambda *_a, **_k: False
    agent._effective_lmstudio_context_length = lambda cfg, _loaded: cfg
    return agent


def _active_pin(tokens):
    """Install a frozen launch policy carrying *tokens* as its ``model.context_length`` pin."""
    from gateway import session_policy
    frozen = {"model": {"default": "mimo-v2.6-flash", "provider": "xiaomi", "base_url": ROUTE,
                        "context_length": tokens}}
    return session_policy._ACTIVE_POLICY.set(SimpleNamespace(config=lambda *_a, **_k: frozen))


def _resolve(agent, cfg):
    from agent.agent_init import _resolve_context_length
    return _resolve_context_length(agent, cfg, agent.base_url)


def test_a_session_pin_survives_a_runtime_that_is_not_the_profiles_default():
    """#101: the picker's budget is session-scoped, so the default-runtime check must not drop it."""
    from gateway import session_policy
    agent = _stub_agent()
    token = _active_pin(PIN)
    try:
        config_context_length, _custom, effective, _model_cfg = _resolve(agent, PROFILE_CFG)
    finally:
        session_policy._ACTIVE_POLICY.reset(token)

    assert config_context_length == PIN
    # What the compressor is built with (``ContextCompressor(config_context_length=...)``).
    assert effective == PIN


def test_without_a_session_pin_a_stale_profile_pin_is_still_dropped():
    """The configured-default scoping keeps its meaning for the profile's own pin."""
    from gateway import session_policy
    agent = _stub_agent()
    token = session_policy._ACTIVE_POLICY.set(None)
    try:
        config_context_length, _custom, _effective, _model_cfg = _resolve(
            agent, {"model": {"default": "deepseek-flash", "provider": "deepseek", "context_length": PIN}})
    finally:
        session_policy._ACTIVE_POLICY.reset(token)

    assert config_context_length is None


def test_a_session_pin_on_the_profiles_default_runtime_is_unchanged():
    from gateway import session_policy
    agent = _stub_agent(model="deepseek-flash", provider="deepseek", base_url="https://api.deepseek.com/v1")
    token = _active_pin(PIN)
    try:
        config_context_length, _custom, effective, _model_cfg = _resolve(agent, PROFILE_CFG)
    finally:
        session_policy._ACTIVE_POLICY.reset(token)

    assert config_context_length == PIN
    assert effective == PIN
