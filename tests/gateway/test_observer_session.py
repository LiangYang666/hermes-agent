"""A bystander session: no recall, no self-review, no memory writes — for THAT session only.

A cockpit's observer (工作播报) watches another session for a few minutes and says one sentence.
Long-term recall is re-read on every beat (pure cost, and it drags the wording toward whatever was
recalled), a background self-review runs ``skill_view``/``skill_manage`` — writes — on the operator's
tab, and a memory toolset would append to the operator's memory. Under 0.22 each ACP session was its
own process, so the role could be read off the environment at construction. The agent now lives in
the shared owner, so the role has to travel as a POLICY field of the session that asked for it.
"""
from types import SimpleNamespace

import pytest

from gateway.session_policy import LocalSessionPolicy, build_policy, restore_policy


def _policy(**params):
    return build_policy({'cwd': '/tmp', 'source': 'acp', 'model': 'fixture', **params}, {})


def test_a_bystander_session_is_marked_in_its_frozen_policy():
    assert _policy(observer=True).observer is True
    assert _policy().observer is False


def test_junk_is_refused_rather_than_read_as_truthy():
    from hermes_state_runtime import RuntimeStoreError
    for bad in ('true', 1, 'yes'):
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            _policy(observer=bad)


def test_the_mark_survives_the_round_trip_to_the_receipt():
    policy = _policy(observer=True)
    restored = restore_policy({**policy.__dict__})
    assert restored.observer is True


def test_a_receipt_from_a_newer_owner_does_not_brick_an_older_one():
    """A field a newer build writes must not make the whole session unreadable here: unknown keys
    are dropped, everything the older owner does know is still validated."""
    policy = _policy(observer=True)
    restored = restore_policy({**policy.__dict__, 'from_the_future': {'x': 1}})
    assert restored.observer is True and restored.cwd == policy.cwd


def _fresh_agent_kwargs(monkeypatch, observer):
    """The kwargs the turn runner hands AIAgent for a session carrying *observer*."""
    from gateway import session_execution
    from gateway.session_policy import LocalSessionPolicy

    policy = LocalSessionPolicy(**{**_policy().__dict__, 'observer': observer})
    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda runner, src: policy)
    captured = {}

    class AIAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    ctx = SimpleNamespace(
        AIAgent=AIAgent, user_config={}, session_id='s', session_key='k',
        source=SimpleNamespace(chat_id='c', user_id=None, user_id_alt=None, user_name=None,
                               chat_name=None, chat_type=None, thread_id=None),
        enabled_toolsets=['core'], disabled_toolsets=['terminal'], verbose=False,
        ephemeral_system_prompt='')
    runner = SimpleNamespace(_session_db=None, _prefill_messages=None, _service_tier=None,
                             _refresh_fallback_model=lambda: None, _checkpoint_agent_kwargs=None)
    turn_route = {'model': 'fixture', 'runtime': {}, 'request_overrides': None}
    monkeypatch.setattr('gateway.run._checkpoint_agent_kwargs', lambda user_config: {})
    monkeypatch.setattr('gateway.session_worker_construct.session_prefill', lambda *a, **k: None)
    monkeypatch.setattr(session_execution, 'api_runtime_locked', lambda: None)
    session_execution.GatewaySessionAgentMixin._build_fresh_agent(
        SimpleNamespace(_ctx=ctx, _runner=runner), turn_route, 'acp', None, 10, None,
        SimpleNamespace(get=lambda name, default=None: default), False)
    return captured


def test_a_bystander_gets_no_recall_no_self_review_and_no_memory_writes(monkeypatch):
    kwargs = _fresh_agent_kwargs(monkeypatch, observer=True)
    assert kwargs['skip_memory'] is True
    assert kwargs['skip_background_review'] is True
    assert 'memory' in kwargs['disabled_toolsets']


def test_an_ordinary_session_keeps_all_three(monkeypatch):
    kwargs = _fresh_agent_kwargs(monkeypatch, observer=False)
    assert kwargs['skip_memory'] is False
    assert kwargs['skip_background_review'] is False
    assert kwargs['disabled_toolsets'] == ['terminal']


def test_a_bystander_is_only_a_per_session_mark_not_a_profile_setting():
    """The owner serves every other session from the same config: the mark must never be read from
    the profile, and an unmarked session created next to it must be untouched."""
    marked = _policy(observer=True)
    plain = _policy()
    assert marked.observer is True and plain.observer is False
    assert 'observer' not in marked.config()
