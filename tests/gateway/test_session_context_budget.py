"""ACP-7: a session's context budget (the context-window picker) is owner data.

Main's in-process ACP kept ``_context_window`` on its own agent; on the shared owner it is a
``model.context_length`` pin inside the session's frozen LocalSessionPolicy, written by a
revision-fenced ``session.mutate context_budget``, read back by every viewer's ``session.info``, and
applied by the next turn's agent — which must read the SESSION's pin, not the profile's window.

The pin lives in the frozen ``model`` block, and ``model`` is a FOLLOWED section: a straight follow
would replace the whole block with the profile's and silently hand the slot its full window back.
"""
import asyncio
from types import SimpleNamespace

import pytest


def _runner(store):
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _set_session_reasoning_override=lambda route, value: None,
                             _handle_message=None, _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return runner


def _mutation(db, sid, request_id, payload, operation='context_budget'):
    snap = db.get_session(sid)
    return {'id': request_id, 'method': 'session.mutate', 'params': {
        'session_id': sid, 'request_id': request_id, 'operation': operation, 'payload': payload,
        'expected_revision': snap['runtime_revision'], 'expected_generation': snap['runtime_generation']}}


async def _info(connection, sid):
    reply = await connection.dispatch({'id': 'i', 'method': 'session.info', 'params': {'session_id': sid}})
    assert 'result' in reply, reply
    return reply['result']


@pytest.mark.asyncio
async def test_a_context_budget_is_a_durable_session_setting_every_viewer_reads(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_contract import SessionRef
    from gateway.session_policy import policy_for_source, session_context_pin
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    owner = await initialize_session_authority(_runner(store), profile_id='owned', instance_id='first')
    editor = AuthorityConnection(owner, object(), {'user_id': 'human'})
    other = AuthorityConnection(owner, object(), {'user_id': 'human'})
    try:
        created = await editor.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': 'acp', 'source': 'acp', 'cwd': str(tmp_path), 'model': 'frozen',
            'editor': {'mcp_servers': []}}})
        sid = created['result']['session_id']
        # A session that never picks one runs at the model's own window.
        assert (await _info(editor, sid)).get('context_length') is None

        applied = await editor.dispatch(_mutation(db, sid, 'm1', {'tokens': 262_144}))
        assert applied['result']['context_length'] == 262_144, applied
        # What a second attached surface reads, and what the next turn's agent constructs with.
        assert (await _info(other, sid))['context_length'] == 262_144
        live = owner.sessions[sid]
        policy = policy_for_source(owner.runner, live.source)
        assert policy.config()['model']['context_length'] == 262_144
        assert policy.effective(owner)[0]['model']['context_length'] == 262_144
        # The profile's own config is never rewritten by a slot's budget.
        assert 'context_length' not in (policy.read_profile(owner) or {}).get('model', {})

        # Refusals: a bad number never becomes a pin the next turn would choke on.
        for value in [0, -1024, '256K', True, 12.5, None if False else []]:
            bad = await editor.dispatch(_mutation(db, sid, 'm2', {'tokens': value}))
            assert bad['error']['data']['reason'] == 'invalid_params', (value, bad)
        bad_key = await editor.dispatch(_mutation(db, sid, 'm3', {'tokens': 1024, 'extra': 1}))
        assert bad_key['error']['data']['reason'] == 'invalid_params', bad_key

        # ``auto`` hands the window back to the model: the pin is gone, not zero.
        cleared = await editor.dispatch(_mutation(db, sid, 'm4', {'tokens': None}))
        assert cleared['result'].get('context_length') is None, cleared
        assert 'context_length' not in (await _info(editor, sid))
    finally:
        await editor.close()
        await other.close()

    # An owner restart restores the budget from the stored receipt, and a turn under it reads the pin.
    cold = await initialize_session_authority(_runner(store), profile_id='owned', instance_id='restarted')
    viewer = AuthorityConnection(cold, object(), {'user_id': 'human'})
    try:
        await viewer.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': sid}})
        again = await viewer.dispatch(_mutation(db, sid, 'm5', {'tokens': 131_072}))
        assert again['result']['context_length'] == 131_072, again
        policy = policy_for_source(cold.runner, cold.sessions[sid].source)
        assert policy.config()['model']['context_length'] == 131_072
        assert policy.effective(cold)[0]['model']['context_length'] == 131_072
        assert session_context_pin() is None  # outside a turn there is no active policy
        from gateway.session_policy import policy_scope
        with policy_scope(policy, authority=cold):
            assert session_context_pin() == 131_072
        assert SessionRef(cold.profile_id, sid).session_id == sid
    finally:
        await viewer.close()
        store.close_all_db_handles()


def test_a_following_session_keeps_its_own_budget_over_the_profiles_window(tmp_path, monkeypatch):
    """``model`` is a followed section: a slot's pin must survive the follow, and the profile's pin
    must not leak into a session that never set one."""
    from dataclasses import replace
    from gateway import run
    from gateway.session_policy import build_policy, session_context_pin
    profile = {'model': {'default': 'frozen', 'provider': 'p', 'context_length': 999_000}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: profile)
    from tests.gateway.test_session_edit_policy import _runner  # noqa: F401  (shared fixture shape)
    authority = SimpleNamespace(profile_id=str(tmp_path))
    monkeypatch.setattr('gateway.session_policy.profile_config', lambda a: profile)
    follow = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'frozen',
                           'follow_profile_config': True}, {'model': {'default': 'frozen', 'provider': 'p'}})
    follow = replace(follow, config_json=follow.config_json.replace(
        '"model": {', '"model": {"context_length": 131072, ', 1))
    assert follow.follows_profile
    pinned, _model = follow.effective(authority)
    assert pinned['model']['context_length'] == 131_072
    assert pinned['model']['default'] == 'frozen'

    # A session with no pin of its own follows the profile's window untouched.
    plain = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'frozen',
                          'follow_profile_config': True}, {'model': {'default': 'frozen', 'provider': 'p'}})
    assert plain.effective(authority)[0]['model']['context_length'] == 999_000
    assert session_context_pin() is None
