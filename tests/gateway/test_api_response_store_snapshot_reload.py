"""issue #114: a queued API turn must not execute the frozen ResponseStore snapshot.

Admission snapshots the store's ``conversation_history`` into the payload; a turn that
drains later must drop that snapshot and read the session transcript instead, unless the
transcript cannot carry the history (carrier key) or the history came from a source that
never persists its prefix rows.
"""
import json

import pytest

from gateway.session_api_turn import (
    admit_api_turn,
    check_api_turn,
    prepare_api_execution,
)

SNAPSHOT = [
    {'role': 'user', 'content': 'first question'},
    {'role': 'assistant', 'content': 'first answer'},
]


def _admit(api, sid, history=SNAPSHOT, **kwargs):
    return admit_api_turn(api, session_id=sid, request_id=sid + '-req',
                          user_message='follow up', conversation_history=history, **kwargs)


def _flagged(row, **flags):
    payload = json.loads(json.dumps(row['payload']))
    payload['api_turn_v1'].update(flags)
    return payload


def _carrier(sid):
    return 'api.chain_store_carrier.' + sid


# --- execution side: snapshot vs transcript -----------------------------------------------


def test_store_snapshot_is_dropped_when_transcript_can_replace_it(api, owner):
    """One store turn with a complete transcript uses the transcript, not the stale snapshot."""
    _, ref, row = _admit(api, 'snapshot-drop')
    owner.db.append_message(ref.session_id, 'user', 'first question')
    owner.db.append_message(ref.session_id, 'assistant', 'first answer')
    assert prepare_api_execution(owner, ref, row['payload'])['history'] == SNAPSHOT  # control

    prepared = prepare_api_execution(owner, ref, _flagged(row, history_from_store=True))
    assert prepared['history'] is None  # None => TurnRunner reloads from the transcript


def test_store_snapshot_survives_when_carrier_key_is_already_set(api, owner):
    """Once the session is marked as store-carried, later store turns keep their snapshot."""
    _, ref, row = _admit(api, 'carrier-existing')
    owner.db.append_message(ref.session_id, 'assistant', 'committed')
    owner.db.set_meta(_carrier(ref.session_id), '1')

    prepared = prepare_api_execution(owner, ref, _flagged(row, history_from_store=True))
    assert prepared['history'] == SNAPSHOT


def test_store_snapshot_on_empty_transcript_sets_carrier_and_is_kept(api, owner):
    """A store chain whose prefix rows never landed must keep its snapshot and be marked."""
    _, ref, row = _admit(api, 'carrier-empty')
    assert owner.db.get_meta(_carrier(ref.session_id)) is None

    prepared = prepare_api_execution(owner, ref, _flagged(row, history_from_store=True))
    assert prepared['history'] == SNAPSHOT
    assert owner.db.get_meta(_carrier(ref.session_id)) == '1'


def test_unmarked_legacy_payload_keeps_snapshot_without_carrier(api, owner):
    """An admission persisted by an older build has no provenance: today's behaviour stays."""
    _, ref, row = _admit(api, 'legacy-unmarked')
    owner.db.append_message(ref.session_id, 'assistant', 'committed')

    prepared = prepare_api_execution(owner, ref, row['payload'])
    assert prepared['history'] == SNAPSHOT
    assert owner.db.get_meta(_carrier(ref.session_id)) is None


@pytest.mark.parametrize('client_history', [
    [{'role': 'user', 'content': 'client-owned prefix'}],
    [{'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'second'}],
], ids=['user-only', 'with-assistant'])
def test_not_transcript_provenance_sets_carrier_and_keeps_history(api, owner, client_history):
    """Client-owned / multi-item prefix rows never reach the transcript; keep and mark them."""
    _, ref, row = _admit(api, 'client-owned-' + str(len(client_history)), client_history)
    assert owner.db.get_meta(_carrier(ref.session_id)) is None

    prepared = prepare_api_execution(owner, ref, _flagged(row, history_not_transcript=True))
    assert prepared['history'] == client_history
    assert owner.db.get_meta(_carrier(ref.session_id)) == '1'


def test_not_transcript_alone_drops_none_history_without_carrier(api, owner):
    """history=None (session-chat / wake) carries no foreign prefix: never poison the session."""
    _, ref, row = _admit(api, 'session-source', None)

    prepared = prepare_api_execution(owner, ref, _flagged(row, history_not_transcript=True))
    assert prepared['history'] is None
    assert owner.db.get_meta(_carrier(ref.session_id)) is None


# --- admission side: provenance on the persisted payload -----------------------------------


def test_admit_persists_both_provenance_flags(api):
    _, _, row = _admit(api, 'admit-flags', history_from_store=True, history_not_transcript=True)
    data = row['payload']['api_turn_v1']
    assert data['history_from_store'] is True
    assert data['history_not_transcript'] is True


def test_admit_without_provenance_keeps_legacy_payload(api):
    """Older callers (session-chat, wake, runs lane) must stay unmarked."""
    _, _, row = _admit(api, 'admit-legacy')
    data = row['payload']['api_turn_v1']
    assert 'history_from_store' not in data
    assert 'history_not_transcript' not in data


def test_check_api_turn_accepts_bool_flags_and_rejects_other_types(api, owner):
    from hermes_state_runtime import RuntimeStoreError

    _, ref, row = _admit(api, 'flag-validation')
    check_api_turn(owner, ref, _flagged(row, history_from_store=True, history_not_transcript=True))
    for bad in ('yes', 1, [], None):
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            check_api_turn(owner, ref, _flagged(row, history_from_store=bad))
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            check_api_turn(owner, ref, _flagged(row, history_not_transcript=bad))


class _ResponseStore:
    def __init__(self, response=None, conversation=None):
        self.response = response
        self.conversation = conversation

    def get(self, response_id):
        return self.response

    def get_conversation(self, name):
        return self.conversation

    async def off_loop(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class _ContextAdapter:
    """Only the seams prepare_context uses; the parsing/provenance code under test is real."""

    def __init__(self, store):
        self._store = store
        self._model_name = 'test'
        self._run_agent = None
        self.gateway_runner = object()

    def _current_response_store(self):
        return self._store

    def _select_request_route(self, body, **kwargs):
        return {}, {}, None

    def _declared_conversation_session(self, key):
        return None


@pytest.mark.parametrize('store_field', ['previous_response_id', 'conversation'], ids=['id', 'name'])
@pytest.mark.asyncio
async def test_prepare_context_marks_single_store_snapshot(monkeypatch, store_field):
    from gateway.platforms import api_server_response_admissions as admissions

    monkeypatch.setattr(admissions, '_stored_owner', lambda adapter, sid: sid)

    async def _continued(adapter, sid, request_id=None):
        return sid, None
    monkeypatch.setattr('gateway.platforms.api_server_authority_runs.continued_session', _continued)

    store = _ResponseStore(response={
        'conversation_history': SNAPSHOT, 'session_id': 'stored-session', 'instructions': None})
    store.conversation = 'resp-1'
    adapter = _ContextAdapter(store)
    body = {'input': 'follow up'}
    body[store_field] = 'resp-1' if store_field == 'previous_response_id' else 'chat'
    context, error = await admissions.prepare_context(
        adapter, None, body, None, None, 'api', 'key')
    assert error is None
    run_kwargs = context['run_kwargs']
    assert run_kwargs['history_from_store'] is True
    assert 'history_not_transcript' not in run_kwargs
    assert run_kwargs['conversation_history'][:2] == SNAPSHOT


@pytest.mark.asyncio
async def test_prepare_context_marks_client_owned_history():
    from gateway.platforms import api_server_response_admissions as admissions

    store = _ResponseStore()
    adapter = _ContextAdapter(store)
    context, error = await admissions.prepare_context(adapter, None, {
        'input': 'follow up',
        'conversation_history': [{'role': 'user', 'content': 'client prefix'}],
    }, None, None, 'api', 'key')
    assert error is None
    run_kwargs = context['run_kwargs']
    assert run_kwargs['history_not_transcript'] is True
    assert 'history_from_store' not in run_kwargs


@pytest.mark.asyncio
async def test_prepare_context_marks_multi_item_store_turn_as_not_transcript(monkeypatch):
    """Prefix rows are only in the snapshot; dropping it at execution would swallow them."""
    from gateway.platforms import api_server_response_admissions as admissions

    monkeypatch.setattr(admissions, '_stored_owner', lambda adapter, sid: sid)

    async def _continued(adapter, sid, request_id=None):
        return sid, None
    monkeypatch.setattr('gateway.platforms.api_server_authority_runs.continued_session', _continued)

    store = _ResponseStore(response={'conversation_history': SNAPSHOT, 'session_id': None})
    adapter = _ContextAdapter(store)
    context, error = await admissions.prepare_context(adapter, None, {
        'previous_response_id': 'resp-1',
        'input': [{'role': 'user', 'content': 'prefix'}, {'role': 'user', 'content': 'follow up'}],
    }, None, None, 'api', 'key')
    assert error is None
    run_kwargs = context['run_kwargs']
    assert 'history_from_store' not in run_kwargs
    assert run_kwargs['history_not_transcript'] is True
    assert run_kwargs['conversation_history'] == SNAPSHOT + [{'role': 'user', 'content': 'prefix'}]


@pytest.mark.asyncio
async def test_prepare_context_leaves_plain_turn_unmarked():
    from gateway.platforms import api_server_response_admissions as admissions

    adapter = _ContextAdapter(_ResponseStore())
    context, error = await admissions.prepare_context(
        adapter, None, {'input': 'follow up'}, None, None, 'api', 'key')
    assert error is None
    run_kwargs = context['run_kwargs']
    assert 'history_from_store' not in run_kwargs
    assert 'history_not_transcript' not in run_kwargs


@pytest.mark.asyncio
async def test_prepare_context_client_history_wins_over_store_snapshot(monkeypatch):
    """An explicit conversation_history is client-owned even when the chain also names a response."""
    from gateway.platforms import api_server_response_admissions as admissions

    monkeypatch.setattr(admissions, '_stored_owner', lambda adapter, sid: sid)

    async def _continued(adapter, sid, request_id=None):
        return sid, None
    monkeypatch.setattr('gateway.platforms.api_server_authority_runs.continued_session', _continued)

    store = _ResponseStore(response={'conversation_history': SNAPSHOT, 'session_id': None})
    adapter = _ContextAdapter(store)
    context, error = await admissions.prepare_context(adapter, None, {
        'previous_response_id': 'resp-1', 'input': 'follow up',
        'conversation_history': [{'role': 'user', 'content': 'client prefix'}],
    }, None, None, 'api', 'key')
    assert error is None
    run_kwargs = context['run_kwargs']
    assert 'history_from_store' not in run_kwargs
    assert run_kwargs['history_not_transcript'] is True


# --- transport / retirement ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_forwards_provenance_to_canonical_turn(api, monkeypatch):
    from gateway import session_api_turn

    seen = {}

    async def _run(adapter, **kwargs):
        seen.update(kwargs)
        return {'final_response': 'ok'}, {}

    monkeypatch.setattr(session_api_turn, 'run_api_turn', _run)
    await api._run_agent('follow up', SNAPSHOT, session_id='forward',
                         history_from_store=True, history_not_transcript=True)
    assert seen['history_from_store'] is True
    assert seen['history_not_transcript'] is True


def test_retire_routes_clears_only_that_sessions_carrier_key(owner):
    from hermes_state_mutation_retirement import retire_routes

    owner.db.create_session(session_id='retire-carrier', source='api_server')
    owner.db.create_session(session_id='keep-carrier', source='api_server')
    owner.db.set_meta(_carrier('retire-carrier'), '1')
    owner.db.set_meta(_carrier('keep-carrier'), '1')

    owner.db._execute_write(lambda conn: retire_routes(conn, ['retire-carrier']))

    assert owner.db.get_meta(_carrier('retire-carrier')) is None
    assert owner.db.get_meta(_carrier('keep-carrier')) == '1'
