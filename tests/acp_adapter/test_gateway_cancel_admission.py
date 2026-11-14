"""session/cancel stops this session's turn: our own admission first, otherwise the running generation."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from acp.schema import TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError


def agent_with_queued_prompt():
    agent = GatewayACPAgent()
    # Another viewer's turn is executing under generation 7 when we are queued.
    agent._snapshots['s'] = {'execution_generation': 7}
    agent._gateway = AsyncMock()
    agent._conn = AsyncMock()
    return agent


async def queued(agent):
    agent._gateway.rpc.return_value = {'admission_id': 'ours'}
    task = asyncio.create_task(agent.prompt([TextContentBlock(type='text', text='hello')], 's'))
    await asyncio.sleep(0)
    assert not task.done()
    agent._gateway.rpc.reset_mock()
    return task


async def settle(agent, task, outcome):
    await agent._project({'session_id': 's', 'type': 'message.complete', 'admission_id': 'ours',
                          'payload': {'outcome': outcome, 'text': ''}})
    return await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_cancel_targets_queued_acp_admission_not_the_running_generation():
    agent = agent_with_queued_prompt()
    task = await queued(agent)
    agent._gateway.rpc.return_value = {'admission_id': 'ours', 'status': 'terminal', 'outcome': 'cancelled'}
    await agent.cancel('s')
    methods = [call.args[0] for call in agent._gateway.rpc.await_args_list]
    assert methods == ['prompt.cancel']
    assert agent._gateway.rpc.await_args.kwargs == {'session_id': 's', 'admission_id': 'ours'}
    assert (await settle(agent, task, 'cancelled')).stop_reason == 'cancelled'


@pytest.mark.asyncio
async def test_cancel_interrupts_only_when_our_admission_is_the_one_running():
    agent = agent_with_queued_prompt()
    task = await queued(agent)

    async def rpc(method, **params):
        if method == 'prompt.cancel':
            raise GatewayClientError('stale_generation')
        if method == 'prompt.receipt':
            return {'admission_id': 'ours', 'status': 'started', 'execution_generation': 8}
        return {}
    agent._gateway.rpc.side_effect = rpc
    await agent.cancel('s')
    calls = {call.args[0]: call.kwargs for call in agent._gateway.rpc.await_args_list}
    assert calls['session.interrupt'] == {'session_id': 's', 'execution_generation': 8}
    assert (await settle(agent, task, 'cancelled')).stop_reason == 'cancelled'


@pytest.mark.asyncio
async def test_cancel_during_submit_is_applied_when_admission_id_arrives():
    agent = agent_with_queued_prompt()
    submit_started = asyncio.Event()
    release_submit = asyncio.Event()
    calls = []

    async def rpc(method, **params):
        calls.append((method, params))
        if method == 'prompt.submit':
            submit_started.set()
            await release_submit.wait()
            return {'admission_id': 'ours'}
        if method == 'prompt.cancel':
            return {'admission_id': 'ours', 'status': 'terminal', 'outcome': 'cancelled'}
        return {}

    agent._gateway.rpc.side_effect = rpc
    task = asyncio.create_task(agent.prompt([TextContentBlock(type='text', text='hello')], 's'))
    await asyncio.wait_for(submit_started.wait(), 2)
    await agent.cancel('s')
    assert [method for method, _ in calls] == ['prompt.submit']

    release_submit.set()
    for _ in range(20):
        if any(method == 'prompt.cancel' for method, _ in calls):
            break
        await asyncio.sleep(0)
    assert ('prompt.cancel', {'session_id': 's', 'admission_id': 'ours'}) in calls
    assert (await settle(agent, task, 'cancelled')).stop_reason == 'cancelled'


@pytest.mark.asyncio
async def test_cancel_without_an_acp_prompt_stops_the_sessions_running_turn():
    """#108: stop belongs to the session, not to the surface that submitted the prompt.

    A gateway-owned session can be woken by a notice, a scheduler, or another viewer; the operator
    watching it has one Stop button and it has to reach that turn. Returning silently here (the old
    contract) made the cockpit report a stop that never happened.
    """
    agent = agent_with_queued_prompt()
    await agent.cancel('s')
    agent._gateway.rpc.assert_awaited_once_with(
        'session.interrupt', session_id='s', execution_generation=7)


@pytest.mark.asyncio
async def test_cancel_without_a_generation_has_nothing_to_interrupt():
    agent = agent_with_queued_prompt()
    agent._snapshots['s'] = {}
    await agent.cancel('s')
    agent._gateway.rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_of_a_turn_that_just_settled_is_not_an_error():
    """`stale_generation` means the turn the operator wanted gone is gone: report nothing."""
    agent = agent_with_queued_prompt()
    agent._gateway.rpc.side_effect = GatewayClientError('stale_generation')
    await agent.cancel('s')
    agent._gateway.rpc.assert_awaited_once_with(
        'session.interrupt', session_id='s', execution_generation=7)


@pytest.mark.asyncio
async def test_cancel_never_retargets_a_newer_generation():
    """A stale answer must NOT be retried at the generation that replaced it.

    The operator aimed at the turn they could see. Re-reading the snapshot after `stale_generation`
    (what this did first) would substitute the successor generation and stop a stranger's fresh work
    — the same substitution `_cancel_admission` forbids. Here the event pump reports generation 8
    while the interrupt is in flight; only 7 may ever be interrupted.
    """
    agent = agent_with_queued_prompt()

    async def rpc(method, **params):
        if method == 'session.interrupt':
            # The pump advanced the snapshot before the answer came back.
            agent._snapshots['s']['execution_generation'] = 8
            raise GatewayClientError('stale_generation')
        return {}
    agent._gateway.rpc.side_effect = rpc
    await agent.cancel('s')
    generations = [call.kwargs['execution_generation'] for call in agent._gateway.rpc.await_args_list]
    assert generations == [7]


@pytest.mark.asyncio
async def test_cancel_surfaces_a_real_interrupt_failure():
    agent = agent_with_queued_prompt()
    agent._gateway.rpc.side_effect = GatewayClientError('not_authorized')
    with pytest.raises(GatewayClientError, match='not_authorized'):
        await agent.cancel('s')

@pytest.mark.asyncio
async def test_deferred_cancel_failure_cleans_mapping_before_next_submit():
    agent = agent_with_queued_prompt()
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()
    calls = []
    submits = 0

    async def rpc(method, **params):
        nonlocal submits
        calls.append((method, params))
        if method == 'prompt.submit':
            submits += 1
            if submits == 1:
                first_started.set()
                await first_release.wait()
                return {'admission_id': 'first'}
            second_started.set()
            await second_release.wait()
            return {'admission_id': 'second'}
        if method == 'prompt.cancel':
            if params['admission_id'] == 'first':
                raise GatewayClientError('cancel_failed')
            return {'admission_id': 'second', 'status': 'terminal', 'outcome': 'cancelled'}
        return {}

    agent._gateway.rpc.side_effect = rpc

    first = asyncio.create_task(
        agent.prompt([TextContentBlock(type='text', text='first')], 's'))
    await asyncio.wait_for(first_started.wait(), 2)
    await agent.cancel('s')
    first_release.set()
    with pytest.raises(GatewayClientError, match='cancel_failed'):
        await asyncio.wait_for(first, 2)
    assert 's' not in agent._admissions

    second = asyncio.create_task(
        agent.prompt([TextContentBlock(type='text', text='second')], 's'))
    await asyncio.wait_for(second_started.wait(), 2)
    await agent.cancel('s')
    second_release.set()

    for _ in range(20):
        if any(method == 'prompt.cancel' and params.get('admission_id') == 'second'
               for method, params in calls):
            break
        await asyncio.sleep(0)

    cancelled = [params['admission_id'] for method, params in calls if method == 'prompt.cancel']
    assert cancelled == ['first', 'second']

    # The target assertion is the contract under test. Retire the synthetic
    # second prompt explicitly rather than manufacturing a terminal event.
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert 's' not in agent._admissions


@pytest.mark.asyncio
async def test_interrupt_completion_race_rechecks_exact_admission():
    agent = agent_with_queued_prompt()
    receipts = iter([
        {'admission_id': 'ours', 'status': 'started', 'execution_generation': 8},
        {'admission_id': 'ours', 'status': 'terminal', 'outcome': 'completed',
         'execution_generation': 8},
    ])
    calls = []

    async def rpc(method, **params):
        calls.append((method, params))
        if method == 'prompt.cancel':
            raise GatewayClientError('stale_generation')
        if method == 'prompt.receipt':
            return next(receipts)
        if method == 'session.interrupt':
            raise GatewayClientError('stale_generation')
        return {}

    agent._gateway.rpc.side_effect = rpc
    await agent._cancel_admission('s', 'ours')

    assert calls == [
        ('prompt.cancel', {'session_id': 's', 'admission_id': 'ours'}),
        ('prompt.receipt', {'session_id': 's', 'admission_id': 'ours'}),
        ('session.interrupt', {'session_id': 's', 'execution_generation': 8}),
        ('prompt.receipt', {'session_id': 's', 'admission_id': 'ours'}),
    ]
