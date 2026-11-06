"""The context-budget picker on the gateway path: one fenced ``context_budget`` mutation per pick."""
import pytest

import acp
from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.gateway_session_meta import (CONTEXT_BUDGET_CONFIG_ID, CONTEXT_BUDGET_PRESETS,
                                             config_options)
from hermes_cli.gateway_client import GatewayClientError


def _agent(rpc):
    from unittest.mock import AsyncMock

    agent = GatewayACPAgent()
    agent._gateway, agent._conn = AsyncMock(), AsyncMock()
    agent._gateway.rpc.side_effect = rpc
    agent._snapshots['s'] = {'execution_generation': 7}
    return agent


@pytest.mark.asyncio
async def test_the_picker_sends_tokens_and_auto_clears_the_pin():
    sent = []

    async def rpc(method, **params):
        if method == 'session.resume':
            return {'revision': 3, 'execution_generation': 7}
        if method == 'session.info':
            return {}
        assert method == 'session.mutate', method
        sent.append((params['operation'], params['payload'], params['expected_generation']))
        return {'session_id': 's', 'operation': 'context_budget', 'revision': 4}

    agent = _agent(rpc)
    await agent.set_config_option(CONTEXT_BUDGET_CONFIG_ID, 's', '262144')
    await agent.set_config_option(CONTEXT_BUDGET_CONFIG_ID, 's', 'auto')
    assert sent == [('context_budget', {'tokens': 262144}, 7), ('context_budget', {'tokens': None}, 7)]
    # A value the picker cannot mean is refused here, never written as a pin.
    sent.clear()
    for bad in ['256K', '0', '-1']:
        with pytest.raises(GatewayClientError):
            await agent.set_config_option(CONTEXT_BUDGET_CONFIG_ID, 's', bad)
    assert sent == []
    # An empty row means the same as ``auto``: the window goes back to the model's own.
    await agent.set_config_option(CONTEXT_BUDGET_CONFIG_ID, 's', '')
    assert sent == [('context_budget', {'tokens': None}, 7)]


def test_the_picker_offers_auto_presets_and_the_sessions_own_value():
    def options(info):
        return {option.id: option for option in config_options(None, info, 'default')}

    budget = options({})[CONTEXT_BUDGET_CONFIG_ID]
    values = [row.value for row in budget.options]
    assert budget.current_value == 'auto'
    assert values == ['auto', *(str(tokens) for tokens in CONTEXT_BUDGET_PRESETS)]
    # A pinned session shows its own value, even one outside the presets.
    pinned = options({'context_length': 131_072})[CONTEXT_BUDGET_CONFIG_ID]
    assert pinned.current_value == '131072'
    assert [row.value for row in pinned.options][-1] == '131072'
