"""Zed renders its model and effort pickers only from ``configOptions`` (#143547 ACP-24)."""
import pytest
from acp.schema import ModelInfo, SessionModelState

from acp_adapter.server import HermesACPAgent


@pytest.mark.asyncio
async def test_effort_pick_applies_through_the_reasoning_mutation_and_returns_the_pickers(monkeypatch):
    monkeypatch.setattr('acp_adapter.gateway_session_meta.model_state', lambda info: SessionModelState(
        available_models=[ModelInfo(model_id='openrouter:fixture', name='fixture')], current_model_id='openrouter:fixture'))
    info = {'model': 'fixture', 'provider': 'openrouter', 'reasoning_effort': 'medium'}
    mutations = []

    class RPC:
        async def rpc(self, method, **params):
            if method == 'session.mutate':
                mutations.append((params['operation'], params['payload']))
                info['reasoning_effort'] = params['payload']['effort']
                return {'status': 'applied'}
            if method == 'session.info':
                return dict(info)
            return {'revision': 1, 'execution_generation': 1, 'info': dict(info)}

    agent = HermesACPAgent()
    agent._gateway = RPC()
    agent._snapshots['s'] = {'revision': 1, 'execution_generation': 1}
    reply = await agent.set_config_option(config_id='reasoning_effort', session_id='s', value='high')
    assert mutations == [('reasoning', {'effort': 'high'})]
    pickers = {option.category: option.current_value for option in reply.config_options}
    assert pickers == {'mode': 'ask', 'model': 'openrouter:fixture', 'thought_level': 'high', 'context': 'auto'}


def test_the_effort_picker_offers_default_only_while_no_level_is_set():
    from acp_adapter.gateway_session_meta import config_options
    from acp_adapter.modes import DEFAULT_MODE

    def effort(level):
        option = {option.id: option for option in config_options(None, {'reasoning_effort': level},
                                                                DEFAULT_MODE)}['reasoning_effort']
        return option.current_value, [choice.value for choice in option.options]
    # ``default`` (a reset) reads back as the profile's level: offering it would make the pick jump.
    current, values = effort('medium')
    assert current == 'medium' and 'default' not in values
    current, values = effort('')
    assert current == 'default' and values[0] == 'default'


@pytest.mark.asyncio
async def test_the_project_skills_card_answers_through_the_owner(monkeypatch, tmp_path):
    """The viewer shows the row; the gateway owns the config file and the agent's cached skills, so
    the answer is a ``config.set`` RPC — never a local write in this process."""
    monkeypatch.setattr('acp_adapter.gateway_session_meta.model_state', lambda info: None)
    info = {'model': 'fixture', 'provider': 'openrouter', 'reasoning_effort': 'medium'}
    calls = []

    class RPC:
        async def rpc(self, method, **params):
            calls.append((method, params))
            if method == 'session.info':
                return dict(info)
            return {'key': 'project_skills', 'value': 'trusted', 'root': '/repo', 'changed': True}

    agent = HermesACPAgent()
    agent._gateway = RPC()
    agent._snapshots['s'] = {'revision': 1, 'execution_generation': 1}
    await agent.set_config_option(config_id='project_skills', session_id='s', value='trusted')
    assert calls[0] == ('config.set', {'session_id': 's', 'key': 'project_skills',
                                       'value': 'trusted'})

    with pytest.raises(Exception):
        await agent.set_config_option(config_id='project_skills', session_id='s', value='sure')
