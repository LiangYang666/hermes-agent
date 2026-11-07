"""The bystander role reaches the OWNER: a per-session request, not a process-wide env read.

Under 0.22 every ACP session *was* its own process, so `AGENTUS_SESSION_ROLE=observer` could be read
where the agent was built. In 0.23 the agent is built by a shared owner that never inherits a
viewer's environment — so the role has to be frozen into the session's policy at `session.create`,
or an observer silently gets recall, self-review and memory writes.
"""
import pytest

from acp_adapter.gateway_server import GatewayACPAgent, _observing_session


class _Client:
    def __init__(self):
        self.calls = []

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        if method == 'runtime.describe':
            return {'session_create': {'sources': ['acp']}, 'capabilities': ['acp-editor-policy-v1']}
        if method == 'session.create':
            return {'session_id': 'new', 'info': {'cwd': '/w', 'model': 'm', 'provider': 'p'},
                    'messages': [], 'prompts': []}
        if method == 'session.info':
            return {'cwd': '/w', 'model': 'm', 'provider': 'p'}
        return {}


@pytest.fixture
def agent(monkeypatch, tmp_path):
    a = GatewayACPAgent()
    a._gateway = _Client()
    a._home = tmp_path
    a._failure = None
    monkeypatch.setattr(a, '_transport_closed', lambda: False)
    async def adopt(*args, **kwargs):
        return None

    async def live_reasoning(*args, **kwargs):
        return None

    monkeypatch.setattr(a, '_adopt', adopt)
    monkeypatch.setattr(a, '_request_live_reasoning', live_reasoning)

    async def fields(session_id, info, editor_id):
        return {}

    monkeypatch.setattr(a, '_session_fields', fields)
    return a


def _create_call(agent):
    return next(params for method, params in agent._gateway.calls if method == 'session.create')


def test_the_role_is_read_from_this_viewers_environment(monkeypatch):
    monkeypatch.setenv('AGENTUS_SESSION_ROLE', 'Observer ')
    assert _observing_session() is True
    monkeypatch.setenv('AGENTUS_SESSION_ROLE', 'primary')
    assert _observing_session() is False
    monkeypatch.delenv('AGENTUS_SESSION_ROLE')
    assert _observing_session() is False


@pytest.mark.asyncio
async def test_a_bystander_viewer_asks_for_a_bystander_session(agent, monkeypatch):
    monkeypatch.setenv('AGENTUS_SESSION_ROLE', 'observer')
    await agent.new_session(cwd='/w', mcp_servers=[])
    assert _create_call(agent)['observer'] is True


@pytest.mark.asyncio
async def test_an_ordinary_viewer_asks_for_nothing(agent, monkeypatch):
    monkeypatch.delenv('AGENTUS_SESSION_ROLE', raising=False)
    await agent.new_session(cwd='/w', mcp_servers=[])
    assert 'observer' not in _create_call(agent)
