"""A guarded model switch must be a FYI, not an approval card (#Studio 0.26.1).

The operator already picked the model in the picker; asking them to click "switch anyway"
a second time is the card the user complained about. The ACP surface's confirmer should
auto-accept the owner's guard and report the guard's reason as a plain message — never a
``session/request_permission``. Non-interactive (no editor connection) still declines.
"""
import pytest
from unittest.mock import AsyncMock

from acp_adapter.gateway_server import GatewayACPAgent

REFUSAL = {
    'status': 'confirmation_required',
    'confirm': 'tok-1',
    'target_model': 'second-model',
    'target_provider': 'custom:second',
    'confirm_message': '!!! LARGE CONTEXT MODEL SWITCH !!!\nThis session holds ~250,000 tokens.',
    'warnings': [{'kind': 'context_cache', 'title': 'Large Context Switch Warning', 'message': 'large'}],
}


def agent(with_conn=True):
    a = GatewayACPAgent()
    a._snapshots['s'] = {}
    a._conn = AsyncMock() if with_conn else None
    return a


def shown(a):
    texts = []
    for call in a._conn.session_update.await_args_list:
        upd = call.kwargs.get('update')
        content = getattr(upd, 'content', None)
        if content is None:
            continue
        blocks = content if isinstance(content, (list, tuple)) else [content]
        for block in blocks:
            t = getattr(block, 'text', None)
            if t:
                texts.append(t)
    return "\n".join(texts)


@pytest.mark.asyncio
async def test_guarded_switch_does_not_raise_an_approval_card():
    """The whole point: no session/request_permission frame for a model pick."""
    a = agent()
    confirm = a._model_confirmer('s')
    assert confirm is not None
    assert await confirm(REFUSAL) is True
    a._conn.request_permission.assert_not_called()


@pytest.mark.asyncio
async def test_guarded_switch_reports_the_guard_reason_without_claiming_success():
    """B1: the FYI is emitted BEFORE the confirm-token re-send, which can still fail. It must report
    the guard's reason only — never 'switching/switched to X' — so a later 'not switched' message
    cannot contradict it. The model name is left to the authoritative success/failure message."""
    a = agent()
    confirm = a._model_confirmer('s')
    await confirm(REFUSAL)
    text = shown(a)
    assert 'CONTEXT' in text.upper() or 'SWITCH' in text.upper(), text  # guard reason present
    low = text.lower()
    assert 'switching' not in low and 'switched' not in low and 'second-model' not in low, text


@pytest.mark.asyncio
async def test_guarded_switch_auto_accepts_even_without_any_click():
    """Old code returned True only on an 'allow_once' permission answer; a deny meant no switch.
    New code accepts from the picker action itself — no answer to wait for."""
    a = agent()
    a._conn.request_permission.return_value = type('R', (), {'outcome': None})()
    confirm = a._model_confirmer('s')
    assert await confirm(REFUSAL) is True
    a._conn.request_permission.assert_not_called()


def test_no_editor_connection_still_declines():
    """Non-interactive surfaces (no editor) must not silently switch a guarded model."""
    assert agent(with_conn=False)._model_confirmer('s') is None



@pytest.mark.asyncio
async def test_apply_still_raises_on_stale_token_without_asking_twice():
    """N4(a): with a confirmer that always agrees, a second confirmation_required (stale token /
    another pick landed) must still raise and must NOT re-prompt. The neutral FYI already emitted
    stays consistent with the resulting 'not switched' error."""
    from unittest.mock import AsyncMock
    from hermes_cli.gateway_mutations import PreparedMutations, GatewayClientError

    client = AsyncMock()
    # resume -> snapshot; first mutate -> confirmation_required; re-sent mutate -> confirmation_required again
    client.rpc.side_effect = [
        {'revision': 1, 'execution_generation': 1},
        {'status': 'confirmation_required', 'confirm': 'tok-1', 'confirm_message': 'guard'},
        {'status': 'confirmation_required', 'confirm': 'tok-2', 'confirm_message': 'guard'},
    ]
    confirm = AsyncMock(return_value=True)
    with pytest.raises(GatewayClientError):
        await PreparedMutations().apply(client, 's', 'model', {'model': 'm'}, confirm=confirm)
    confirm.assert_awaited_once()  # never re-asked on the confirmed send
