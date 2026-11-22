"""Editor policy lives with execution; an ACP connection is only a viewer."""
from contextlib import contextmanager
from dataclasses import asdict
import uuid

from hermes_state_runtime import RuntimeStoreError


# A cockpit editor may add toolsets to the per-session disable, never remove the profile's. These
# guards keep an arbitrary ACP _meta client from (a) disabling everything via a wildcard, or (b)
# smuggling an unbounded list into the persisted policy. Reused by build_policy and the resume
# refresh so both paths enforce the same "only ever fewer tools" invariant.
COCKPIT_DISABLE_MAX = 16
_COCKPIT_DISABLE_FORBIDDEN = frozenset({'all', '*', ''})


def merge_cockpit_disabled(existing, requested):
    """Union of the profile's disabled toolsets and a cockpit editor's request, sorted + bounded.

    `requested` entries that are wildcards or non-strings are dropped. Cap the total so a hostile
    client cannot persist an enormous list. Existing (profile) entries are always kept — this is the
    capability-escalation guard: an editor can only ADD disables, never clear one the operator set.
    """
    from agent.skill_utils import parse_config_string_list
    merged = set(parse_config_string_list(existing))
    if isinstance(requested, list):
        for item in requested[:COCKPIT_DISABLE_MAX]:
            if isinstance(item, str):
                name = item.strip().lower()
                if name and name not in _COCKPIT_DISABLE_FORBIDDEN:
                    merged.add(name)
    return sorted(merged)[:COCKPIT_DISABLE_MAX]


def validate_editor(source, editor):
    """The launch's editor policy, or a refusal that names the key it read (create path totality).

    An ACP connection is a viewer; the policy itself lives with execution. Every branch below refuses
    with its own field, so a rejected editor launch says which key was wrong instead of a bare code.
    """
    if editor is None:
        return
    if source != 'acp':
        raise RuntimeStoreError('invalid_params', field='editor',
                                detail=f'an editor policy is only accepted from an ACP launch, not {source}')
    if not isinstance(editor, dict):
        raise RuntimeStoreError('invalid_params', field='editor',
                                detail=f'must be a mapping, got {type(editor).__name__}')
    unknown = sorted(set(editor) - {'mcp_servers', 'edit_approval_policy', 'disabled_toolsets'})
    if unknown:
        raise RuntimeStoreError('invalid_params', field=f'editor.{unknown[0]}',
                                detail='unknown editor key')
    if editor.get('edit_approval_policy', 'ask') != 'ask':
        raise RuntimeStoreError('invalid_params', field='editor.edit_approval_policy',
                                detail="must be 'ask': every edit is approved one at a time")
    # A cockpit editor may ask that built-in toolsets be disabled for THIS session only (e.g. the
    # Agentus cockpit disables `todo` so its own plan MCP tool is the one the operator watches).
    # Frozen into the session policy below, never read from this process's global config — a CLI or
    # cron session without this editor key keeps its own toolsets.
    dts = editor.get('disabled_toolsets', [])
    if not isinstance(dts, list):
        raise RuntimeStoreError('invalid_params', field='editor.disabled_toolsets',
                                detail=f'must be a list of toolset names, got {type(dts).__name__}')
    if len(dts) > COCKPIT_DISABLE_MAX:
        raise RuntimeStoreError('invalid_params', field='editor.disabled_toolsets',
                                detail=f'at most {COCKPIT_DISABLE_MAX} toolsets')
    if any(not isinstance(t, str) or t.strip().lower() in _COCKPIT_DISABLE_FORBIDDEN for t in dts):
        raise RuntimeStoreError('invalid_params', field='editor.disabled_toolsets',
                                detail='every entry must be a non-empty toolset name (no all/*)')
    if not isinstance(editor.get('mcp_servers', []), list):
        raise RuntimeStoreError('invalid_params', field='editor.mcp_servers',
                                detail=f'must be a list, got {type(editor.get("mcp_servers")).__name__}')
    from gateway.session_local_mcp import validate_servers
    validate_servers(editor.get('mcp_servers', []))


def request_editor_edit(proposal):
    from tools import approval
    from tools.approval_context import get_current_session_key
    from tools.approval_gateway_wait import _await_gateway_decision
    route = get_current_session_key()
    with approval._lock:
        notify = approval._gateway_notify_cbs.get(route)
    if notify is None:
        return False
    # Each diff is a one-use consent, never a command-pattern or session grant.
    data = {'command': f'Edit {proposal.path}', 'description': 'Approve proposed file change',
            'pattern_key': 'editor-' + uuid.uuid4().hex,
            'allow_session': False, 'allow_permanent': False, 'edit': asdict(proposal)}
    data['pattern_keys'] = [data['pattern_key']]
    decision = _await_gateway_decision(route, notify, data, surface='acp')
    return decision.get('resolved') and decision.get('choice') == 'once'


def _policy_requester(policy):
    """The session's edit approval policy in front of the editor's consent (main's ACP modes): Accept
    Edits / Don't Ask let an edit run unasked; sensitive paths and everything else still ask."""
    from acp_adapter.edit_approval import should_auto_approve_edit
    if policy.edit_approval_policy == 'ask':
        return request_editor_edit

    def request(proposal):
        if should_auto_approve_edit(proposal, policy.edit_approval_policy, policy.cwd):
            return True
        return request_editor_edit(proposal)
    return request


@contextmanager
def editor_scope(policy):
    from acp_adapter.edit_approval import set_edit_approval_requester, reset_edit_approval_requester
    token = set_edit_approval_requester(_policy_requester(policy) if policy.source == 'acp' else None)
    try:
        yield
    finally:
        reset_edit_approval_requester(token)
