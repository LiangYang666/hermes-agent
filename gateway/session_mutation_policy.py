"""Prepare the policy-only session edits: session-scope reasoning and the workspace (Move to project).

Both rewrite the frozen LocalSessionPolicy the next turn runs under, re-bound like a model switch
(policy identity covers config/cwd, so its config-secret reference is re-fingerprinted).
"""
import asyncio
import json
from dataclasses import asdict, replace

from hermes_state_runtime import RuntimeStoreError


def _rebind(authority, prepared, old, policy, **extra):
    from gateway.session_policy import bind_launch_key, launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    secrets = recover_config_secrets(authority, old) if old.config_secret_ref else {}
    policy = bind_launch_key(authority, prepared['snapshot']['receipt']['session_id'], policy,
                             launch_key(authority, old), config_secrets=secrets)
    return dict(prepared, policy=asdict(policy), **extra)


async def prepare_reasoning(authority, live, payload, prepared):
    """``/reasoning <level>`` for this session: the level ``build_policy`` freezes at creation. ``None``
    (``/reasoning reset``) gives the level back to the profile."""
    from gateway.session_policy import pin_reasoning_effort, profile_config, restore_policy, unpin_reasoning
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    frozen = old.config()
    if payload['effort'] is not None:
        pin_reasoning_effort(frozen, payload['effort'], session=True)
    else:
        # A bypass launch never reads the profile: its reset is the built-in default.
        profile = {} if old.ignore_user_config else await asyncio.to_thread(profile_config, authority)
        unpin_reasoning(frozen, profile)
    return _rebind(authority, prepared, old, replace(old, config_json=json.dumps(frozen)))


async def prepare_workspace(authority, live, payload, prepared):
    """Move to project: execution cwd and TERMINAL_CWD move together (restore_policy's invariant)."""
    import os
    from gateway.session_policy import restore_policy
    from hermes_constants import translate_cwd_for_wsl_backend
    from tui_gateway import git_probe
    old = restore_policy(prepared['snapshot']['receipt']['policy'])

    def probe():
        # Off-loop: a stat and up to three `git` spawns.
        cwd = os.path.abspath(os.path.expanduser(translate_cwd_for_wsl_backend(payload['cwd'])))
        if not os.path.isdir(cwd):
            raise RuntimeStoreError('invalid_params')
        return cwd, git_probe.branch(cwd), git_probe.common_repo_root(cwd)
    cwd, branch, root = await asyncio.to_thread(probe)
    terminal = json.loads(old.terminal_json)
    terminal['TERMINAL_CWD'] = cwd
    policy = replace(old, cwd=cwd, terminal_json=json.dumps(terminal))
    return _rebind(authority, prepared, old, policy, git={'branch': branch, 'root': root})


async def prepare_edit_policy(authority, live, payload, prepared):
    """Main's ACP modes (Default / Accept Edits / Don't Ask) as the session's frozen edit approval
    policy. Only an ACP session's edits ask an editor; selecting the current policy writes nothing."""
    from gateway.session_policy import restore_policy
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    if old.source != 'acp':
        raise RuntimeStoreError('invalid_params', field='payload.policy',
                                detail='only an ACP session has an editor edit approval policy')
    if old.edit_approval_policy == payload['policy']:
        return {'status': 'noop', 'edit_approval_policy': old.edit_approval_policy}
    return _rebind(authority, prepared, old, replace(old, edit_approval_policy=payload['policy']))


async def prepare_context_budget(authority, live, payload, prepared):
    """The session's context budget: a ``model.context_length`` pin inside the frozen policy config.

    Session-scoped on purpose — the profile's own pin (``hermes config set model.context_length``)
    is what every other session reads, and a cockpit slot adjusting its budget must not rewrite it.
    ``tokens=None`` drops the pin, handing the window back to the model's own.
    """
    from gateway.session_policy import restore_policy
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    frozen = old.config()
    tokens = payload['tokens']
    model_cfg = frozen.get('model')
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    current = model_cfg.get('context_length')
    changed = current != tokens
    if changed:
        if tokens is None:
            model_cfg.pop('context_length', None)
        else:
            model_cfg['context_length'] = tokens
            frozen['model'] = model_cfg
    # Applied even when the frozen value already matches: the picker can be re-stated over a live
    # agent that was constructed before this pin existed (a session carried over from an older build),
    # and re-pointing the compressor is idempotent.
    await asyncio.to_thread(_apply_context_budget_live, authority, prepared['snapshot']['receipt']['session_id'],
                            tokens)
    if not changed:
        return {'status': 'noop', 'context_length': tokens}
    return _rebind(authority, prepared, old, replace(old, config_json=json.dumps(frozen)))


def _apply_context_budget_live(authority, session_id, tokens) -> None:
    """Push the ceiling into the session's running agent, if this session has one.

    ``model.context_length`` is read ONCE, at construction: the frozen policy above governs the next
    agent this session launches, but a picker change must not look applied while the live turn keeps
    compressing against the old window. Mirrors the model-switch path (``_update_switch_compressor``):
    cache the pin on the agent, re-resolve the window, re-point the compressor, re-check feasibility.
    """
    from gateway.session_contract import SessionRef
    try:
        # A resident agent is re-pointed; a cold session has none, and its next agent is constructed
        # from the frozen policy (``session_context_pin``), so there is nothing to do here.
        agent = authority.agent(SessionRef(authority.profile_id, session_id))
    except Exception:
        return
    if agent is None:
        return
    from agent.agent_init import set_config_context_length
    from agent.model_metadata import get_model_context_length
    set_config_context_length(agent, tokens)
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config
        custom_providers = get_compatible_custom_providers(load_config())
    except Exception:
        custom_providers = None
    api_key = agent.api_key if isinstance(agent.api_key, str) else ''
    window = get_model_context_length(agent.model, base_url=agent.base_url, api_key=api_key,
                                      provider=agent.provider, config_context_length=tokens,
                                      custom_providers=custom_providers)
    agent.context_compressor.update_model(model=agent.model, context_length=window, base_url=agent.base_url,
                                          api_key=agent.api_key, provider=agent.provider, api_mode=agent.api_mode)
    from agent.conversation_compression_feasibility import revalidate_compression_feasibility
    revalidate_compression_feasibility(agent)


PREPARE = {'reasoning': prepare_reasoning, 'workspace': prepare_workspace, 'edit_policy': prepare_edit_policy,
           'context_budget': prepare_context_budget}