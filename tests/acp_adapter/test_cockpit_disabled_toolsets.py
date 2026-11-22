"""Per-session `disabled_toolsets` from a cockpit editor (Agentus disables `todo`).

The cockpit must disable its own plan tool's rival (`todo_list`, in the `todo` toolset) ONLY for
sessions it drives over ACP — a CLI/cron/Studio session must keep its tools. The editor key freezes
into the session policy at create, and a resume must not silently re-enable it from the profile.
"""
import json

import pytest

from gateway.session_local_editor import validate_editor
from gateway.session_policy import build_policy
from gateway.session_policy_refresh import refreshed_policy
from hermes_state_runtime import RuntimeStoreError


def _policy(tmp_path, **editor):
    return build_policy(
        {'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture', 'editor': editor},
        {},
    )


def test_validate_editor_accepts_disabled_toolsets(tmp_path):
    validate_editor('acp', {'disabled_toolsets': ['todo']})


@pytest.mark.parametrize('bad', [['todo', 3], 'todo', {'todo': True}])
def test_validate_editor_rejects_malformed_disabled_toolsets(bad):
    with pytest.raises(RuntimeStoreError):
        validate_editor('acp', {'disabled_toolsets': bad})


def test_build_policy_freezes_editor_disabled_toolsets(tmp_path):
    policy = _policy(tmp_path, disabled_toolsets=['todo'])
    config = json.loads(policy.config_json)
    assert config['agent']['disabled_toolsets'] == ['todo']


def test_no_editor_key_leaves_config_without_disabled(tmp_path):
    policy = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture'}, {})
    config = json.loads(policy.config_json)
    assert 'disabled_toolsets' not in config.get('agent', {})


def test_non_acp_source_refuses_editor(tmp_path):
    with pytest.raises(RuntimeStoreError):
        validate_editor('cli', {'disabled_toolsets': ['todo']})


def test_refresh_does_not_reenable_editor_disabled_toolsets(tmp_path):
    """A resume re-reads the profile; the cockpit's frozen disable must survive it."""
    policy = _policy(tmp_path, disabled_toolsets=['todo'])
    # The profile's config (live) does NOT disable todo — a refresh must not copy that back over the
    # editor's explicit selection.
    live = {'agent': {'disabled_toolsets': []}}
    refreshed, _, _ = refreshed_policy(policy, live, json.loads(policy.terminal_json))
    config = json.loads(refreshed.config_json)
    assert config['agent']['disabled_toolsets'] == ['todo']


def test_refresh_follows_profile_when_no_editor_selection(tmp_path):
    """A normal ACP session (no editor disable) still follows the profile on resume."""
    policy = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture'}, {})
    live = {'agent': {'disabled_toolsets': ['project']}}
    refreshed, _, _ = refreshed_policy(policy, live, json.loads(policy.terminal_json))
    config = json.loads(refreshed.config_json)
    assert config['agent']['disabled_toolsets'] == ['project']


# ── B1: a cockpit editor can only ADD disables, never revoke the operator's ───────────────────
def test_build_policy_MERGES_with_profile_disabled_not_replaces(tmp_path):
    """B1 regression: editor `['todo']` must not erase profile `['terminal','delegate']`."""
    profile = {'agent': {'disabled_toolsets': ['terminal', 'delegate']}}
    policy = build_policy(
        {'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture',
         'editor': {'disabled_toolsets': ['todo']}},
        profile,
    )
    config = json.loads(policy.config_json)
    assert set(config['agent']['disabled_toolsets']) == {'terminal', 'delegate', 'todo'}


def test_refresh_preserves_profile_gain_and_cockpit_disable(tmp_path):
    """B1 regression on resume: profile gained `project`, cockpit still has `todo`."""
    policy = build_policy(
        {'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture',
         'editor': {'disabled_toolsets': ['todo']}},
        {'agent': {'disabled_toolsets': ['terminal']}},
    )
    live = {'agent': {'disabled_toolsets': ['terminal', 'project']}}
    refreshed, _, _ = refreshed_policy(policy, live, json.loads(policy.terminal_json))
    config = json.loads(refreshed.config_json)
    # Gained `project` followed, operator's `terminal` kept, cockpit's `todo` kept.
    assert set(config['agent']['disabled_toolsets']) == {'terminal', 'project', 'todo'}


@pytest.mark.parametrize('bad', [['all'], ['*'], ['todo', ''], ['a'] * 17])
def test_validate_editor_rejects_wildcards_and_oversize(bad):
    """N1: no `all`/`*` wildcard, no unbounded list smuggled into the policy."""
    with pytest.raises(RuntimeStoreError):
        validate_editor('acp', {'disabled_toolsets': bad})


def test_merge_helper_drops_wildcards_keeps_profile(tmp_path):
    from gateway.session_local_editor import merge_cockpit_disabled
    out = merge_cockpit_disabled(['terminal'], ['todo', 'all', '*', 3, 'todo'])
    assert set(out) == {'terminal', 'todo'}


def test_merge_helper_NEVER_truncates_profile_disable(tmp_path):
    """B1-NEW regression: a >16-entry profile must keep ALL its entries; only the cockpit request is capped."""
    from gateway.session_local_editor import merge_cockpit_disabled
    profile = [f'z{i:02d}' for i in range(20)]          # 20 operator-disabled toolsets
    attack = [f'a{i:02d}' for i in range(16)]           # 16 lexically-earlier names from a client
    out = merge_cockpit_disabled(profile, attack)
    # Every operator entry survives (no truncation of the authoritative side).
    assert set(profile) <= set(out)
    # The cockpit additions are capped at 16, no wildcards/duplicates.
    assert len([x for x in out if x.startswith('a')]) <= 16
    # Even a single harmless request must not drop the tail of a >16 profile.
    out2 = merge_cockpit_disabled(profile, ['todo'])
    assert set(profile) <= set(out2) and 'todo' in out2


def test_build_policy_keeps_all_profile_disables_over_16(tmp_path):
    """B1-NEW end-to-end via build_policy: profile's 20 disables all survive."""
    profile_entries = [f'z{i:02d}' for i in range(20)]
    profile = {'agent': {'disabled_toolsets': profile_entries}}
    policy = build_policy(
        {'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture',
         'editor': {'disabled_toolsets': ['todo']}},
        profile,
    )
    config = json.loads(policy.config_json)
    assert set(profile_entries) <= set(config['agent']['disabled_toolsets'])
    assert 'todo' in config['agent']['disabled_toolsets']

