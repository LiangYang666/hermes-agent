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
