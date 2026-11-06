"""`config.set key="project_skills"` — the cockpit's trust card, answered through the owner.

The agent lives in the gateway process, so that is where the decision has to land: it writes the
profile's own config.yaml (atomic replace, the same file `hermes skills trust` maintains) and drops
the cached skills section, otherwise the next turn would still build its prompt from the old answer.
The repo comes from the SESSION's policy cwd — a cockpit session is usually in a repo of its own.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import session_tui_config


def _repo(tmp_path):
    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    for name in ("deploy", "triage"):
        d = root / ".agents" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n\nbody\n")
    return root


@pytest.fixture
def wired(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    cfg = {}
    cleared = []
    monkeypatch.setattr(session_tui_config, "_write_profile_config",
                        lambda connection, edit: bool(edit(cfg)))
    monkeypatch.setattr("agent.prompt_builder.clear_skills_system_prompt_cache",
                        lambda **kwargs: cleared.append(kwargs))
    policy = SimpleNamespace(cwd=str(repo))
    return repo, cfg, cleared, policy


def test_trusting_the_sessions_repo_writes_the_profile_and_drops_the_cache(wired):
    repo, cfg, cleared, policy = wired
    result = session_tui_config._set_project_skills(None, policy, "trusted")
    assert result == {"key": "project_skills", "value": "trusted", "root": str(repo), "changed": True}
    assert cfg["skills"]["trusted_project_dirs"] == [str(repo)]
    assert cleared, "the next turn would otherwise reuse the prompt built without these skills"


def test_a_repeat_answer_says_nothing_changed_and_keeps_the_cache(wired):
    repo, cfg, cleared, policy = wired
    session_tui_config._set_project_skills(None, policy, "trusted")
    cleared.clear()
    assert session_tui_config._set_project_skills(None, policy, "trusted")["changed"] is False
    assert cleared == []


def test_never_ask_records_the_answer_and_untrusts(wired):
    repo, cfg, _cleared, policy = wired
    session_tui_config._set_project_skills(None, policy, "trusted")
    session_tui_config._set_project_skills(None, policy, "ignore")
    assert cfg["skills"]["trusted_project_dirs"] == []
    assert cfg["skills"]["project_skills_ignored"] == [str(repo)]


def test_a_profile_level_call_has_no_repo_to_answer_for(wired):
    """Without a session there is no cwd, so there is no honest answer: refuse, never the gateway's
    own cwd (that would trust whatever directory the daemon happens to run in)."""
    _repo, cfg, _cleared, _policy = wired
    with pytest.raises(session_tui_config.RuntimeStoreError):
        session_tui_config._set_project_skills(None, None, "trusted")
    assert cfg == {}


def test_an_unknown_value_is_refused(wired):
    _repo, cfg, _cleared, policy = wired
    with pytest.raises(session_tui_config.RuntimeStoreError):
        session_tui_config._set_project_skills(None, policy, "sure")
    assert cfg == {}
