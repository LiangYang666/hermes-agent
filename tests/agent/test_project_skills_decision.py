"""A repo's project skills: one decision, three surfaces, one file.

Project-local skills (``<repo>/.agents/skills``) load only for a repo root the operator trusted.
Every non-interactive surface used to skip them silently, so a cockpit could show a
playbook-bearing workspace while the agent saw none of it. The decision now has one writer
(``agent.skill_utils.apply_project_root_decision``) reached from the CLI's load/save pair and from
the gateway's atomic profile replace, and one setter on the wire (``project_skills``).
"""
import pytest

from agent import skill_utils


def _write(root, skills):
    for name in skills:
        d = root / ".agents" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: test skill\n---\n\nbody\n")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    return root


def test_a_repo_with_skills_reports_untrusted_until_the_decision_is_made(repo):
    _write(repo, ["deploy", "triage"])
    assert skill_utils.project_skills_state(repo) == (False, 2)
    skill_utils.apply_project_root_decision(cfg := {}, repo, "trusted")
    assert cfg["trusted_project_dirs"] == [str(repo)]
    assert cfg["project_skills_ignored"] == []


def test_trusting_clears_an_earlier_never_ask(repo):
    """The two lists are opposite answers to one question: a root in both would make the notice
    and the loader disagree, so the newest answer wins."""
    cfg = {"project_skills_ignored": [str(repo)]}
    assert skill_utils.apply_project_root_decision(cfg, repo, "trusted") is True
    assert cfg["trusted_project_dirs"] == [str(repo)]
    assert cfg["project_skills_ignored"] == []


def test_ignoring_untrusts(repo):
    cfg = {"trusted_project_dirs": [str(repo)]}
    skill_utils.apply_project_root_decision(cfg, repo, "ignore")
    assert cfg["trusted_project_dirs"] == []
    assert cfg["project_skills_ignored"] == [str(repo)]


def test_ask_is_undecided_again(repo):
    """`ask` has to clear BOTH answers, or the notice stays suppressed and the row lies."""
    cfg = {"trusted_project_dirs": [str(repo)], "project_skills_ignored": [str(repo)]}
    assert skill_utils.apply_project_root_decision(cfg, repo, "ask") is True
    assert cfg == {"trusted_project_dirs": [], "project_skills_ignored": []}


def test_an_unchanged_decision_changes_nothing(repo):
    cfg = {"trusted_project_dirs": [str(repo)]}
    assert skill_utils.apply_project_root_decision(cfg, repo, "trusted") is False
    assert cfg == {"trusted_project_dirs": [str(repo)]}


def test_an_unknown_value_is_refused_rather_than_written(repo):
    with pytest.raises(ValueError):
        skill_utils.apply_project_root_decision({}, repo, "maybe")
