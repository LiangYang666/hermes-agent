"""Project skills over ACP: the row, the notice, and the answer's route.

The row is only advertised when the repo actually ships project skills (a plain checkout gets no
dead option), the notice is a question rather than a status line, and the answer goes to the
gateway — the process that holds the config file and the agent whose cached skills section has to
be dropped.
"""
from pathlib import Path

import pytest

from acp_adapter.gateway_session_meta import PROJECT_SKILLS_CONFIG_ID, project_skills_option


def _repo(tmp_path, skills=("deploy", "triage")):
    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    for name in skills:
        d = root / ".agents" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n\nbody\n")
    return root


@pytest.fixture
def trusted_dirs(monkeypatch):
    """The profile's answer, without touching a real config.yaml."""
    state = {"trusted": set(), "ignored": set()}
    monkeypatch.setattr("agent.skill_utils._project_trusted_dirs_from_config",
                        lambda: {Path(p) for p in state["trusted"]})
    monkeypatch.setattr("agent.skill_utils.is_project_root_ignored",
                        lambda root: str(root) in state["ignored"])
    return state


def test_an_undecided_repo_is_offered_with_its_count(tmp_path, trusted_dirs):
    root = _repo(tmp_path)
    option = project_skills_option(str(root))
    assert option.id == PROJECT_SKILLS_CONFIG_ID
    assert option.current_value == "ask"
    assert [o.value for o in option.options] == ["ask", "trusted", "ignore"]
    assert option.field_meta["skillCount"] == 2
    assert "NOT loaded" in option.description


def test_a_trusted_repo_shows_as_trusted_and_not_as_a_question(tmp_path, trusted_dirs):
    root = _repo(tmp_path)
    trusted_dirs["trusted"].add(str(root))
    option = project_skills_option(str(root))
    assert option.current_value == "trusted"
    assert "are loaded" in option.description


def test_a_repo_the_operator_waved_off_shows_ignore(tmp_path, trusted_dirs):
    root = _repo(tmp_path)
    trusted_dirs["ignored"].add(str(root))
    assert project_skills_option(str(root)).current_value == "ignore"


def test_a_repo_without_project_skills_advertises_nothing(tmp_path, trusted_dirs):
    root = _repo(tmp_path, skills=())
    assert project_skills_option(str(root)) is None
    assert project_skills_option(None) is None
    assert project_skills_option(str(tmp_path / "not-a-repo")) is None


def test_the_notice_is_a_question_a_host_can_answer(tmp_path):
    from acp_adapter.events import make_project_skills_update

    root = _repo(tmp_path)
    hermes = make_project_skills_update(str(root), 2).model_dump(by_alias=True)["_meta"]["hermes"]
    assert hermes["kind"] == "project_skills"
    assert hermes["state"] == "untrusted"
    assert hermes["skillCount"] == 2
    assert hermes["root"] == str(root)
    assert "2 个项目 skill" in hermes["text"]
