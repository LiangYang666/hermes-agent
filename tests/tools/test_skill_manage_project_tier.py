"""`skill_manage` must be able to improve a TRUSTED repo's own skills, and nothing else.

Why this exists: a workspace can keep its playbooks as project-local skills
(`<repo>/.agents/skills/<name>/SKILL.md`), which is versioned, reviewable and auto-loaded — but
Hermes' self-evolution path only ever searched the profile tier (`get_all_skills_dirs()` excludes
project roots), so the agent could read those skills and then be unable to fix them. The fix is
tier-aware lookup, NOT a wider root set: an untrusted repo stays unreadable AND unwritable, so the
trust decision continues to be the single gate for both directions.

Pinned here:
1. A skill in a trusted project root is found, and found BEFORE a same-named profile copy (the
   loader reads the project tier first; patching the shadowed copy would change nothing).
2. The same skill in an untrusted repo is not found — the write path refuses to guess.
3. A patch lands in the repo's file, not in the profile's skills dir.
"""

import json
from pathlib import Path

import pytest

import agent.skill_utils as su
from tools import skill_manager_tool as smt


def _make_skill(root: Path, name: str, body: str = "body\n") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    md = d / "SKILL.md"
    md.write_text(f"---\nname: {name}\ndescription: pb\n---\n\n{body}", encoding="utf-8")
    return md


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    (home / "config.yaml").write_text("skills:\n  external_dirs: []\n", encoding="utf-8")

    repo = tmp_path / "ws"
    (repo / ".git").mkdir(parents=True)
    repo_md = _make_skill(repo / ".agents" / "skills", "shared-pb", "project copy\n")
    home_md = _make_skill(home / "skills", "shared-pb", "profile copy\n")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(repo)
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()
    yield {"home": home, "repo": repo, "repo_md": repo_md, "home_md": home_md}
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()


def _trust(repo: Path) -> None:
    assert su.set_project_root_trust(repo, True) is True
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()


def test_untrusted_repo_skill_is_not_a_write_target(env):
    assert su.is_project_root_trusted(env["repo"]) is False
    found = smt._find_skill("shared-pb")
    assert found is not None
    assert found["path"] == env["home_md"].parent, \
        "an untrusted repo must resolve to the profile copy, never the checkout"


def test_trusted_repo_skill_wins_over_the_profile_copy(env):
    _trust(env["repo"])
    found = smt._find_skill("shared-pb")
    assert found is not None
    assert found["path"] == env["repo_md"].parent, \
        "the project tier shadows the profile copy — the write must land where the loader reads"


def test_patch_writes_into_the_repo(env):
    _trust(env["repo"])
    out = json.loads(smt.skill_manage(
        action="patch", name="shared-pb",
        old_string="project copy", new_string="project copy v2"))
    assert out.get("success") is True, out
    assert "v2" in env["repo_md"].read_text(encoding="utf-8")
    assert "v2" not in env["home_md"].read_text(encoding="utf-8"), \
        "the profile copy must stay untouched"
