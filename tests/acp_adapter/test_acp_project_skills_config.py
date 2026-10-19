"""ACP must let a client SEE and SET the trust decision for project-local skills.

Hermes only loaded `<repo>/.agents/skills` (and `.hermes/skills`) when the repo root was listed in
``skills.trusted_project_dirs``, and the only surface that ever said so was the interactive CLI
banner ("N project skills not enabled — run `hermes skills trust`"). Every non-interactive entry
point — gateway, desktop, and the ACP adapter that every cockpit speaks to — skipped them in
silence: a workspace full of playbooks looked like it was loaded and was not.

Invariants pinned here:

1. A session whose repo ships project skills advertises a ``project_skills`` select, with the
   state the config actually holds (ask / trusted / ignore) and the root it applies to.
2. A repo with none advertises nothing — the option must never be a dead row.
3. ``set_config_option(project_skills)`` writes the same config key `hermes skills trust` writes,
   through the same helper, and the rebuilt option list reports the new state.
4. The "not now" answer is remembered per root and also clears trust, so the two lists cannot
   contradict each other.
5. The session-start notice rides ``_meta.hermes`` on an empty thought chunk (never a fabricated
   ``sessionUpdate`` type), and is absent once the repo is trusted or dismissed.
"""

import asyncio
import json

import acp
import pytest

import agent.skill_utils as su
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager

from tests.acp_adapter.test_acp_commands import FakeAgent


class RecordingDb:
    def __init__(self):
        self.rows = {}

    def get_session(self, session_id, *_a, **_k):
        return self.rows.get(session_id)

    def create_session(self, session_id, source, model=None, model_config=None, **_k):
        self.rows[session_id] = {"id": session_id, "source": source, "model": model,
                                 "model_config": json.dumps(model_config) if model_config else None}
        return session_id

    def update_session_meta(self, session_id, model_config_json, model=None):
        row = self.rows.get(session_id)
        if row is not None:
            row["model_config"] = model_config_json

    def replace_messages(self, *_a, **_k):
        return None

    def get_messages_as_conversation(self, *_a, **_k):
        return []


class StubConn:
    """Collects the session_update frames the adapter sends."""

    def __init__(self):
        self.updates = []

    async def session_update(self, *, session_id, update, **_kw):
        self.updates.append((session_id, update))


def _write_config(home, body: str) -> None:
    (home / "config.yaml").write_text(body, encoding="utf-8")
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()


@pytest.fixture
def repo_env(tmp_path, monkeypatch):
    """Temp HERMES_HOME + a git repo carrying two project skills."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    _write_config(home, "skills:\n  external_dirs: []\n")

    repo = tmp_path / "ws"
    (repo / ".git").mkdir(parents=True)
    for name in ("alpha-pb", "beta-pb"):
        d = repo / ".agents" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name}\n---\nbody\n", encoding="utf-8")

    bare = tmp_path / "bare"
    (bare / ".git").mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(home))
    yield {"home": home, "repo": repo, "bare": bare}
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()


def _session(repo):
    manager = SessionManager(agent_factory=lambda **_k: FakeAgent(), db=RecordingDb())
    state = manager.create_session(cwd=str(repo))
    state.history.append({"role": "user", "content": "hi"})
    return manager, HermesACPAgent(session_manager=manager), state


def _option(acp_agent, state):
    opts = acp_agent._build_config_options(state)
    return next((o for o in opts if o.id == "project_skills"), None)


def test_untrusted_repo_advertises_the_decision(repo_env):
    _manager, acp_agent, state = _session(repo_env["repo"])
    opt = _option(acp_agent, state)
    assert opt is not None, "a repo with project skills must advertise the trust row"
    assert opt.type == "select"
    assert opt.current_value == "ask"
    assert [o.value for o in opt.options] == ["ask", "trusted", "ignore"]
    assert opt.field_meta["root"] == str(repo_env["repo"].resolve())
    assert opt.field_meta["skillCount"] == 2
    assert "2 skill(s)" in opt.description


def test_repo_without_project_skills_has_no_row(repo_env):
    _manager, acp_agent, state = _session(repo_env["bare"])
    assert _option(acp_agent, state) is None


def test_trust_writes_the_same_key_the_cli_writes(repo_env):
    _manager, acp_agent, state = _session(repo_env["repo"])
    resp = asyncio.run(acp_agent.set_config_option(
        config_id="project_skills", session_id=state.session_id, value="trusted"))

    cfg = su._skills_cfg()
    assert str(repo_env["repo"].resolve()) in cfg.get("trusted_project_dirs", [])
    assert su.is_project_root_trusted(repo_env["repo"])
    rebuilt = next(o for o in resp.config_options if o.id == "project_skills")
    assert rebuilt.current_value == "trusted"
    assert "are loaded" in rebuilt.description


def test_ignore_is_remembered_per_root_and_clears_trust(repo_env):
    _manager, acp_agent, state = _session(repo_env["repo"])
    asyncio.run(acp_agent.set_config_option(
        config_id="project_skills", session_id=state.session_id, value="trusted"))
    asyncio.run(acp_agent.set_config_option(
        config_id="project_skills", session_id=state.session_id, value="ignore"))

    cfg = su._skills_cfg()
    assert str(repo_env["repo"].resolve()) in cfg.get("project_skills_ignored", [])
    assert su.is_project_root_trusted(repo_env["repo"]) is False, \
        "ignore left trust in place — the prompt and the loader would disagree"
    assert _option(acp_agent, state).current_value == "ignore"

    # `ask` is the way back: it must clear BOTH answers.
    asyncio.run(acp_agent.set_config_option(
        config_id="project_skills", session_id=state.session_id, value="ask"))
    assert _option(acp_agent, state).current_value == "ask"


def test_unknown_value_is_rejected(repo_env):
    _manager, acp_agent, state = _session(repo_env["repo"])
    with pytest.raises(acp.RequestError):
        asyncio.run(acp_agent.set_config_option(
            config_id="project_skills", session_id=state.session_id, value="yes-please"))
    assert su.is_project_root_trusted(repo_env["repo"]) is False


def test_notice_rides_meta_hermes_and_disappears_once_decided(repo_env):
    _manager, acp_agent, state = _session(repo_env["repo"])
    acp_agent._conn = StubConn()

    asyncio.run(acp_agent._notify_project_skills(state))
    assert len(acp_agent._conn.updates) == 1
    _sid, update = acp_agent._conn.updates[0]
    payload = update.model_dump(by_alias=True, exclude_none=True)
    assert payload["sessionUpdate"] == "agent_thought_chunk", \
        "the notice must not invent a sessionUpdate type"
    assert payload["content"]["text"] == "", "an empty thought is the carrier"
    meta = payload["_meta"]["hermes"]
    assert meta["kind"] == "project_skills" and meta["state"] == "untrusted"
    assert meta["skillCount"] == 2
    assert "2 个项目 skill 未启用" in meta["text"]

    asyncio.run(acp_agent.set_config_option(
        config_id="project_skills", session_id=state.session_id, value="trusted"))
    acp_agent._conn.updates.clear()
    asyncio.run(acp_agent._notify_project_skills(state))
    assert acp_agent._conn.updates == [], "trusted repo must stop nagging"

    asyncio.run(acp_agent.set_config_option(
        config_id="project_skills", session_id=state.session_id, value="ignore"))
    asyncio.run(acp_agent._notify_project_skills(state))
    assert acp_agent._conn.updates == [], "'never ask' must stay quiet on later sessions"
