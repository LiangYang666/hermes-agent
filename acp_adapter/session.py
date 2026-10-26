"""ACP session manager — maps ACP sessions to Hermes AIAgent instances.

Sessions are persisted to the shared SessionDB (``~/.hermes/state.db``) so they
survive process restarts and appear in ``session_search``; ``load_session`` /
``resume_session`` after an editor reconnect restore the full history from there.
"""
from __future__ import annotations

from hermes_constants import get_hermes_home, translate_cwd_for_wsl_backend, windows_path_to_wsl

import copy
import contextlib
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone, UTC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bound on the compaction-continuation walk in ``SessionManager._restore``. Each notification compaction
# adds one hop; the bound only exists so a corrupt/looping lineage cannot spin a restore.
_MAX_COMPACTION_CONTINUATIONS = 32

# A session whose provider cannot be resolved is retried on every prompt of that session, so the
# fallback warning is emitted once per session id (bounded: oldest entries are dropped wholesale).
_WARNED_UNROUTABLE: dict[str, None] = {}
_WARNED_UNROUTABLE_MAX = 512


def _heal_custom_identity(provider: str | None, *, base_url: str | None, model: str | None) -> str | None:
    """Return a routable provider identity for a persisted one, or ``None`` when nothing better exists.

    A named custom provider reaches the agent as its billing class ``"custom"`` — a label, not a menu
    key. Persisting it creates two distinct failures, and they behave differently:

    * bare ``custom`` still *resolves* — to some compatible endpoint (OpenRouter on a box that has an
      OpenRouter key), so the session silently bills a vendor it was never configured for and every
      turn dies as that vendor's ``model_not_found`` (#63);
    * a key whose config entry has been renamed or deleted (``custom:gone``) *raises* instead, so the
      caller falls back to the config default and the turn is answered by yet another vendor.

    Both are handled here by recovering the durable ``custom:<name>`` key from the endpoint
    (preferred) or from the model — the same recovery the CLI and the TUI gateway already perform.
    The ACP paths (restore and fork) were the ones that skipped it. Anything already routable is
    returned unchanged, so a deliberate pivot — the user moved ``model.provider`` elsewhere while
    this session keeps ``custom:lab`` — is never "healed" into the new default.
    """
    raw = str(provider or "").strip()
    if not raw:
        return None
    try:
        from hermes_cli.runtime_provider import canonical_custom_identity
        from hermes_cli.runtime_provider_custom import is_routable_provider

        if is_routable_provider(raw):
            return raw
        return canonical_custom_identity(base_url=base_url or None, model=model or None) or None
    except Exception:
        logger.debug("Could not recover a provider identity for %r (%r)", provider, base_url, exc_info=True)
        return None


def _translate_acp_cwd(cwd: str) -> str:
    """Translate Windows ACP cwd values (``E:\\Projects``, ``\\\\wsl.localhost\\``) to POSIX form
    when Hermes runs in WSL so agents, tools, and persisted sessions agree; no-op elsewhere."""
    return translate_cwd_for_wsl_backend(str(cwd))


def _normalize_cwd_for_compare(cwd: str | None) -> str:
    expanded = os.path.expanduser(str(cwd or ".").strip() or ".")

    # Windows drive paths -> WSL mount form so history filters match across hosts.
    translated = windows_path_to_wsl(expanded)
    if translated is not None:
        expanded = translated
    elif re.match(r"^/mnt/[A-Za-z]/", expanded):
        expanded = f"/mnt/{expanded[5].lower()}/{expanded[7:]}"

    # realpath resolves symlink aliases (macOS ``/var`` vs ``/private/var``, ``/tmp`` vs
    # ``/private/tmp``) that otherwise drop a workspace's own sessions; it is lexical
    # for missing paths (e.g. WSL-translated drives).
    try:
        # Resolve symlink aliases so equivalent spellings of the same directory compare equal — macOS
        # reports editor workspaces as ``/var/...`` while sessions get stored under ``/private/var/...``
        # (and ``/tmp`` vs ``/private/tmp``), which made ACP history filters silently drop a workspace's own
        # sessions. WSL-translated Windows drives — keep the previous normpath behavior. Ported from
        # PrimeIntellect-ai/prime-agent#628.
        return os.path.realpath(expanded)
    except OSError:
        return os.path.normpath(expanded)


def _build_session_title(title: Any, preview: Any, cwd: str | None) -> str:
    leaf = os.path.basename(str(cwd or "").rstrip("/\\"))
    return str(title or "").strip() or str(preview or "").strip() or leaf or "New thread"


def _format_updated_at(value: Any) -> str | None:
    if value is None or (isinstance(value, str) and value.strip()):
        return value
    try:
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    except Exception:
        return None


def _updated_at_sort_key(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return float("-inf")
    for parse in (lambda s: datetime.fromisoformat(s).timestamp(), float):
        try:
            return parse(raw)
        except Exception:
            continue
    return float("-inf")


def _acp_stderr_print(*args, **kwargs) -> None:
    """Route incidental AIAgent output to stderr; ACP reserves stdout for JSON-RPC."""
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _register_task_cwd(task_id: str, cwd: str) -> None:
    """Bind a task/session id to the editor cwd for tools. Zed may send a Windows cwd while
    the ACP process runs in WSL; tools need the WSL mount or subprocess creation fails."""
    if not task_id:
        return
    try:
        from tools.terminal_tool import register_task_env_overrides
        register_task_env_overrides(task_id, {"cwd": _translate_acp_cwd(cwd)})
    except Exception:
        logger.debug("Failed to register ACP task cwd override", exc_info=True)


def _expand_acp_enabled_toolsets(toolsets: list[str] | None = None,
                                 mcp_server_names: list[str] | None = None) -> list[str]:
    """Return ACP toolsets plus explicit MCP server toolsets for this session."""
    names = [n for n in (["hermes-acp"] if toolsets is None else toolsets) if n]
    names += [f"mcp-{s}" for s in (mcp_server_names or []) if s]
    return list(dict.fromkeys(names))


def _parse_model_config(mc: Any) -> dict:
    """Decode a persisted model_config JSON blob; ``{}`` when absent/invalid/non-dict."""
    try:
        meta = json.loads(mc) if mc else None
    except (json.JSONDecodeError, TypeError):
        meta = None
    return meta if isinstance(meta, dict) else {}


def _session_info(sid: str, cwd: str, model: Any, history_len: int, title: Any, preview: Any,
                  updated_at: Any) -> dict[str, Any]:
    return {"session_id": sid, "cwd": cwd, "model": model, "history_len": history_len,
            "title": _build_session_title(title, preview, cwd), "updated_at": _format_updated_at(updated_at)}


def _first_user_preview(history: list[dict[str, Any]], default: str) -> str:
    return next((str(m.get("content") or "").strip() for m in history
                 if m.get("role") == "user" and str(m.get("content") or "").strip()), default)


@dataclass
class SessionState:
    """Tracks per-session state for an ACP-managed Hermes agent."""

    session_id: str
    agent: Any  # AIAgent instance
    cwd: str = "."
    model: str = ""
    # Reasoning effort chosen via ACP session/config_options (id=reasoning_effort).
    # Persisted in model_config so a process restart / session/load does not silently
    # revert thinking depth to the config default while the cockpit still shows the
    # picked level. (``mode`` below rides the same blob for the same reason.)
    reasoning_config: Optional[Dict[str, Any]] = None
    # ACP permission mode (default / accept_edits / dont_ask). Persisted in the
    # session's model_config JSON so a process restart or session/load keeps the
    # user's approval setting instead of silently falling back to the default.
    mode: str = ""
    # Context budget picked over ACP (config option ``context_budget``): the window Hermes budgets
    # its compression against, in tokens. None follows the model/config default. Persisted like
    # ``reasoning_config`` so a restart or session/load keeps the ceiling the operator chose —
    # without it a cockpit keeps showing the pick while the session silently compresses against
    # the model's full window again.
    context_budget: Optional[int] = None
    history: list[dict[str, Any]] = field(default_factory=list)
    cancel_event: Any = None  # threading.Event
    is_running: bool = False
    # A state-mutating slash command (/reset, /compress, /model) is in flight. Turn claims
    # must queue behind it: /compress's LLM call and /model's agent rebuild take seconds, so
    # a bare is_running check in the slash thread would leave a check-then-act window where
    # a prompt claims the turn mid-mutation.
    command_op: bool = False
    queued_prompts: list[str] = field(default_factory=list)
    runtime_lock: Any = field(default_factory=threading.Lock)
    current_prompt_text: str = ""
    interrupted_prompt_text: str = ""
    # Per-session allocator for ACP assistant messageIds (lazily created by
    # the server so streamed chunks group into distinct assistant replies).
    message_ids: Any = None


def apply_context_budget(agent, tokens: Optional[int]) -> bool:
    """Pin (or clear) the context window Hermes budgets and compresses against.

    ``context_budget`` is a real knob, not a display number: the compressor derives its trigger as a
    percentage of this window, so picking 64K on a 200K model makes the session compress early
    instead of at the model's own ceiling. Reuses the same pair the TUI/gateway config path uses
    (#116467): ``set_config_context_length`` refreshes BOTH cached copies of the pin (agent +
    compressor), then the compressor re-derives percent and threshold from the new window, so it can
    never report one ceiling while compressing against another. ``None`` drops the override and lets
    construction's deferred resolution infer the model's own window again (re-applying the
    small-context floor). Returns False when the agent has no compressor to steer — a test double, in
    which case the caller has still recorded the pick.
    """
    from agent.agent_init import set_config_context_length

    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return False
    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0:
        set_config_context_length(agent, tokens)
        compressor.context_length = tokens
    else:
        set_config_context_length(agent, None)
        # Force re-inference from model metadata on next access (mirrors the config-save path).
        compressor._resolved_context_length = None
    # Invalidate the cached trigger so the next preflight re-derives from the new window.
    compressor._threshold_tokens = compressor._tail_token_budget = None
    return True


def model_window_pin(provider: str, model: str) -> Optional[int]:
    """The window THIS (provider, model) is pinned to in config.yaml, or None when unpinned.

    Reads the runtime's own explicit-override step (``model_overrides.<provider>.<model>.
    context_window``, model_metadata resolution 0b) rather than the catalog: a models.dev number is
    not a setting, and advertising it would make the picker's ``currentValue`` claim a window nobody
    chose. Deliberately not ``lookup_models_dev_context``, which falls back to catalog/``_default``.
    """
    provider, model = str(provider or "").strip(), str(model or "").strip()
    if not provider or not model:
        return None
    with contextlib.suppress(Exception):
        from agent.models_dev import _override_context_window

        return _override_context_window(provider, model)
    return None


def write_model_window_pin(provider: str, model: str, tokens: Optional[int]) -> bool:
    """Persist (or clear) the window for one (provider, model) in config.yaml; True when written.

    Global on purpose: this is the pin every Hermes runtime reads at resolution 0b — the TUI, the
    gateway, another ACP slot, a plain CLI run — so the window belongs to the MODEL, not to the
    session that happened to set it. ``tokens=None`` removes the key, and any provider/model
    scaffolding left empty with it, so ``auto`` is indistinguishable from never having chosen one.

    Print-free under the config lock: stdout carries the ACP JSON-RPC stream (see ``entry.py``), so
    ``set_config_value`` — which prints its result — is unusable from inside a session.
    """
    provider, model = str(provider or "").strip(), str(model or "").strip()
    if not provider or not model:
        return False
    try:
        from hermes_cli import config as config_mod

        if config_mod.is_managed():
            # Managed installs own config.yaml; the in-memory pin still applies.
            logger.info("model_overrides is administrator-managed; the window stays session-scoped")
            return False
        with config_mod._CONFIG_LOCK:  # noqa: SLF001 — the lock every config writer holds
            config_path = config_mod.get_config_path()
            config = config_mod.require_readable_config_before_write(config_path)  # fail-closed read
            overrides = config.get("model_overrides")
            overrides = dict(overrides) if isinstance(overrides, dict) else {}
            section = overrides.get(provider)
            section = dict(section) if isinstance(section, dict) else {}
            entry = section.get(model)
            entry = dict(entry) if isinstance(entry, dict) else {}
            if tokens is not None:
                entry["context_window"] = int(tokens)
            else:
                entry.pop("context_window", None)
            # Never leave empty scaffolding: `auto` must read back as "nothing was ever set".
            if entry:
                section[model] = entry
            else:
                section.pop(model, None)
            if section:
                overrides[provider] = section
            else:
                overrides.pop(provider, None)
            if overrides:
                config["model_overrides"] = overrides
            else:
                config.pop("model_overrides", None)
            config_mod._write_user_config(config_path, config)  # noqa: SLF001 — atomic writer
        return True
    except Exception:
        logger.warning("Could not write the context window for %s/%s to config.yaml; the pick stays "
                       "session-scoped", provider, model, exc_info=True)
        return False


class SessionManager:
    """Thread-safe manager for ACP sessions backed by Hermes AIAgent instances.

    Sessions are held in-memory for fast access **and** persisted to the shared
    SessionDB so they survive restarts and are searchable via ``session_search``."""

    def __init__(self, agent_factory=None, db=None):
        """``agent_factory``: AIAgent-like factory (tests); default builds a real AIAgent from
        the runtime provider config. ``db``: SessionDB; default lazily opens ``~/.hermes/state.db``."""
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        # Serializes DB restores: session construction runs off the event loop, so two
        # overlapping session/load for one id must share a single agent build.
        self._restore_lock = threading.Lock()
        self._agent_factory = agent_factory
        self._db_instance = db  # None → lazy-init on first use
        self._cwd_backfilled = False

    # ---- public API ---------------------------------------------------------

    def create_session(self, cwd: str = ".") -> SessionState:
        """Create a new session with a unique ID and a fresh AIAgent."""
        cwd = _translate_acp_cwd(cwd)
        session_id = str(uuid.uuid4())
        agent = self._make_agent(session_id=session_id, cwd=cwd)
        state = self._install_state(session_id, agent, cwd, getattr(agent, "model", "") or "", [])
        logger.info("Created ACP session %s (cwd=%s)", session_id, cwd)
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Return the session, transparently restoring it from the DB (e.g. after
        a process restart) when it is not in memory; ``None`` if unknown."""
        with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            return state
        with self._restore_lock:
            with self._lock:
                state = self._sessions.get(session_id)  # a concurrent restore may have installed it
            return state if state is not None else self._restore(session_id)

    def fork_session(self, session_id: str, cwd: str = ".") -> Optional[SessionState]:
        """Deep-copy a session's history into a new session."""
        cwd = _translate_acp_cwd(cwd)
        original = self.get_session(session_id)  # checks DB too
        if original is None:
            return None
        new_id = str(uuid.uuid4())
        # A fork is a continuation, not a fresh session: carry the parent's session-scoped
        # reasoning effort into the child's agent AND its state (same idea as the parallel
        # mode-inheritance feature). ``None`` parent picks -> child follows the config default.
        parent_reasoning = getattr(original, "reasoning_config", None)
        if not isinstance(parent_reasoning, dict):
            live = getattr(original.agent, "reasoning_config", None)
            parent_reasoning = live if isinstance(live, dict) else None
        agent = self._make_agent(
            session_id=new_id, cwd=cwd, model=original.model or None,
            # A fork continues the parent's *route*, not just its transcript: carrying only the model
            # hands a custom-entry session to the config default, which is #63's symptom one entry
            # further along (#63 review A1). ``original.agent`` may itself have just been restored,
            # in which case its ``provider`` is the bare class and the heal resolves it again here.
            requested_provider=getattr(original.agent, "provider", None),
            base_url=getattr(original.agent, "base_url", None),
            api_mode=getattr(original.agent, "api_mode", None),
            reasoning_config=parent_reasoning)
        model = getattr(agent, "model", original.model) or original.model
        state = self._install_state(new_id, agent, cwd, model, copy.deepcopy(original.history),
                                    mode=getattr(original, "mode", "") or "")
        if isinstance(parent_reasoning, dict):
            state.reasoning_config = parent_reasoning
        logger.info("Forked ACP session %s -> %s", session_id, new_id)
        return state

    def live_sessions_on_route(
        self, provider: str, model: str, *, exclude: str = ""
    ) -> List[SessionState]:
        """Running sessions whose route is (provider, model) — the ones a model-scoped change must reach.

        A window pinned for a model applies to every session running that model, so the session that
        set it is not the only one to update: its siblings on the same route get the new ceiling
        immediately instead of quietly compressing against the old one until they are restarted.
        Cold rows are deliberately excluded — they re-resolve config when restored.
        """
        provider = str(provider or "").strip()
        model = str(model or "").strip()
        if not model:
            return []
        with self._lock:
            states = list(self._sessions.values())
        out: List[SessionState] = []
        for state in states:
            if state.session_id == exclude:
                continue
            route_provider = str(getattr(state.agent, "provider", "") or "").strip()
            route_model = str(getattr(state.agent, "model", "") or state.model or "").strip()
            if route_model != model:
                continue
            if provider and route_provider and route_provider != provider:
                continue  # same slug, different provider: the other route's pin does not apply
            out.append(state)
        return out

    def list_sessions(self, cwd: str | None = None) -> list[dict[str, Any]]:
        """Return lightweight info dicts for all sessions (memory + database)."""
        normalized_cwd = _normalize_cwd_for_compare(cwd) if cwd else None
        db = self._get_db()
        persisted_rows: dict[str, dict[str, Any]] = {}
        try:
            for row in (db.list_sessions_rich(source="acp", limit=1000) if db is not None else ()):
                persisted_rows[str(row["id"])] = dict(row)
        except Exception:
            logger.debug("Failed to load ACP sessions from DB", exc_info=True)

        def _matches(session_cwd: str) -> bool:
            return not normalized_cwd or _normalize_cwd_for_compare(session_cwd) == normalized_cwd

        # In-memory sessions first.
        with self._lock:
            seen_ids = set(self._sessions.keys())
            results = []
            for s in self._sessions.values():
                if not s.history or not _matches(s.cwd):
                    continue
                persisted = persisted_rows.get(s.session_id, {})
                results.append(_session_info(
                    s.session_id, s.cwd, s.model, len(s.history), persisted.get("title"),
                    _first_user_preview(s.history, persisted.get("preview") or ""),
                    persisted.get("last_active") or persisted.get("started_at") or time.time(),
                ))

        # Then persisted sessions not currently in memory.
        for sid, row in persisted_rows.items():
            message_count = int(row.get("message_count") or 0)
            session_cwd = _parse_model_config(row.get("model_config")).get("cwd", ".")
            if sid in seen_ids or message_count <= 0 or not _matches(session_cwd):
                continue
            results.append(_session_info(
                sid, session_cwd, row.get("model") or "", message_count, row.get("title"),
                row.get("preview"), row.get("last_active") or row.get("started_at"),
            ))

        results.sort(key=lambda item: _updated_at_sort_key(item.get("updated_at")), reverse=True)
        return results

    def update_cwd(self, session_id: str, cwd: str) -> Optional[SessionState]:
        """Update the working directory for a session and its tool overrides."""
        cwd = _translate_acp_cwd(cwd)
        state = self.get_session(session_id)  # checks DB too
        if state is None:
            return None
        state.cwd = cwd
        _register_task_cwd(session_id, cwd)
        self._persist(state)
        # Promote the authoritative column and claim an ordering generation, the
        # same contract tui_gateway/session_workdir.py uses: a git probe may only
        # publish while its generation is still current, so a slow probe for a
        # previous workspace cannot overwrite a newer claim (A -> B -> A).
        self._schedule_git_metadata(state, self._claim_cwd_generation(state))
        return state

    def save_session(self, session_id: str) -> None:
        """Persist a session; called by the server after prompt completion,
        history-mutating slash commands, and model switches."""
        with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            self._persist(state)

    def end_all_sessions(self, end_reason: str = "acp_disconnect") -> int:
        """Stamp ``ended_at`` on every live session (#118216).

        ACP v0.9 has no per-session destroy, so the stdio shutdown that ends
        this process is the session end: the client that drove the
        conversation is gone. Without this writer, source='acp' rows keep
        ``ended_at`` NULL forever and the ended-session guard shared by
        prune/archive (``hermes_state_maintenance``) can never reach them.
        A later load/resume reopens the row (see ``_restore``), the same
        contract the TUI gateway's resume path uses. Best-effort: teardown
        must never raise. Returns the number of sessions ended.
        """
        db = self._get_db()
        if db is None:
            return 0
        with self._lock:
            session_ids = list(self._sessions.keys())
        ended = 0
        for session_id in session_ids:
            try:
                db.end_session(session_id, end_reason)
                ended += 1
            except Exception:
                logger.debug("Failed to end ACP session %s", session_id, exc_info=True)
        if ended:
            logger.info("Ended %d ACP session(s) on shutdown (%s)", ended, end_reason)
        return ended

    # ---- persistence via SessionDB ------------------------------------------

    def _install_state(self, session_id: str, agent: Any, cwd: str, model: str,
                       history: list[dict[str, Any]], *, persist: bool = True,
                       mode: str = "") -> SessionState:
        """Build a SessionState, register it in memory, bind its cwd for tools, optionally persist."""
        state = SessionState(session_id=session_id, agent=agent, cwd=cwd, model=model,
                             mode=mode, history=history, cancel_event=threading.Event())
        with self._lock:
            self._sessions[session_id] = state
        _register_task_cwd(session_id, cwd)
        if persist:
            self._persist(state)
        return state

    def _get_db(self):
        """Lazily acquire the process-shared SessionDB; ``None`` if unavailable (e.g. import
        error in a minimal test env). ``HERMES_HOME`` is resolved here, not via the import-time
        ``DEFAULT_DB_PATH``, so test fixtures that change the env var later are honoured. The
        registry handle is the one in-process tools (delegation, session_search, goals) also
        acquire, so the ACP server holds ONE writer on state.db instead of two (#100896)."""
        if self._db_instance is None:
            try:
                from hermes_state_registry import acquire
                self._db_instance = acquire(get_hermes_home() / "state.db")
            except Exception:
                logger.debug("SessionDB unavailable for ACP persistence", exc_info=True)
        if self._db_instance is not None and not self._cwd_backfilled:
            # Rows minted before the adapter wrote the cwd column still carry the workspace
            # in model_config; one idempotent UPDATE per process repairs them (#115705).
            self._cwd_backfilled = True
            try:
                repaired = self._db_instance.backfill_acp_session_cwd()
                if repaired:
                    logger.info("Backfilled cwd for %d ACP session(s) from model_config", repaired)
            except Exception:
                logger.debug("ACP session cwd backfill failed", exc_info=True)
        return self._db_instance

    def _persist(self, state: SessionState) -> None:
        """Create/update the session record, then sync the live message set."""
        db = self._get_db()
        if db is None:
            return

        # Ensure model is a plain string (not a MagicMock or other proxy).
        model_str = str(state.model) if state.model else None
        session_meta: dict[str, Any] = {"cwd": state.cwd}
        for key in ("provider", "base_url", "api_mode"):
            value = getattr(state.agent, key, None)
            if isinstance(value, str) and value.strip():
                session_meta[key] = value.strip()
        # A named custom provider reaches the agent as its billing class ``custom``; writing that into
        # the blob leaves the row with no routable key at all, so a restore can only guess from the
        # endpoint — and when the guess fails it silently bills the config default (#63). Store the
        # durable ``custom:<name>`` menu key instead.
        #
        # The value written is the *intent* (what this row asked for), never a fallback the resolver
        # picked after failing: a session whose entry is temporarily unreachable must stay "custom"
        # so a later restore can heal it, instead of being rewritten into the config default forever.
        intent = getattr(state.agent, "_acp_requested_provider", None)
        if not isinstance(intent, str) or not intent.strip():
            # Mock agents (tests) and agents built by a factory carry no intent; anything else here
            # would be written into the JSON blob verbatim.
            intent = None
        healed_provider = _heal_custom_identity(
            intent or session_meta.get("provider"), base_url=session_meta.get("base_url"), model=model_str)
        if healed_provider:
            session_meta["provider"] = healed_provider
        elif intent:
            session_meta["provider"] = intent
        # Session-scoped reasoning effort picked over ACP (config_options) must ride in
        # the metadata blob, or a restart silently reverts thinking depth to the config
        # default while the client UI keeps showing the user's selection.
        if isinstance(getattr(state, "reasoning_config", None), dict):
            session_meta["reasoning_config"] = state.reasoning_config
        # Persist the ACP permission mode so session/load after a restart keeps the
        # user's approval setting. set_session_mode/set_config_option write
        # state.mode then call save_session() — without this field the mode
        # silently reverts to the default on restore.
        if isinstance(getattr(state, "mode", None), str) and state.mode.strip():
            session_meta["mode"] = state.mode.strip()
        # Same reason for the context budget: a picked window that is not persisted reverts to the
        # model default on the next restore while the cockpit still shows the pick.
        if isinstance(getattr(state, "context_budget", None), int) and (state.context_budget or 0) > 0:
            session_meta["context_budget"] = state.context_budget

        try:
            if db.get_session(state.session_id) is None:
                if not state.history:
                    # Empty editor probes stay ephemeral; copied fork history persists.
                    return
                db.create_session(session_id=state.session_id, source="acp", model=model_str,
                                  model_config=session_meta, cwd=state.cwd or None)
            else:
                try:
                    db.update_session_meta(state.session_id, json.dumps(session_meta), model_str)
                except Exception:
                    logger.debug("Failed to update ACP session metadata", exc_info=True)
                # The create branch above is not the live path: an agent that owns
                # persistence to this same DB flushes the transcript incrementally,
                # so the row already exists by the time we get here.
                # update_session_meta touches only model_config/model, so without
                # this promotion the column stays NULL for the whole session and
                # Desktop files it as unassigned. Claiming a generation keeps the
                # A -> B -> A ordering contract shared with update_cwd().
                if state.cwd:
                    self._schedule_git_metadata(state, self._claim_cwd_generation(state))

            # An agent that owns persistence to this same DB already flushed the transcript
            # incrementally (append_message) and keeps pre-compaction turns as archived
            # active=0 rows; replace_messages() would DELETE those (and, after a compression
            # id rotation, clobber the ended parent transcript). Skip it in that case.
            # Calling replace_messages() here would then be a redundant double-write that DELETEs exactly
            # those archived rows (and, after a compression-driven id rotation where agent.session_id no
            # longer equals state.session_id, clobbers the ended parent transcript) — silent data loss for
            # any ACP conversation long enough to compress. Only fall back to the destructive atomic replace
            # when the agent is NOT persisting itself to this DB (e.g. a test agent factory, or a fresh
            # create/fork whose copied history the agent has not flushed yet). That path still rolls back on
            # a mid-rewrite failure so the previously persisted conversation survives (salvaged from
            # #13675).
            agent = state.agent
            if getattr(agent, "_session_db", None) is db and getattr(agent, "_session_db_created", False):
                return
            # A non-owning agent (model switch, /restore: fresh agent, _session_db_created=False)
            # may still sit on archived rows, so replace ONLY the active=1 set: on a fresh
            # create/fork every row is active (== full replace), and archived rows survive.
            # Unconditional because an existence probe would fail OPEN on DB error and can
            # race a concurrent archive_and_compact. Still rolls back on mid-rewrite failure.
            db.replace_messages(state.session_id, state.history, active_only=True)
        except Exception:
            logger.warning("Failed to persist ACP session %s", state.session_id, exc_info=True)

    def _claim_cwd_generation(self, state: SessionState) -> Optional[int]:
        """Write the cwd column and return its new git-metadata generation.

        Returns None when the row does not exist yet (a contentless session is
        deliberately ephemeral until it has history) or the DB is unavailable.
        """
        db = self._get_db()
        if db is None or not state.cwd:
            return None
        try:
            return db.update_session_cwd(state.session_id, state.cwd)
        except Exception:
            logger.debug("Failed to persist ACP session cwd column for %s",
                         state.session_id, exc_info=True)
            return None

    def _schedule_git_metadata(self, state: SessionState, generation: Optional[int]) -> None:
        """Probe git off the critical path and publish under ``generation``.

        ``session/new`` is on the editor's interactive path; ``git rev-parse``
        on a cold or networked filesystem is not something to put in front of
        the user. The generation guard in ``publish_session_git_metadata``
        means a slow probe for a previous workspace is dropped rather than
        applied to the new one.
        """
        if not generation or not state.cwd:
            return

        session_id, cwd = state.session_id, state.cwd

        def _run() -> None:
            try:
                from tui_gateway import git_probe
                branch, root = git_probe.branch(cwd), git_probe.common_repo_root(cwd)
                if not (branch or root):
                    return
                db = self._get_db()
                if db is not None:
                    db.publish_session_git_metadata(session_id, cwd, generation, branch, root)
            except Exception:
                logger.debug("Failed to publish ACP git metadata for %s", session_id, exc_info=True)

        threading.Thread(target=_run, name=f"acp-git-meta-{session_id[:8]}", daemon=True).start()

    def _restore(self, session_id: str) -> Optional[SessionState]:
        """Load an ACP session from the database into memory, recreating the AIAgent."""
        db = self._get_db()
        if db is None:
            return None
        try:
            row = db.get_session(session_id)
        except Exception:
            logger.debug("Failed to query DB for ACP session %s", session_id, exc_info=True)
            return None
        if row is None or row.get("source") != "acp":
            return None

        # Compaction that cannot commit in place ends this row with end_reason='compression' and publishes
        # the compacted transcript as a continuation row. Our client-facing id never changes, so restoring
        # THIS row hands the resumed agent the pre-compaction transcript: the tokens the compaction just
        # removed come straight back, the next request is over the trigger again, and every agent restart
        # buys another ~95s summary call for nothing (the repeat the operator reports as "压缩又来了").
        # Follow the edge first — the resumed agent then writes to the tip, which is also the row that must
        # be reopened (reopening the parent would clear end_reason and break the edge for the next restart).
        client_session_id = session_id
        for _ in range(_MAX_COMPACTION_CONTINUATIONS):
            try:
                continuation = db.get_compression_continuation(session_id)
            except Exception:
                logger.debug("Failed to resolve compaction continuation of %s", session_id, exc_info=True)
                break
            if not continuation or continuation == session_id:
                break
            try:
                next_row = db.get_session(continuation)
            except Exception:
                next_row = None
            if next_row is None:
                break
            session_id, row = continuation, next_row
        if session_id != client_session_id:
            logger.info(
                "ACP session %s was compacted; restoring from continuation %s", client_session_id, session_id)

        # A previous adapter process stamped the row ended at its stdio
        # shutdown (#118216); resuming the conversation reopens it, the same
        # contract the TUI gateway's cold-resume path uses.
        if row.get("ended_at") is not None:
            try:
                db.reopen_session(session_id)
            except Exception:
                logger.debug("Failed to reopen ACP session %s", session_id, exc_info=True)

        meta = _parse_model_config(row.get("model_config"))
        cwd, model = meta.get("cwd", "."), row.get("model") or None
        # Session-scoped reasoning effort picked over ACP rides in the metadata blob; legacy
        # rows simply have no key -> None -> _make_agent follows the config default.
        restored_reasoning = meta.get("reasoning_config")
        if not isinstance(restored_reasoning, dict):
            restored_reasoning = None
        restored_mode = str(meta.get("mode") or "")
        # Context budget picked over ACP (legacy rows simply have no key -> None -> model default).
        restored_budget = meta.get("context_budget")
        if not isinstance(restored_budget, int) or isinstance(restored_budget, bool) or restored_budget <= 0:
            restored_budget = None

        # repair_alternation: this list becomes the resumed agent's LIVE conversation; a durable
        # ``user;user`` violation in state.db would otherwise re-fire the pre-request repair every request.
        try:
            history = db.get_messages_as_conversation(session_id, repair_alternation=True)
        except Exception:
            logger.warning("Failed to load messages for ACP session %s", session_id, exc_info=True)
            history = []

        try:
            agent = self._make_agent(
                session_id=session_id, cwd=cwd, model=model, api_mode=meta.get("api_mode") or None,
                requested_provider=meta.get("provider") or row.get("billing_provider"),
                base_url=meta.get("base_url") or row.get("billing_base_url"),
                reasoning_config=restored_reasoning)
        except Exception:
            logger.warning("Failed to recreate agent for ACP session %s", session_id, exc_info=True)
            return None
        # Registered under the CLIENT id: every later session/update and prompt is keyed by it. The agent
        # itself carries the continuation id, so its writes land on the compacted tip.
        state = self._install_state(client_session_id, agent, cwd, model or getattr(agent, "model", "") or "",
                                    history, persist=False, mode=restored_mode)
        if restored_reasoning is not None:
            state.reasoning_config = restored_reasoning
        # The model's pin wins over the session's stored mirror: another session may have changed the
        # window for this model while this one was cold, and the config.yaml value is what the agent
        # was just constructed with (resolution 0b). Only a pinless model falls back to the mirror.
        restored_budget = model_window_pin(
            str(getattr(agent, "provider", "") or ""),
            str(getattr(agent, "model", "") or model or ""),
        ) or restored_budget
        if restored_budget is not None:
            state.context_budget = restored_budget
            apply_context_budget(agent, restored_budget)
        logger.info("Restored ACP session %s from DB (%d messages, tip=%s)",
                    client_session_id, len(history), session_id)
        return state

    # ---- internal -----------------------------------------------------------

    def _make_agent(self, *, session_id: str, cwd: str, model: str | None = None,
                    requested_provider: str | None = None, base_url: str | None = None, api_mode: str | None = None,
                    enabled_toolsets: list[str] | None = None, disabled_toolsets: list[str] | None = None,
                    reasoning_config: dict | None = None):
        """``enabled_toolsets``/``disabled_toolsets`` carry a live session's toolsets into a rebuild; ``None`` derives
        them from config (fresh session). ``reasoning_config`` carries a session-scoped effort picked over ACP
        (``session/set_config_option``) into a rebuild; ``None`` follows the config default."""
        if self._agent_factory is not None:
            return self._agent_factory()

        from run_agent import AIAgent
        from agent.skill_utils import parse_config_string_list
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.tools_config import _get_platform_tools, enabled_mcp_server_names
        from hermes_constants import resolve_reasoning_config

        config = load_config()
        model_cfg = config.get("model")
        default_model, config_provider = "", None
        if isinstance(model_cfg, dict):
            default_model, config_provider = str(model_cfg.get("default") or ""), model_cfg.get("provider")
        elif isinstance(model_cfg, str):
            default_model = model_cfg.strip()

        if enabled_toolsets is None:
            # The same per-platform resolver as the gateway/cron/api_server: platform_toolsets.acp wins, else
            # hermes-acp; its MCP half (every enabled server, a listed-name allowlist, or none for ``no_mcp``)
            # comes back as bare server names, which ACP keys as ``mcp-<server>`` like its session servers.
            resolved = _get_platform_tools(config, "acp")
            mcp_servers = resolved & enabled_mcp_server_names(config)
            enabled_toolsets = _expand_acp_enabled_toolsets(sorted(resolved - mcp_servers), sorted(mcp_servers))
        kwargs = {
            "platform": "acp", "quiet_mode": True, "session_id": session_id, "session_db": self._get_db(),
            "enabled_toolsets": list(enabled_toolsets),
            # agent.disabled_toolsets is subtracted at tool granularity by the agent, as on the CLI/gateway/cron.
            "disabled_toolsets": (list(disabled_toolsets) if disabled_toolsets is not None
                                  else parse_config_string_list((config.get("agent") or {}).get("disabled_toolsets")) or None),
            "model": model or default_model,
            "cwd": cwd,
            # Same chokepoint as the CLI/gateway/TUI/cron: without it ``agent.reasoning_effort: none`` never
            # reaches an ACP session and the transport applies its default effort (a 400 on non-reasoning
            # models). Resolved against the session's model so per-model overrides apply. A session-scoped
            # pick over ACP (``reasoning_config`` param, persisted in model_config) wins over the config
            # default — the same precedence the TUI gives ``create_reasoning_override``.
            "reasoning_config": (reasoning_config if isinstance(reasoning_config, dict)
                                 else resolve_reasoning_config(config, model or default_model)),
        }
        resolve_error: Exception | None = None
        # A persisted bare ``custom`` is a billing class, not a menu key; heal it to the durable
        # ``custom:<name>`` before resolution. Without this the class still resolves — to whatever
        # compatible endpoint is configured — so the turn is answered (and billed) by a vendor this
        # session was never configured for (#63).
        healed_identity = _heal_custom_identity(
            requested_provider, base_url=base_url, model=model or default_model)
        requested_identity = healed_identity or config_provider
        if requested_provider and not healed_identity:
            logger.warning(
                "ACP session %s could not recover a provider entry for %r (base_url=%r, model=%r); "
                "falling back to the config default provider",
                session_id, requested_provider, base_url, model or default_model)
        try:
            runtime = resolve_runtime_provider(
                requested=requested_identity, target_model=(model or default_model) or None)
            # A *recovered* identity owns the endpoint: if the row's base_url is not where that entry
            # lives, the row is stale and pairing the entry's key with the old host is a
            # working-looking request to the wrong place (#63 review C). An explicitly named
            # ``custom:<name>`` keeps honouring a deliberately overridden base_url.
            if healed_identity and base_url:
                entry_url = str(runtime.get("base_url") or "")
                if entry_url and entry_url.rstrip("/") != base_url.rstrip("/"):
                    logger.warning(
                        "ACP session %s: persisted base_url %r is not where %s lives (%r); using the "
                        "provider entry's endpoint", session_id, base_url, healed_identity, entry_url)
                    base_url = None
            kwargs.update({
                "provider": runtime.get("provider"), "api_mode": api_mode or runtime.get("api_mode"),
                "base_url": base_url or runtime.get("base_url"), "api_key": runtime.get("api_key"),
                "credential_pool": runtime.get("credential_pool"),
                "command": runtime.get("command"), "args": list(runtime.get("args") or []),
            })
            # The resolved provider's request body (a custom entry's extra_body); an explicit base_url pointing
            # elsewhere is another endpoint, which must not inherit it.
            if runtime.get("request_overrides") and (not base_url or base_url == runtime.get("base_url")):
                kwargs["request_overrides"] = runtime["request_overrides"]
        except Exception as exc:
            resolve_error = exc
            # Not debug: a resolution failure here is the difference between "this session talks to
            # the endpoint it was created for" and "every turn of it silently goes to another
            # vendor" (#63). Said once per session — an unusable session retries this on every
            # prompt, and the traceback belongs to the first attempt only.
            if session_id and session_id not in _WARNED_UNROUTABLE:
                if len(_WARNED_UNROUTABLE) >= _WARNED_UNROUTABLE_MAX:
                    _WARNED_UNROUTABLE.clear()
                _WARNED_UNROUTABLE[session_id] = None
                logger.warning(
                    "ACP session %s could not resolve provider %r (base_url=%r); falling back to the "
                    "config default — turns will be answered by that provider, not this one: %s",
                    session_id, requested_identity, base_url, exc)

        _register_task_cwd(session_id, cwd)

        # Bounded wait for the background MCP discovery started by entry.py: the agent
        # snapshots tools once at build and never re-reads the registry, so without this
        # join a slow-but-reachable server would be invisible all session. ensure_* also
        # (re)starts discovery if the entry spawn never ran or connected zero servers.
        # Bounded by ``mcp_discovery_timeout`` (config.yaml, ~1.5s); late servers are
        # picked up by HermesACPAgent._schedule_mcp_late_refresh.
        try:
            from hermes_cli.mcp_startup import ensure_mcp_discovery_before_agent_build

            ensure_mcp_discovery_before_agent_build(logger=logger, thread_name="acp-mcp-discovery")
        except Exception:
            logger.debug("ACP: bounded MCP discovery wait failed", exc_info=True)

        try:
            agent = AIAgent(**kwargs)
        except Exception as exc:
            # The bare-AIAgent fallback dies with "No LLM provider configured. Run `hermes setup`" on a
            # machine that is configured and was working a call earlier; the swallowed resolution
            # failure (revoked OAuth, disabled provider, ...) is the actionable error (#91090).
            if resolve_error is not None:
                raise resolve_error from exc
            raise
        # ACP stdio: stdout is protocol-only JSON-RPC; agent chatter goes to stderr.
        agent._print_fn = _acp_stderr_print
        # Keep the *intent* (what the row asked for) next to the result (who actually answered), so the
        # persist path never writes a fallback back as if it were this session's own provider (#63
        # review B): that would erase the only evidence a row belongs to a custom entry.
        if requested_provider:
            agent._acp_requested_provider = requested_provider  # type: ignore[attr-defined]
        return agent
