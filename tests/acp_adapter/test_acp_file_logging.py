"""Regression: the ACP entry point must write the normal Hermes log files (issue #45).

``acp_adapter.entry._setup_logging`` used to install a stderr-only root handler and stop there, so
cockpit sessions were the one launch mode that never appeared in ``~/.hermes/logs/agent.log``: a
compaction timeout or a provider 504 inside an agent left no process-side trace at all (the
cockpit server keeps only an in-memory stderr ring for crash text).

Two invariants:
1. ``_setup_logging`` joins the shared logging subsystem (``setup_logging(mode="acp")``).
2. It still never routes console logging to stdout — ACP JSON-RPC owns stdout.
"""

import logging
import os
from pathlib import Path

import hermes_logging


def test_setup_logging_joins_the_file_logging_subsystem(monkeypatch):
    from acp_adapter import entry

    calls = []

    def fake_setup_logging(*, mode=None, **kwargs):
        calls.append(mode)
        return Path("/tmp/logs")

    monkeypatch.setattr(hermes_logging, "setup_logging", fake_setup_logging)
    entry._setup_logging()

    assert calls == ["acp"], "the ACP entry must add the shared FILE handlers, in its own mode"


def test_setup_logging_never_streams_to_stdout(monkeypatch):
    from acp_adapter import entry

    monkeypatch.setattr(hermes_logging, "setup_logging", lambda **kw: Path("/tmp/logs"))
    entry._setup_logging()

    streams = [getattr(h, "stream", None) for h in logging.getLogger().handlers]
    import sys
    assert sys.stdout not in streams, "stdout is reserved for the ACP JSON-RPC wire"


def test_it_does_not_tear_down_the_cli_file_logging(monkeypatch):
    """``hermes acp`` runs inside the CLI, which already initialised file logging at import time.

    ``setup_logging`` attaches its QueueHandler on the FIRST call only, so an ACP entry that cleared
    root's handler list would leave the process with no file logging — and the follow-up
    ``setup_logging(mode="acp")`` could not restore it (path dedup + the initialised guard).
    """
    from acp_adapter import entry

    monkeypatch.setattr(hermes_logging, "setup_logging", lambda **kw: Path("/tmp/logs"))
    root = logging.getLogger()
    queue_ish = logging.NullHandler()
    queue_ish._hermes_queue = True  # same marker hermes_logging uses
    root.handlers.clear()
    root.addHandler(queue_ish)
    try:
        entry._setup_logging()
        assert queue_ish in root.handlers, "the CLI's file-logging handler must survive an ACP start"
    finally:
        root.handlers.clear()


def test_a_second_call_replaces_only_its_own_stderr_handler(monkeypatch):
    from acp_adapter import entry

    monkeypatch.setattr(hermes_logging, "setup_logging", lambda **kw: Path("/tmp/logs"))
    root = logging.getLogger()
    root.handlers.clear()
    try:
        entry._setup_logging()
        entry._setup_logging()
        stderr_handlers = [h for h in root.handlers if getattr(h, "_hermes_acp_stderr", False)]
        assert len(stderr_handlers) == 1, "repeated setup must not stack stderr handlers"
    finally:
        root.handlers.clear()


def test_an_acp_session_actually_lands_in_agent_log(capsys):
    """End-to-end: a record logged from the adapter reaches ``agent.log``."""
    from acp_adapter import entry

    home = Path(os.environ["HERMES_HOME"])
    hermes_logging._logging_initialized = False
    hermes_logging._reset_queued_handlers()
    try:
        entry._setup_logging()
        logging.getLogger("acp_adapter.server").info("ACP_SESSION_MARKER turn failed")
        hermes_logging._reset_queued_handlers()   # stops the queue listener, flushing to disk
        agent_log = home / "logs" / "agent.log"
        assert agent_log.exists(), "the ACP entry must produce agent.log"
        assert "ACP_SESSION_MARKER" in agent_log.read_text(encoding="utf-8", errors="replace")
    finally:
        hermes_logging._reset_queued_handlers()
        logging.getLogger().handlers.clear()
        hermes_logging._logging_initialized = False
