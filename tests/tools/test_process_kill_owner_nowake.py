"""A kill the agent issued for its OWN process must not wake the agent (issue #49).

A kill is a synchronous, deliberate decision: whoever issued it already has the outcome. The contract
used to be "any exit of a notify_on_complete session wakes the agent", so a turn that cleaned up three
bounded retry loops it had started got three spurious turns afterwards, each announcing a result it
already knew.

Suppressed only when the exit is (a) a kill, (b) from the tool surface, and (c) issued by the same
task that spawned the process. Everything the agent could NOT know still wakes it — a kill from
another session, a supervisor shutdown, `backend_lost`, `failed_start`.
"""

import contextlib
import shlex
import sys
import time

import tools.process_registry as module
from tools.process_registry import ProcessRegistry, ProcessSession


def _spawn(registry: ProcessRegistry, owner: str = "owner-1", notify: bool = True):
    # SIGTERM-ignoring child: a plain sleeper exits so fast that the reader thread finalises the
    # exit before the kill path can write its receipt, which makes every assertion about the receipt
    # a coin flip. The real case (a retry loop, a npm publish probe) ignores/defers signals long
    # enough for the kill path to own the receipt — reproduce THAT.
    code = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n    time.sleep(0.05)"
    )
    session = registry.spawn_local(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
        owner_task_id=owner,
        task_id="task",
    )
    session.notify_on_complete = notify
    # wait until the process is really up, so a kill lands on a live child
    deadline = time.monotonic() + 10
    while session.pid is None and time.monotonic() < deadline:
        time.sleep(0.01)
    return session


def _notices_for(session_id: str) -> list[dict]:
    return [e for e in list(module.process_registry.completion_queue.queue) if e.get("session_id") == session_id]


@contextlib.contextmanager
def _clean_registry():
    registry = ProcessRegistry()
    previous = module.process_registry
    module.process_registry = registry
    try:
        yield registry
    finally:
        for session in list(registry._running.values()) + list(registry._finished.values()):
            proc = session.process
            if proc is not None:
                with contextlib.suppress(Exception):
                    proc.kill()
        module.process_registry = previous


def test_owner_kill_does_not_wake_the_owner():
    """The reported case: the agent kills a process it started in the same turn."""
    with _clean_registry() as registry:
        session = _spawn(registry, owner="owner-1")
        result = registry.kill_process(session.id, caller_task_id="owner-1")
        assert result["status"] == "killed"
        time.sleep(0.2)
        assert _notices_for(session.id) == [], "a self-inflicted kill woke the agent anyway"
        # …and nothing is lost: the result is still readable
        polled = registry.poll(session.id)
        # The result stays readable — that is the promise. WHICH receipt wins (`killed` from the kill
        # path vs `exited` from the reader thread that can finalise first) is a pre-existing race and
        # deliberately not pinned here: this change only removes the WAKE, never the record.
        assert polled["status"] in ("killed", "exited")
        assert session.exited is True


def test_kill_from_another_session_still_wakes():
    """A different session killing your job is news: it must wake you."""
    with _clean_registry() as registry:
        session = _spawn(registry, owner="owner-1")
        registry.kill_process(session.id, caller_task_id="someone-else")
        deadline = time.monotonic() + 10
        while not _notices_for(session.id) and time.monotonic() < deadline:
            time.sleep(0.02)
        notices = _notices_for(session.id)
        assert len(notices) == 1, f"expected exactly one wake, got {len(notices)}"
        assert notices[0]["completion_reason"] in ("killed", "exited")


def test_kill_without_a_caller_still_wakes():
    """Shutdown / abandoned-turn sweeps pass no caller: the agent cannot know, so it wakes."""
    with _clean_registry() as registry:
        session = _spawn(registry, owner="owner-1")
        registry.kill_process(session.id)
        deadline = time.monotonic() + 10
        while not _notices_for(session.id) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(_notices_for(session.id)) == 1


def test_backend_lost_still_wakes_even_for_the_owner():
    """`backend_lost` is exactly the case the agent does NOT know about."""
    with _clean_registry() as registry:
        # No child and no reader thread: this test is about the DECISION, and a live process would
        # race it (its exit can be finalised before the fields below are set).
        session = ProcessSession(
            id="proc_backend_lost", command="cmd", task_id="task", owner_task_id="owner-1",
            started_at=time.time(), notify_on_complete=True,
        )
        registry._running[session.id] = session
        with session._lock:
            session.mark_exited(-9, "lost", "backend_lost")
        session.kill_requested_by_owner = True   # the owner did ask… but the backend died, not the kill
        registry._move_to_finished(session)
        deadline = time.monotonic() + 10
        while not _notices_for(session.id) and time.monotonic() < deadline:
            time.sleep(0.02)
        notices = _notices_for(session.id)
        assert len(notices) == 1, "backend_lost must always wake: the agent cannot detect it"
        assert notices[0]["completion_reason"] == "lost"


def test_plain_exit_still_wakes():
    with _clean_registry() as registry:
        session = _spawn(registry, owner="owner-1")
        proc = session.process
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()  # the reader thread finishes it as a normal exit
        deadline = time.monotonic() + 10
        while not _notices_for(session.id) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(_notices_for(session.id)) == 1


def test_tool_surface_passes_the_caller_through():
    """The gate is useless if the tool handler forgets who called it."""
    import json

    with _clean_registry() as registry:
        session = _spawn(registry, owner="owner-1")
        module._handle_process({"action": "kill", "session_id": session.id}, task_id="owner-1")
        time.sleep(0.2)
        assert _notices_for(session.id) == []
        assert json.loads(json.dumps(registry.poll(session.id)))["status"] in ("killed", "exited")


# --------------------------------------------------------------------------------------------------
# Review round 1 (changes-requested): a sticky boolean could not tell "the kill I asked for" from
# "the exit that happened anyway". These four cases pin the corrected semantics. They drive the
# DECISION directly (no child, no reader thread) so they are deterministic.
# --------------------------------------------------------------------------------------------------

def _decided_session(registry: ProcessRegistry, *, owner="owner-1", **fields) -> ProcessSession:
    session = ProcessSession(
        id=f"proc_decided_{time.monotonic_ns()}", command="cmd", task_id="task",
        owner_task_id=owner, started_at=time.time(), notify_on_complete=True,
    )
    registry._running[session.id] = session
    for key, value in fields.items():
        setattr(session, key, value)
    return session


def test_failed_kill_does_not_silence_the_later_natural_exit():
    """B1: a kill that reported "still running" must not mute the exit that follows it."""
    with _clean_registry() as registry:
        session = _decided_session(registry, kill_intent_caller="", kill_intent_at=0.0)
        with session._lock:
            session.mark_exited(0, "exited")
        registry._move_to_finished(session)
        assert len(_notices_for(session.id)) == 1, "a natural exit after a failed kill must still wake"


def test_kill_from_another_task_after_an_owner_attempt_still_wakes():
    """B2: the gate must follow who COMPLETED the kill, not who once asked for one."""
    with _clean_registry() as registry:
        session = _decided_session(
            registry,
            # the owner asked earlier and its kill failed; a different task then killed it
            kill_intent_caller="owner-1", kill_intent_at=time.time() - 5,
            termination_caller="someone-else", termination_source="process.kill",
            exit_code=-15,
        )
        with session._lock:
            session.mark_exited(-15, "killed", "process.kill")
        registry._move_to_finished(session)
        assert len(_notices_for(session.id)) == 1, "another session's kill is news to the owner"


def test_owner_kill_intent_does_not_swallow_a_clean_exit():
    """M1: the child exited 0 on its own before our signal landed — that is not a kill."""
    with _clean_registry() as registry:
        session = _decided_session(registry, kill_intent_caller="owner-1", kill_intent_at=time.time() - 1)
        with session._lock:
            session.mark_exited(0, "exited")
        registry._move_to_finished(session)
        assert len(_notices_for(session.id)) == 1, "exit 0 must be reported with its real status"


def test_owner_kill_on_an_already_exited_child_is_silent_when_it_was_our_signal():
    """M2: the wake path that actually fires for a self-kill is `already_exited` + reader draining."""
    with _clean_registry() as registry:
        session = _decided_session(registry, kill_intent_caller="owner-1", kill_intent_at=time.time() - 1)
        with session._lock:
            session.mark_exited(-15, "exited")   # reader-authored receipt for our SIGTERM
        registry._move_to_finished(session)
        assert _notices_for(session.id) == [], "our own SIGTERM, finalised by the reader: stay silent"
