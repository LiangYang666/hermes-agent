"""Reclaiming a turn lease whose turn is gone: the ``session_busy`` that outlives its turn.

A turn stamps its lease on entry and deletes it on exit, so a process killed
mid-turn (an editor's SIGTERM of ``hermes acp``, a promote restart) leaves the row
behind and every resume answers ``session_busy`` until the TTL lapses — 841 s in
the incident this guards. ``release_orphan_turn_lease`` reclaims it early, but only
on proof; the four keeps below are the ones that must never be waived.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil
import pytest

import hermes_state_pidns
from hermes_state import SessionDB
from hermes_state_pidns import LocalPidNamespace

_OURS, _SIBLING = "111", "222"


@pytest.fixture(autouse=True)
def _pinned_namespace(monkeypatch):
    # The probe must behave identically on every host: pin the local namespace.
    monkeypatch.setattr(hermes_state_pidns, "_LOCAL_PID_NS", LocalPidNamespace(_OURS, True))


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    assert not psutil.pid_exists(proc.pid), "reaped pid unexpectedly reused"
    return proc.pid


def _holder(pid: int, *, pidns: str = _OURS, turn: str = "t1") -> str:
    return f"pid={pid}:pidns={pidns}:turn={turn}:platform=acp"


def _live() -> str:
    # The governor treats our own pid as alive without probing it.
    return _holder(os.getpid(), turn="live")


def _leased(tmp_path, holder: str, *, ttl: float = 300.0, session: str = "s1") -> SessionDB:
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session, source="test")
    assert db.try_acquire_session_turn_lease(session, holder, ttl_seconds=ttl) is True
    return db


# ── admitted: the turn behind the row is gone ──────────────────────────────


def test_expired_lease_is_reclaimed(tmp_path):
    db = _leased(tmp_path, _live(), ttl=0.05)
    time.sleep(0.1)
    assert db.release_orphan_turn_lease("s1") == "expired"
    assert db.session_turn_lease_expires_at("s1", _live()) is None


def test_dead_holder_lease_is_reclaimed_and_a_resume_gets_in(tmp_path):
    """The incident: the ACP child was SIGTERM'd mid-turn, its row outlived it."""
    dead = _holder(_dead_pid())
    db = _leased(tmp_path, dead)
    assert db.release_orphan_turn_lease("s1") == "holder_dead"
    contender = _holder(os.getpid(), turn="resume")
    assert db.try_acquire_session_turn_lease("s1", contender, ttl_seconds=300) is True


def test_detached_evidence_reclaims_a_live_holder(tmp_path):
    """The holder still lives, but its editor is gone: nobody can observe that turn."""
    db = _leased(tmp_path, _live())
    assert db.release_orphan_turn_lease("s1", evidence="detached") == "detached"


def test_aborted_evidence_reclaims_a_live_holder(tmp_path):
    db = _leased(tmp_path, _live())
    assert db.release_orphan_turn_lease("s1", evidence="aborted") == "aborted"


# ── kept: the row still names a turn somebody is running ───────────────────


def test_live_holder_without_evidence_is_kept(tmp_path):
    holder = _live()
    db = _leased(tmp_path, holder)
    assert db.release_orphan_turn_lease("s1") is None
    assert db.session_turn_lease_expires_at("s1", holder) is not None


def test_unprobeable_namespace_is_kept(tmp_path):
    """A sibling namespace's live turn reads as a dead pid here — never steal it."""
    db = _leased(tmp_path, _holder(_dead_pid(), pidns=_SIBLING))
    assert db.release_orphan_turn_lease("s1") is None


def test_active_holder_is_kept_even_with_evidence(tmp_path):
    holder = _live()
    db = _leased(tmp_path, holder)
    assert db.release_orphan_turn_lease("s1", active_holder=holder, evidence="detached") is None
    assert db.session_turn_lease_expires_at("s1", holder) is not None


def test_no_row_is_a_noop(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    assert db.release_orphan_turn_lease("s1") is None
    assert db.release_orphan_turn_lease("s1", evidence="detached") is None


# ── the exit path: a SIGTERM'd turn releases its own row ───────────────────


def test_shutdown_hook_releases_the_in_flight_lease(tmp_path):
    """`hermes acp` killed mid-turn never reaches its own release; the exit hook must."""
    from types import SimpleNamespace

    from agent import turn_facade_lease as lease_mod

    holder = _holder(os.getpid(), turn="killed")
    db = _leased(tmp_path, holder)
    lease = lease_mod.DurableTurnLease(SimpleNamespace(), db, "s1", holder)
    lease_mod._LIVE_LEASES.add(lease)

    assert lease_mod.release_live_leases_on_shutdown() == 1
    assert db.session_turn_lease_expires_at("s1", holder) is None
    # Idempotent: the atexit pass after a signal pass finds nothing left to do.
    assert lease_mod.release_live_leases_on_shutdown() == 0


def test_finished_lease_is_not_released_twice(tmp_path):
    from types import SimpleNamespace

    from agent import turn_facade_lease as lease_mod

    holder = _holder(os.getpid(), turn="done")
    db = _leased(tmp_path, holder)
    lease = lease_mod.DurableTurnLease(SimpleNamespace(), db, "s1", holder)
    lease_mod._LIVE_LEASES.add(lease)
    lease.release()
    assert holder not in {getattr(item, "holder", None) for item in lease_mod._LIVE_LEASES}
    assert lease_mod.release_live_leases_on_shutdown() == 0
