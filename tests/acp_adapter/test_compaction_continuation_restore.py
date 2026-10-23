"""An ACP restore must follow the compaction continuation, not reload the pre-compaction transcript.

Operator report (2026-10-23): "为什么一直触发这个压缩 … 压缩完上下文长度没变". Logs on a live cockpit
session showed four summary calls in nine hours, two of them three minutes apart:

    08:31:14 compression 392->41 messages (~262K -> ~53K)
    08:33:09 Restored ACP session 61315704… from DB (490 messages)
    08:34:16 compression 494->21 messages (~352K -> ~53K)

The summariser worked every time. What undid it is this: a compaction that cannot commit in place ROTATES —
it ends the parent row with ``end_reason='compression'`` and publishes the compacted transcript as a NEW row
whose ``parent_session_id`` points back at the parent. The ACP adapter, however, keeps ONE client-facing
session id for the life of a conversation (the cockpit's), so ``_restore`` loaded the PARENT transcript:
every restart re-fed the tokens the compaction had just removed, the next request crossed the trigger again,
and the operator paid another ~50-100s summary call for nothing.

Invariants pinned here:

1. Restoring a compacted session loads the TIP's compacted history, not the parent's full transcript.
2. The parent keeps ``end_reason='compression'`` (the edge a later restore depends on) — ``reopen_session``
   runs on the tip, never on the parent.
3. A chain of rotations (parent -> child -> grandchild) resolves to the grandchild.
4. A session that never rotated restores exactly as before (no regression).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from acp_adapter.session import SessionManager
from hermes_state import SessionDB


def _row(db, sid):
    return db.get_session(sid)


def _session(db, sid, *, cwd, ended=False, parent=None):
    db.create_session(
        sid, "acp",
        model="fixture-model",
        model_config=json.dumps({"cwd": cwd}),
        parent_session_id=parent,
    )
    if ended:
        db.end_session(sid, "compression")
    return sid


def _turn(db, sid, text):
    db.append_message(sid, "user", text)
    db.append_message(sid, "assistant", f"re: {text}")


def _manager(db):
    return SessionManager(db=db, agent_factory=lambda: SimpleNamespace(model="fixture-model"))


def test_restore_loads_the_compacted_continuation(tmp_path):
    """The reported shape: parent compacted (ended, compression), child holds the summary."""
    db = SessionDB(tmp_path / "state.db")
    cwd = str(tmp_path)
    parent = _session(db, "parent-acp-id", cwd=cwd)
    for i in range(3):
        _turn(db, parent, f"old turn {i}")
    db.end_session(parent, "compression")
    child = _session(db, "child-after-compaction", cwd=cwd, parent=parent)
    _turn(db, child, "[summary of the conversation so far]")

    state = _manager(db)._restore(parent)

    assert state is not None
    # Compacted history (2 rows), not the 6 pre-compaction rows.
    assert len(state.history) == 2
    assert "summary of the conversation so far" in state.history[0]["content"]
    assert all("old turn" not in (m.get("content") or "") for m in state.history)
    # The client-facing id stays what the cockpit knows.
    assert state.session_id == parent
    # The edge survives: reopening the parent would clear end_reason and strand the next restore.
    assert _row(db, parent)["end_reason"] == "compression"
    db.close()


def test_restore_follows_a_chain_of_rotations(tmp_path):
    """Two compactions in a row: parent -> child -> grandchild; the grandchild is the live tip."""
    db = SessionDB(tmp_path / "state.db")
    cwd = str(tmp_path)
    parent = _session(db, "p", cwd=cwd)
    _turn(db, parent, "ancient")
    db.end_session(parent, "compression")
    child = _session(db, "c1", cwd=cwd, parent=parent)
    _turn(db, child, "first summary")
    db.end_session(child, "compression")
    grandchild = _session(db, "c2", cwd=cwd, parent=child)
    _turn(db, grandchild, "second summary")

    state = _manager(db)._restore(parent)

    assert state is not None
    assert len(state.history) == 2
    assert "second summary" in state.history[0]["content"]
    db.close()


def test_restore_of_an_uncompacted_session_is_unchanged(tmp_path):
    """No continuation edge: behaviour is byte-for-byte the old path."""
    db = SessionDB(tmp_path / "state.db")
    cwd = str(tmp_path)
    plain = _session(db, "plain", cwd=cwd)
    for i in range(2):
        _turn(db, plain, f"turn {i}")

    state = _manager(db)._restore(plain)

    assert state is not None
    assert len(state.history) == 4
    assert state.session_id == plain
    db.close()
