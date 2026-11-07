"""A pre-gateway chat adopted at its compression tip still answers to the id an editor holds.

`sessions adopt` binds a pre-gateway chat at its compression TIP only — the receipt's lineage is
``[tip]``, because ``validate_local_lineage`` forces ``lineage[0] == legacy_lineage_root()`` and that
root is the tip itself for a legacy receipt. An editor that talked to the pre-gateway ACP server
holds the chain's CREATION id, so resolving the owner at the root alone answers ``not_found`` and
the session never reopens.
"""
import json
from contextlib import closing

from acp_adapter.catalog import logical_session_id
from hermes_state import SessionDB
from hermes_state_local import POLICY_PREFIX


def _adopted_chain(db, cwd):
    """root —compressed→ mid —compressed→ tip, with one receipt bound at the tip."""
    for sid, parent in (("chain-root", None), ("chain-mid", "chain-root"), ("chain-tip", "chain-mid")):
        extra = {} if parent is None else {
            "parent_session_id": parent, "model_config": {"_compressed_from": parent}}
        db.create_session(sid, source="acp", cwd=str(cwd), **extra)
        db.append_message(sid, "user", sid)
        if sid != "chain-tip":
            db.end_session(sid, "compression")
    db.set_meta(POLICY_PREFIX + "chain-tip",
                json.dumps({"session_id": "chain-tip", "lineage": ["chain-tip"]}))
    return "chain-root", "chain-mid", "chain-tip"


def test_every_id_in_an_adopted_chain_resolves_to_the_owner(tmp_path):
    path = tmp_path / "state.db"
    with closing(SessionDB(path)) as db:
        root, mid, tip = _adopted_chain(db, tmp_path)
        # The editor's id really is the root of a real compression chain (not a reset fork).
        assert db.get_compression_lineage(root) == [root, mid, tip]
    for held in (root, mid, tip):
        assert logical_session_id(path, held) == tip, held


def test_a_chain_nobody_adopted_resolves_to_its_creation_id(tmp_path):
    """Without a receipt the tip-first walk finds nothing and the compression root stands."""
    path = tmp_path / "state.db"
    with closing(SessionDB(path)) as db:
        db.create_session("orphan", source="acp", cwd=str(tmp_path))
        db.append_message("orphan", "user", "hi")
        db.end_session("orphan", "compression")
        db.create_session("orphan-tip", source="acp", cwd=str(tmp_path),
                          parent_session_id="orphan", model_config={"_compressed_from": "orphan"})
    assert logical_session_id(path, "orphan") == "orphan"
    assert logical_session_id(path, "orphan-tip") == "orphan"
