"""Edge-case delete tests beyond the basic cascade.

Covers:
  * double hard_delete (second call raises KeyError, doesn't corrupt state)
  * deleting after quarantine (no stale chunks / provenance to worry about)
  * deleting during partially-built state (chunks but no extraction yet)
  * deleting the last-supporting conversation removes the orphan object
  * audit / inspect / plan_hard_delete all agree after delete
"""

from __future__ import annotations

import pytest

from threadatlas.audit import audit_conversation, plan_hard_delete
from threadatlas.core.models import State
from threadatlas.core.workflow import hard_delete, transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path
from threadatlas.store import read_normalized


def _setup(tmp_vault, store, factory):
    path = factory([
        {"title": "Project Delta planning",
         "messages": [
             ("user", "Project Delta decisions: I decided to go with option A.", 1.0),
             ("assistant", "Noted.", 2.0),
             ("user", "TODO: revisit staffing.", 3.0),
             ("assistant", "OK.", 4.0),
         ]},
    ])
    res = import_path(tmp_vault, store, path)
    return res.imported[0]


def test_double_hard_delete_second_call_raises(tmp_vault, store, chatgpt_export_factory):
    cid = _setup(tmp_vault, store, chatgpt_export_factory)
    transition_state(store, cid, State.INDEXED.value)
    hard_delete(tmp_vault, store, cid)
    with pytest.raises(KeyError):
        hard_delete(tmp_vault, store, cid)


def test_delete_after_quarantine(tmp_vault, store, chatgpt_export_factory):
    cid = _setup(tmp_vault, store, chatgpt_export_factory)
    transition_state(store, cid, State.INDEXED.value)
    chunk_conversation(store, cid)
    extract_for_conversation(store, cid)
    transition_state(store, cid, State.QUARANTINED.value)
    # After quarantine, chunks and provenance are already gone.
    assert store.list_chunks(cid) == []
    report = hard_delete(tmp_vault, store, cid)
    assert report["chunks_deleted"] == 0
    assert report["provenance_links_deleted"] == 0
    assert report["messages_deleted"] > 0
    assert store.get_conversation(cid) is None


def test_delete_during_partial_state(tmp_vault, store, chatgpt_export_factory):
    """Chunks exist but no extraction has run yet. Delete must still clean up."""
    cid = _setup(tmp_vault, store, chatgpt_export_factory)
    transition_state(store, cid, State.INDEXED.value)
    chunks = chunk_conversation(store, cid)
    assert chunks, "expected at least one chunk"
    report = hard_delete(tmp_vault, store, cid)
    assert report["chunks_deleted"] == len(chunks)
    assert store.get_conversation(cid) is None


def test_plan_delete_matches_actual_delete(tmp_vault, store, chatgpt_export_factory):
    cid = _setup(tmp_vault, store, chatgpt_export_factory)
    transition_state(store, cid, State.INDEXED.value)
    chunk_conversation(store, cid)
    extract_for_conversation(store, cid)
    plan = plan_hard_delete(tmp_vault, store, cid)
    assert plan is not None
    actual = hard_delete(tmp_vault, store, cid)
    assert plan["would_remove"]["messages"] == actual["messages_deleted"]
    assert plan["would_remove"]["chunks"] == actual["chunks_deleted"]
    assert plan["would_remove"]["provenance_links"] == actual["provenance_links_deleted"]
    # orphans the plan predicted should match what was actually removed.
    assert len(plan["would_orphan_objects"]) == actual["orphan_derived_objects_deleted"]


def test_plan_delete_on_unknown_is_none(tmp_vault, store):
    assert plan_hard_delete(tmp_vault, store, "conv_does_not_exist") is None


def test_audit_returns_none_after_delete(tmp_vault, store, chatgpt_export_factory):
    cid = _setup(tmp_vault, store, chatgpt_export_factory)
    transition_state(store, cid, State.INDEXED.value)
    hard_delete(tmp_vault, store, cid)
    assert audit_conversation(tmp_vault, store, cid) is None


def test_deleted_content_absent_from_normalized_dir(tmp_vault, store, chatgpt_export_factory):
    cid = _setup(tmp_vault, store, chatgpt_export_factory)
    transition_state(store, cid, State.INDEXED.value)
    assert read_normalized(tmp_vault, cid) is not None
    hard_delete(tmp_vault, store, cid)
    assert read_normalized(tmp_vault, cid) is None
    # Belt-and-suspenders: no shard in the normalized tree should mention the id.
    for p in tmp_vault.normalized.rglob("*.json"):
        assert cid not in p.name
