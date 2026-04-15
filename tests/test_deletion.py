"""Hard deletion cascade tests.

These are the most security-critical tests in the suite. If they regress,
deletion is no longer trustworthy and the privacy model breaks.
"""

from __future__ import annotations

from threadatlas.core.models import State
from threadatlas.core.workflow import hard_delete, transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path
from threadatlas.store import read_normalized
from threadatlas.search import search_conversations, search_chunks


def _setup_indexed_with_extraction(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "Project Atlas planning",
         "messages": [
             ("user", "Project Atlas. Decision: I will pick the smaller team.", 1.0),
             ("assistant", "Sounds good. We agreed to ship in April.", 2.0),
             ("user", "TODO: follow up on staffing next week.", 3.0),
             ("assistant", "Noted. I'll remind you.", 4.0),
             ("user", "I prefer async standups.", 5.0),
             ("assistant", "Confirmed.", 6.0),
         ]},
        {"title": "Other thread",
         "messages": [
             ("user", "Project Atlas comes up again here.", 7.0),
             ("assistant", "Yes, we discussed it last time.", 8.0),
         ]},
    ])
    res = import_path(tmp_vault, store, path)
    cid_a, cid_b = res.imported
    transition_state(store, cid_a, State.INDEXED.value)
    transition_state(store, cid_b, State.INDEXED.value)
    chunk_conversation(store, cid_a)
    chunk_conversation(store, cid_b)
    extract_for_conversation(store, cid_a)
    extract_for_conversation(store, cid_b)
    return cid_a, cid_b


def test_hard_delete_removes_messages_chunks_and_normalized_file(tmp_vault, store, chatgpt_export_factory):
    cid_a, cid_b = _setup_indexed_with_extraction(tmp_vault, store, chatgpt_export_factory)
    assert read_normalized(tmp_vault, cid_a) is not None
    report = hard_delete(tmp_vault, store, cid_a)
    assert report["messages_deleted"] > 0
    assert report["normalized_file_removed"] is True
    assert store.get_conversation(cid_a) is None
    assert store.list_messages(cid_a) == []
    assert store.list_chunks(cid_a) == []
    assert read_normalized(tmp_vault, cid_a) is None
    # Other conversation untouched.
    assert store.get_conversation(cid_b) is not None


def test_hard_delete_removes_orphan_derived_objects(tmp_vault, store, chatgpt_export_factory):
    """Derived objects that exist only because of the deleted conversation must go."""
    path = chatgpt_export_factory([
        {"title": "Lonely thread",
         "messages": [
             ("user", "I will pick option A. TODO: revisit later.", 1.0),
             ("assistant", "Okay.", 2.0),
             ("user", "Project Lonelyproject is a thing.", 3.0),
             ("assistant", "Noted.", 4.0),
         ]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value)
    chunk_conversation(store, cid)
    extract_for_conversation(store, cid)
    pre = store.conn.execute("SELECT COUNT(*) AS c FROM derived_objects").fetchone()["c"]
    assert pre > 0
    hard_delete(tmp_vault, store, cid)
    post = store.conn.execute("SELECT COUNT(*) AS c FROM derived_objects").fetchone()["c"]
    assert post == 0


def test_shared_derived_object_survives_partial_delete(tmp_vault, store, chatgpt_export_factory):
    """A derived object referenced by two conversations should survive when
    only one of them is deleted, just losing one provenance link."""
    cid_a, cid_b = _setup_indexed_with_extraction(tmp_vault, store, chatgpt_export_factory)
    # 'Project Atlas' should be derived from both convs.
    rows = store.conn.execute(
        "SELECT object_id, title FROM derived_objects WHERE kind='project' AND lower(title) LIKE '%atlas%'"
    ).fetchall()
    assert rows, "Expected a Project Atlas object linked from both conversations"
    obj_id = rows[0]["object_id"]
    pre_links = store.list_provenance_for_object(obj_id)
    assert len({l.conversation_id for l in pre_links}) >= 2

    hard_delete(tmp_vault, store, cid_a)
    surviving = store.get_derived_object(obj_id)
    assert surviving is not None
    post_links = store.list_provenance_for_object(obj_id)
    assert all(l.conversation_id != cid_a for l in post_links)


def test_hard_delete_removes_from_search(tmp_vault, store, chatgpt_export_factory):
    cid_a, cid_b = _setup_indexed_with_extraction(tmp_vault, store, chatgpt_export_factory)
    visible = (State.INDEXED.value,)
    assert search_conversations(store, "staffing", visible_states=visible)
    hard_delete(tmp_vault, store, cid_a)
    # After delete, no conv search hit should reference the deleted id.
    hits = search_conversations(store, "staffing", visible_states=visible)
    assert all(h.conversation_id != cid_a for h in hits)
    # And chunk search on removed text returns nothing for cid_a.
    chunk_hits = search_chunks(store, "staffing", visible_states=visible)
    assert all(h.conversation_id != cid_a for h in chunk_hits)
