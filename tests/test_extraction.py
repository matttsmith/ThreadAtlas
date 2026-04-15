"""Heuristic extraction sanity tests."""

from __future__ import annotations

from threadatlas.core.models import DerivedKind, State
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path


def _setup(tmp_vault, store, factory):
    path = factory([
        {"title": "Project Atlas roadmap",
         "messages": [
             ("user", "Project Atlas is the new initiative. I will pick option B.", 1.0),
             ("assistant", "We agreed to ship in April.", 2.0),
             ("user", "TODO: revisit staffing next week. Also remember to email Alice Johnson.", 3.0),
             ("assistant", "Got it. I'll keep Alice Johnson posted.", 4.0),
             ("user", "I prefer async standups.", 5.0),
             ("assistant", "Confirmed.", 6.0),
         ]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value)
    chunk_conversation(store, cid)
    return cid


def test_extraction_finds_each_kind(tmp_vault, store, chatgpt_export_factory):
    cid = _setup(tmp_vault, store, chatgpt_export_factory)
    counts = extract_for_conversation(store, cid)
    assert counts.get(DerivedKind.PROJECT.value, 0) >= 1
    assert counts.get(DerivedKind.DECISION.value, 0) >= 1
    assert counts.get(DerivedKind.OPEN_LOOP.value, 0) >= 1
    assert counts.get(DerivedKind.PREFERENCE.value, 0) >= 1
    assert counts.get(DerivedKind.ENTITY.value, 0) >= 1


def test_extraction_marks_open_loops_on_conversation(tmp_vault, store, chatgpt_export_factory):
    cid = _setup(tmp_vault, store, chatgpt_export_factory)
    extract_for_conversation(store, cid)
    c = store.get_conversation(cid)
    assert c.has_open_loops is True
    assert c.importance_score > 0
    assert c.resurfacing_score > 0


def test_extraction_idempotent(tmp_vault, store, chatgpt_export_factory):
    cid = _setup(tmp_vault, store, chatgpt_export_factory)
    extract_for_conversation(store, cid)
    pre = store.conn.execute("SELECT COUNT(*) AS c FROM provenance_links WHERE conversation_id=?", (cid,)).fetchone()["c"]
    extract_for_conversation(store, cid)
    post = store.conn.execute("SELECT COUNT(*) AS c FROM provenance_links WHERE conversation_id=?", (cid,)).fetchone()["c"]
    # Same input -> same provenance count (we wipe + re-insert each run).
    assert pre == post


def test_extraction_skipped_for_quarantined(tmp_vault, store, chatgpt_export_factory):
    cid = _setup(tmp_vault, store, chatgpt_export_factory)
    extract_for_conversation(store, cid)
    transition_state(store, cid, State.QUARANTINED.value)
    counts = extract_for_conversation(store, cid)
    assert counts == {}
