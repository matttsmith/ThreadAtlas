"""State transition rules and visibility behavior."""

from __future__ import annotations

import pytest

from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation
from threadatlas.ingest import import_path


def _import_one(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "Sample", "messages": [("user", "x", 1.0), ("assistant", "y", 2.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    return res.imported[0]


def test_pending_to_indexed(tmp_vault, store, chatgpt_export_factory):
    cid = _import_one(tmp_vault, store, chatgpt_export_factory)
    new = transition_state(store, cid, State.INDEXED.value)
    assert new == State.INDEXED.value
    c = store.get_conversation(cid)
    assert c.state == State.INDEXED.value


def test_disallow_direct_delete_via_transition(tmp_vault, store, chatgpt_export_factory):
    cid = _import_one(tmp_vault, store, chatgpt_export_factory)
    with pytest.raises(ValueError):
        transition_state(store, cid, State.DELETED.value)


def test_quarantine_strips_chunks_and_provenance(tmp_vault, store, chatgpt_export_factory):
    cid = _import_one(tmp_vault, store, chatgpt_export_factory)
    transition_state(store, cid, State.INDEXED.value)
    chunk_conversation(store, cid)
    assert store.list_chunks(cid)
    transition_state(store, cid, State.QUARANTINED.value)
    # quarantined conversations have no chunks
    assert store.list_chunks(cid) == []
    c = store.get_conversation(cid)
    assert c.state == State.QUARANTINED.value


def test_unknown_conversation_raises(tmp_vault, store):
    with pytest.raises(KeyError):
        transition_state(store, "conv_nosuch", State.INDEXED.value)


def test_message_state_inherits_on_transition(tmp_vault, store, chatgpt_export_factory):
    cid = _import_one(tmp_vault, store, chatgpt_export_factory)
    transition_state(store, cid, State.PRIVATE.value)
    msgs = store.list_messages(cid)
    assert msgs and all(m.visibility_state_inherited == State.PRIVATE.value for m in msgs)
