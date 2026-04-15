"""Import pipeline: lands in pending_review, dedupes, normalized files written."""

from __future__ import annotations

from threadatlas.core.models import State
from threadatlas.ingest import import_path
from threadatlas.store import read_normalized


def test_import_lands_in_pending_review(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "T1", "messages": [("user", "hi", 1.0), ("assistant", "hello", 2.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    assert len(res.imported) == 1
    assert not res.failed
    convs = store.list_conversations()
    assert len(convs) == 1
    assert convs[0].state == State.PENDING_REVIEW.value
    # Pending review must NOT be MCP-visible: i.e., not in MCP_VISIBLE_STATES.
    from threadatlas.core.models import MCP_VISIBLE_STATES
    assert convs[0].state not in MCP_VISIBLE_STATES


def test_normalized_file_is_written(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "T1", "messages": [("user", "hi", 1.0), ("assistant", "hello", 2.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    payload = read_normalized(tmp_vault, cid)
    assert payload is not None
    assert payload["conversation"]["title"] == "T1"
    assert len(payload["messages"]) == 2


def test_dedupe_on_reimport(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "T1", "messages": [("user", "hi", 1.0), ("assistant", "hello", 2.0)]},
    ])
    r1 = import_path(tmp_vault, store, path)
    r2 = import_path(tmp_vault, store, path)
    assert len(r1.imported) == 1
    assert len(r2.imported) == 0
    assert len(r2.deduped) == 1
    assert len(store.list_conversations()) == 1


def test_raw_input_copied_into_raw_imports(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "T1", "messages": [("user", "hi", 1.0), ("assistant", "hello", 2.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    assert res.raw_path is not None
    assert tmp_vault.raw_imports in res.raw_path.parents
