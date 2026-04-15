"""Disaster recovery: DB rebuild from normalized JSON files."""

from __future__ import annotations

from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.ingest import import_path
from threadatlas.recovery import rebuild_from_normalized


def test_rebuild_restores_conversations_and_messages(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "Project CHS planning",
         "messages": [
             ("user", "Project CHS staffing plan.", 1.0),
             ("assistant", "Got it.", 2.0),
             ("user", "TODO: revisit staffing.", 3.0),
             ("assistant", "Noted.", 4.0),
         ]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    # Pass vault so normalized file captures the new state.
    transition_state(store, cid, State.INDEXED.value, vault=tmp_vault)
    # Close so rebuild can manage the file itself.
    store.close()

    result = rebuild_from_normalized(tmp_vault)
    assert result.conversations_restored == 1
    assert result.backup_path is not None and result.backup_path.exists()
    assert result.chunks_built >= 1
    assert result.extraction_ran == 1

    # Reopen and confirm the conversation + messages are there and state is preserved.
    from threadatlas.store import open_store
    store2 = open_store(tmp_vault)
    try:
        c = store2.get_conversation(cid)
        assert c is not None
        assert c.state == State.INDEXED.value
        assert c.title == "Project CHS planning"
        msgs = store2.list_messages(cid)
        assert len(msgs) == 4
        # Chunks + derived objects restored via re-extraction.
        assert store2.list_chunks(cid)
    finally:
        store2.close()
