"""Chunking tests: thematic boundaries and quarantine clearing."""

from __future__ import annotations

from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation
from threadatlas.ingest import import_path


def test_chunking_preserves_message_boundaries(tmp_vault, store, chatgpt_export_factory):
    convs = [{
        "title": "Mixed",
        "messages": [
            ("user", "Tell me about CHS strategy in depth, with several paragraphs of context.", 1.0),
            ("assistant", "CHS is a public health initiative " * 50, 2.0),
            ("user", "Now totally different subject - help me cook risotto.", 3.0),
            ("assistant", "Risotto requires arborio rice " * 50, 4.0),
            ("user", "Back to CHS for a moment.", 5.0),
            ("assistant", "Sure thing " * 50, 6.0),
        ],
    }]
    path = chatgpt_export_factory(convs)
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value)
    chunks = chunk_conversation(store, cid)
    assert chunks, "expected at least one chunk"
    # All chunks must align to message ordinals.
    msgs = store.list_messages(cid)
    ords = {m.ordinal for m in msgs}
    for c in chunks:
        assert c.start_message_ordinal in ords
        assert c.end_message_ordinal in ords
        assert c.start_message_ordinal <= c.end_message_ordinal


def test_chunking_quarantined_clears(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "X", "messages": [("user", "a", 1.0), ("assistant", "b", 2.0),
                                    ("user", "c", 3.0), ("assistant", "d", 4.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value)
    assert chunk_conversation(store, cid)
    transition_state(store, cid, State.QUARANTINED.value)
    # Quarantine strips chunks and chunking on quarantined state must produce nothing.
    chunks = chunk_conversation(store, cid)
    assert chunks == []
