"""Search visibility tests.

The crux: search must respect visibility states. MCP-visible search may only
return ``indexed`` content. The CLI may opt into private content but must
never see quarantined or pending_review content with text exposed.
"""

from __future__ import annotations

from threadatlas.core.models import MCP_VISIBLE_STATES, State
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation
from threadatlas.ingest import import_path
from threadatlas.search import search_conversations, search_chunks


def test_pending_review_not_searchable(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "Top secret quarterly numbers",
         "messages": [("user", "share quarterly", 1.0), ("assistant", "secret stuff", 2.0)]},
    ])
    import_path(tmp_vault, store, path)
    # Even when querying with the full set of visible states for search,
    # pending_review must be excluded.
    visible = (State.INDEXED.value, State.PRIVATE.value)
    assert search_conversations(store, "quarterly", visible_states=visible) == []


def test_indexed_is_searchable(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "Q4 numbers planning",
         "messages": [("user", "let's plan Q4 quarterly numbers", 1.0),
                      ("assistant", "ok", 2.0),
                      ("user", "add details", 3.0),
                      ("assistant", "done", 4.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value)
    chunk_conversation(store, cid)
    hits = search_conversations(store, "quarterly", visible_states=(State.INDEXED.value,))
    assert hits and any(h.conversation_id == cid for h in hits)


def test_private_excluded_from_mcp_visible(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "Therapy notes",
         "messages": [("user", "I feel anxious about work", 1.0),
                      ("assistant", "Tell me more", 2.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.PRIVATE.value)
    chunk_conversation(store, cid)
    # MCP-visible search must NOT see this.
    assert search_conversations(store, "anxious", visible_states=tuple(MCP_VISIBLE_STATES)) == []
    # CLI-style search with --include-private should see it.
    hits = search_conversations(
        store, "anxious", visible_states=(State.INDEXED.value, State.PRIVATE.value)
    )
    assert any(h.conversation_id == cid for h in hits)


def test_quarantined_never_appears_in_search(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "Sensitive raw notes",
         "messages": [("user", "very personal payload", 1.0),
                      ("assistant", "ack", 2.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.QUARANTINED.value)
    # Quarantined: even with all visible states EXCEPT quarantined we get nothing.
    visible = (State.INDEXED.value, State.PRIVATE.value)
    assert search_conversations(store, "personal", visible_states=visible) == []
    assert search_chunks(store, "personal", visible_states=visible) == []
