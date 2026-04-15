"""State transitions and deletion cascade.

These are the operations that move conversations between visibility states,
plus the hard-delete cascade. They are the trust core of the system; if these
are wrong, privacy is wrong.
"""

from __future__ import annotations

from pathlib import Path

from .models import State
from .vault import Vault
from ..store import Store, transaction, delete_normalized


# Allowed state transitions. A small whitelist is easier to audit than letting
# any state move to any other state.
_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    State.PENDING_REVIEW.value: {
        State.INDEXED.value,
        State.PRIVATE.value,
        State.QUARANTINED.value,
        State.DELETED.value,
    },
    State.INDEXED.value: {
        State.PRIVATE.value,
        State.QUARANTINED.value,
        State.PENDING_REVIEW.value,  # allow re-review
        State.DELETED.value,
    },
    State.PRIVATE.value: {
        State.INDEXED.value,
        State.QUARANTINED.value,
        State.PENDING_REVIEW.value,
        State.DELETED.value,
    },
    State.QUARANTINED.value: {
        State.PENDING_REVIEW.value,
        State.PRIVATE.value,
        State.INDEXED.value,
        State.DELETED.value,
    },
}


def transition_state(store: Store, conversation_id: str, target: str) -> str:
    """Move a conversation's state. Returns the new state.

    Does NOT physically delete; for ``deleted`` use :func:`hard_delete`.
    """
    if target == State.DELETED.value:
        raise ValueError("Use hard_delete() to physically remove a conversation.")
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise KeyError(f"Unknown conversation: {conversation_id}")
    if target == conv.state:
        return target
    allowed = _ALLOWED_TRANSITIONS.get(conv.state, set())
    if target not in allowed:
        raise ValueError(
            f"Disallowed transition: {conv.state} -> {target}"
        )
    with transaction(store):
        store.set_conversation_state(conversation_id, target)
        # Quarantined material has no chunks/extractions/FTS rows.
        if target == State.QUARANTINED.value:
            _strip_derivatives(store, conversation_id)
        else:
            store.reindex_conversation_fts(conversation_id)
    return target


def hard_delete(vault: Vault, store: Store, conversation_id: str) -> dict:
    """Physically remove a conversation and everything derived from it.

    Returns a dict report describing what was removed.
    """
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise KeyError(f"Unknown conversation: {conversation_id}")

    with transaction(store):
        msg_count = store.conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()["c"]
        chunk_count = store.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()["c"]
        prov_count = store.conn.execute(
            "SELECT COUNT(*) AS c FROM provenance_links WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()["c"]
        # Wipe FTS rows tied to this conversation first.
        _wipe_fts_rows(store, conversation_id)
        # Cascade through FK to remove messages, chunks, provenance.
        store.delete_conversation(conversation_id)
        # After cascade, prune any derived objects whose only provenance was
        # this conversation.
        orphans = store.delete_orphan_derived_objects()

    # Remove the normalized file from disk.
    removed_normalized = delete_normalized(vault, conversation_id)

    # Best-effort vacuum so freed pages don't linger on disk.
    try:
        store.vacuum()
    except Exception:
        pass

    return {
        "conversation_id": conversation_id,
        "title": conv.title,
        "messages_deleted": msg_count,
        "chunks_deleted": chunk_count,
        "provenance_links_deleted": prov_count,
        "orphan_derived_objects_deleted": orphans,
        "normalized_file_removed": removed_normalized,
    }


def _wipe_fts_rows(store: Store, conversation_id: str) -> None:
    cv_row = store.conn.execute(
        "SELECT rowid FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    if cv_row:
        store.conn.execute("DELETE FROM fts_conversations WHERE rowid = ?", (cv_row["rowid"],))
    msg_rows = store.conn.execute(
        "SELECT rowid FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchall()
    for r in msg_rows:
        store.conn.execute("DELETE FROM fts_messages WHERE rowid = ?", (r["rowid"],))
    chunk_rows = store.conn.execute(
        "SELECT rowid FROM chunks WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchall()
    for r in chunk_rows:
        store.conn.execute("DELETE FROM fts_chunks WHERE rowid = ?", (r["rowid"],))


def _strip_derivatives(store: Store, conversation_id: str) -> None:
    """Remove chunks, FTS rows, and provenance links for a conversation.

    Used when transitioning into ``quarantined`` - we keep the normalized
    record but delete every derivative.
    """
    _wipe_fts_rows(store, conversation_id)
    store.conn.execute(
        "DELETE FROM provenance_links WHERE conversation_id = ?",
        (conversation_id,),
    )
    store.conn.execute(
        "DELETE FROM chunks WHERE conversation_id = ?",
        (conversation_id,),
    )
    store.delete_orphan_derived_objects()
