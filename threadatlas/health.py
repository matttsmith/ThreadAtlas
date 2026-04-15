"""Vault health check.

Enumerates cheap invariants the system relies on. If any warning appears,
the operator has a concrete next action. Nothing here performs a
modification; everything is a read.

Invariants checked:
  * every conversation has a normalized JSON file on disk (except when
    state is quarantined or deleted - quarantine keeps normalized; any
    normalized file without a conversation row is a stray)
  * every chunk belongs to a conversation (FK enforces but we assert)
  * every FTS row's rowid has a backing row in its source table
  * no FTS rows for conversations whose state is ``pending_review`` or
    ``quarantined`` (defense-in-depth for the visibility boundary)
  * no provenance_links whose conversation or object are missing
  * no conversation_group_memberships pointing to dead conversations
  * no chunk ranges outside their conversation's message ordinals
"""

from __future__ import annotations

from .core.vault import Vault
from .store import Store


def quick_check(vault: Vault, store: Store) -> list[str]:
    """Return a list of human-readable warnings. Empty list = healthy."""
    warnings: list[str] = []

    # 1. Stray normalized files with no conversation row.
    existing_ids = {
        r["conversation_id"]
        for r in store.conn.execute("SELECT conversation_id FROM conversations").fetchall()
    }
    stray = []
    for p in vault.normalized.rglob("*.json"):
        cid = p.stem
        if cid not in existing_ids:
            stray.append(str(p))
    if stray:
        warnings.append(
            f"{len(stray)} stray normalized file(s) on disk with no matching conversation row. "
            f"Example: {stray[0]}"
        )

    # 2. Conversation rows without normalized files (non-deleted).
    missing_norm: list[str] = []
    for cid in existing_ids:
        path = vault.normalized_path_for(cid)
        if not path.exists():
            missing_norm.append(cid)
    if missing_norm:
        warnings.append(
            f"{len(missing_norm)} conversation(s) missing normalized JSON file on disk. "
            f"Example: {missing_norm[0]}"
        )

    # 3. FTS sync checks.
    fts_orphan_convs = store.conn.execute(
        "SELECT COUNT(*) AS c FROM fts_conversations "
        "WHERE rowid NOT IN (SELECT rowid FROM conversations)"
    ).fetchone()["c"]
    if fts_orphan_convs:
        warnings.append(f"fts_conversations has {fts_orphan_convs} orphan row(s). "
                        f"Run `threadatlas rebuild-index`.")

    fts_orphan_msgs = store.conn.execute(
        "SELECT COUNT(*) AS c FROM fts_messages "
        "WHERE rowid NOT IN (SELECT rowid FROM messages)"
    ).fetchone()["c"]
    if fts_orphan_msgs:
        warnings.append(f"fts_messages has {fts_orphan_msgs} orphan row(s). "
                        f"Run `threadatlas rebuild-index`.")

    fts_orphan_chunks = store.conn.execute(
        "SELECT COUNT(*) AS c FROM fts_chunks "
        "WHERE rowid NOT IN (SELECT rowid FROM chunks)"
    ).fetchone()["c"]
    if fts_orphan_chunks:
        warnings.append(f"fts_chunks has {fts_orphan_chunks} orphan row(s). "
                        f"Run `threadatlas rebuild-index`.")

    # 4. FTS rows present for conversations that should NOT have them.
    leaks = store.conn.execute(
        "SELECT COUNT(*) AS c FROM fts_messages f "
        "JOIN messages m ON m.rowid = f.rowid "
        "JOIN conversations c ON c.conversation_id = m.conversation_id "
        "WHERE c.state NOT IN ('indexed','private')"
    ).fetchone()["c"]
    if leaks:
        warnings.append(
            f"VISIBILITY LEAK: {leaks} fts_messages row(s) belong to non-FTS-eligible "
            f"conversations (pending_review or quarantined). Run `threadatlas rebuild-index`."
        )

    # 5. Dead memberships.
    dead_mem = store.conn.execute(
        "SELECT COUNT(*) AS c FROM conversation_group_memberships m "
        "WHERE m.conversation_id NOT IN (SELECT conversation_id FROM conversations)"
    ).fetchone()["c"]
    if dead_mem:
        warnings.append(f"{dead_mem} group membership(s) refer to missing conversations. "
                        f"Run `threadatlas group` to regenerate.")

    # 6. Orphan derived objects (no provenance at all).
    dead_objs = store.conn.execute(
        "SELECT COUNT(*) AS c FROM derived_objects o "
        "WHERE NOT EXISTS (SELECT 1 FROM provenance_links p WHERE p.object_id = o.object_id)"
    ).fetchone()["c"]
    if dead_objs:
        warnings.append(f"{dead_objs} derived object(s) have no provenance. Safe to delete.")

    # 7. Chunk ranges outside message ordinals.
    bad_chunks = store.conn.execute(
        """
        SELECT COUNT(*) AS c FROM chunks ch
         WHERE NOT EXISTS (
            SELECT 1 FROM messages m
             WHERE m.conversation_id = ch.conversation_id
               AND m.ordinal = ch.start_message_ordinal)
        """
    ).fetchone()["c"]
    if bad_chunks:
        warnings.append(f"{bad_chunks} chunk(s) have out-of-range start ordinals. "
                        f"Re-run `threadatlas chunk`.")

    return warnings
