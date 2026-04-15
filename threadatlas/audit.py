"""Operator audit and deletion-planning helpers.

This module is read-only except for :func:`hard_delete` - it is the trust
surface that answers "what is stored?", "why is this object here?", and
"what will delete remove?".

All functions return plain dicts/lists suitable for JSON output.
"""

from __future__ import annotations

from .core.models import MCP_VISIBLE_STATES
from .core.vault import Vault
from .store import Store


def audit_conversation(vault: Vault, store: Store, conversation_id: str) -> dict | None:
    """Full, operator-facing dump of what is stored for one conversation."""
    c = store.get_conversation(conversation_id)
    if c is None:
        return None
    msgs = store.list_messages(conversation_id)
    chunks = store.list_chunks(conversation_id)
    prov = store.list_provenance_for_conversation(conversation_id)

    # Which derived objects does this conversation contribute to?
    contributing = store.conn.execute(
        """
        SELECT DISTINCT o.object_id, o.kind, o.title
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
         WHERE p.conversation_id = ?
         ORDER BY o.kind, o.title
        """,
        (conversation_id,),
    ).fetchall()

    normalized_path = vault.normalized_path_for(conversation_id)

    group_memberships = store.list_group_memberships_for_conversation(conversation_id)

    return {
        "conversation_id": c.conversation_id,
        "source": c.source,
        "source_conversation_id": c.source_conversation_id,
        "title": c.title,
        "state": c.state,
        "mcp_visible": c.state in MCP_VISIBLE_STATES,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
        "imported_at": c.imported_at,
        "summary_short": c.summary_short,
        "manual_tags": c.manual_tags,
        "auto_tags": c.auto_tags,
        "importance_score": c.importance_score,
        "resurfacing_score": c.resurfacing_score,
        "has_open_loops": c.has_open_loops,
        "group_memberships": [
            {
                "group_id": m["group_id"],
                "level": m["level"],
                "keyword_label": m["keyword_label"],
                "llm_label": m["llm_label"],
            }
            for m in group_memberships
        ],
        "counts": {
            "messages": len(msgs),
            "chunks": len(chunks),
            "provenance_links": len(prov),
            "contributed_derived_objects": len(contributing),
        },
        "chunks": [
            {
                "chunk_id": ch.chunk_id,
                "chunk_index": ch.chunk_index,
                "chunk_title": ch.chunk_title,
                "start_message_ordinal": ch.start_message_ordinal,
                "end_message_ordinal": ch.end_message_ordinal,
                "summary_short": ch.summary_short,
            }
            for ch in chunks
        ],
        "contributed_derived_objects": [dict(r) for r in contributing],
        "provenance_sample": [
            {
                "object_id": p.object_id,
                "chunk_id": p.chunk_id,
                "excerpt": p.excerpt,
            }
            for p in prov[:20]
        ],
        "normalized_file_present": normalized_path.exists(),
        "normalized_file_path": str(normalized_path),
        "raw_imports_dir": str(vault.raw_imports),
    }


def audit_object(store: Store, object_id: str) -> dict | None:
    """Dump a derived object and every provenance link with excerpts + source titles."""
    obj = store.get_derived_object(object_id)
    if obj is None:
        return None
    rows = store.conn.execute(
        """
        SELECT p.link_id, p.conversation_id, p.chunk_id, p.excerpt, p.created_at,
               c.title AS conv_title, c.state AS conv_state, c.source AS conv_source
          FROM provenance_links p
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE p.object_id = ?
         ORDER BY c.created_at
        """,
        (object_id,),
    ).fetchall()
    return {
        "object_id": obj.object_id,
        "kind": obj.kind,
        "title": obj.title,
        "description": obj.description,
        "state": obj.state,
        "canonical_key": obj.canonical_key,
        "created_at": obj.created_at,
        "updated_at": obj.updated_at,
        "provenance_count": len(rows),
        "distinct_conversations": len({r["conversation_id"] for r in rows}),
        "provenance": [dict(r) for r in rows],
    }


def plan_hard_delete(vault: Vault, store: Store, conversation_id: str) -> dict | None:
    """Preview what :func:`threadatlas.core.workflow.hard_delete` would remove.

    Read-only. Produces the same shape as the delete report plus a list of
    derived objects that would become orphans and a list that would survive
    (with reduced provenance).
    """
    c = store.get_conversation(conversation_id)
    if c is None:
        return None
    msg_count = store.conn.execute(
        "SELECT COUNT(*) AS c FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()["c"]
    chunk_count = store.conn.execute(
        "SELECT COUNT(*) AS c FROM chunks WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()["c"]
    prov_rows = store.conn.execute(
        """
        SELECT p.object_id, o.kind, o.title
          FROM provenance_links p
          JOIN derived_objects o ON o.object_id = p.object_id
         WHERE p.conversation_id = ?
        """,
        (conversation_id,),
    ).fetchall()
    # For each object referenced by this conversation, count all its provenance
    # links. If they are all tied to this conversation, it would become an
    # orphan after delete.
    would_orphan: list[dict] = []
    would_survive: list[dict] = []
    seen = set()
    for pr in prov_rows:
        oid = pr["object_id"]
        if oid in seen:
            continue
        seen.add(oid)
        total_links = store.conn.execute(
            "SELECT COUNT(*) AS c FROM provenance_links WHERE object_id = ?",
            (oid,),
        ).fetchone()["c"]
        other_conv_links = store.conn.execute(
            "SELECT COUNT(*) AS c FROM provenance_links WHERE object_id = ? AND conversation_id != ?",
            (oid, conversation_id),
        ).fetchone()["c"]
        entry = {
            "object_id": oid,
            "kind": pr["kind"],
            "title": pr["title"],
            "total_provenance_links": total_links,
            "links_from_other_conversations": other_conv_links,
        }
        if other_conv_links == 0:
            would_orphan.append(entry)
        else:
            would_survive.append(entry)

    normalized_path = vault.normalized_path_for(conversation_id)

    return {
        "conversation_id": c.conversation_id,
        "title": c.title,
        "state": c.state,
        "would_remove": {
            "messages": msg_count,
            "chunks": chunk_count,
            "provenance_links": len(prov_rows),
            "orphan_derived_objects": len(would_orphan),
            "normalized_file": str(normalized_path) if normalized_path.exists() else None,
        },
        "would_orphan_objects": would_orphan,
        "would_survive_objects": would_survive,
    }
