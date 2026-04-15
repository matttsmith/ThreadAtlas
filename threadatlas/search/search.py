"""Keyword search + filtered listing.

We always filter by visibility state. Callers must pass the set of states
they consider "visible". The MCP layer always passes ``MCP_VISIBLE_STATES``
(indexed only); the CLI passes a wider set when the user is searching their
own private material.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable

from ..core.models import DerivedKind, MCP_VISIBLE_STATES
from ..store import Store


# FTS5 reserves a small set of operator characters; we sanitize user input.
_FTS_SAFE_RX = re.compile(r"[^\w\s\-]")


def _sanitize_query(q: str) -> str:
    cleaned = _FTS_SAFE_RX.sub(" ", q or "").strip()
    return cleaned


def _state_in_clause(states: Iterable[str]) -> tuple[str, list]:
    s = list(states)
    if not s:
        return "1=0", []
    placeholders = ",".join("?" for _ in s)
    return f"state IN ({placeholders})", s


@dataclass
class SearchHit:
    conversation_id: str
    chunk_id: str | None
    title: str
    snippet: str
    score: float
    state: str
    source: str


def _recency_bonus(updated_at: float | None, now: float) -> float:
    """Small deterministic recency bonus.

    Caps at +0.5 for 'today' and decays to ~0 after ~2 years. Avoids a
    huge recency effect that would drown out content matches.
    """
    if not updated_at:
        return 0.0
    age_days = max(0.0, (now - float(updated_at)) / 86400.0)
    if age_days > 730:
        return 0.0
    # Linear decay from 0.5 (today) to 0 (2y ago). Boring, inspectable.
    return max(0.0, 0.5 * (1.0 - age_days / 730.0))


def _exact_phrase_bonus(query: str, text: str) -> float:
    """+0.75 if the (sanitized) query appears as a contiguous substring."""
    q = (query or "").strip().lower()
    if not q or " " not in q:
        return 0.0
    return 0.75 if q in (text or "").lower() else 0.0


def search_conversations(
    store: Store,
    query: str,
    *,
    visible_states: Iterable[str] = MCP_VISIBLE_STATES,
    limit: int = 25,
) -> list[SearchHit]:
    """Search conversation titles + summaries + tags + message bodies.

    Ranking signals (all deterministic):
      * bm25 lexical score
      * 1.5x title/summary hits
      * +importance_score * 0.05 (capped)
      * +recency bonus (max +0.5)
      * +0.75 exact-phrase bonus (multi-word query appearing verbatim)
    """
    qclean = _sanitize_query(query)
    if not qclean:
        return []

    state_clause, state_params = _state_in_clause(visible_states)
    now = time.time()

    title_rows = store.conn.execute(
        f"""
        SELECT c.conversation_id, c.title, c.summary_short, c.state, c.source,
               c.importance_score, c.updated_at, c.created_at,
               c.manual_tags,
               -bm25(fts_conversations) AS score,
               c.summary_short AS snippet
          FROM fts_conversations
          JOIN conversations c ON c.rowid = fts_conversations.rowid
         WHERE fts_conversations MATCH ? AND c.{state_clause}
         ORDER BY score DESC
         LIMIT ?
        """,
        [qclean, *state_params, limit],
    ).fetchall()

    msg_rows = store.conn.execute(
        f"""
        SELECT c.conversation_id, c.title, c.state, c.source,
               c.importance_score, c.updated_at, c.created_at,
               c.manual_tags,
               -bm25(fts_messages) AS score,
               snippet(fts_messages, 0, '[', ']', '...', 16) AS snippet
          FROM fts_messages
          JOIN messages m ON m.rowid = fts_messages.rowid
          JOIN conversations c ON c.conversation_id = m.conversation_id
         WHERE fts_messages MATCH ? AND c.{state_clause}
         ORDER BY score DESC
         LIMIT ?
        """,
        [qclean, *state_params, limit * 4],
    ).fetchall()

    def _score_row(r, base_boost: float) -> float:
        base = float(r["score"]) * base_boost
        imp = float(r["importance_score"] or 0.0)
        rec = _recency_bonus(r["updated_at"] or r["created_at"], now)
        phrase = _exact_phrase_bonus(query, (r["title"] or "") + " " + (r["snippet"] or ""))
        return base + min(imp * 0.05, 0.5) + rec + phrase

    by_conv: dict[str, SearchHit] = {}
    for r in title_rows:
        by_conv[r["conversation_id"]] = SearchHit(
            conversation_id=r["conversation_id"],
            chunk_id=None,
            title=r["title"],
            snippet=(r["snippet"] or r["title"])[:240],
            score=_score_row(r, 1.5),
            state=r["state"],
            source=r["source"],
        )
    for r in msg_rows:
        cid = r["conversation_id"]
        new_score = _score_row(r, 1.0)
        prev = by_conv.get(cid)
        if prev is None or new_score > prev.score:
            by_conv[cid] = SearchHit(
                conversation_id=cid,
                chunk_id=None,
                title=r["title"],
                snippet=r["snippet"] or "",
                score=new_score,
                state=r["state"],
                source=r["source"],
            )
    hits = sorted(by_conv.values(), key=lambda h: h.score, reverse=True)[:limit]
    return hits


def search_chunks(
    store: Store,
    query: str,
    *,
    visible_states: Iterable[str] = MCP_VISIBLE_STATES,
    limit: int = 25,
) -> list[SearchHit]:
    qclean = _sanitize_query(query)
    if not qclean:
        return []
    state_clause, state_params = _state_in_clause(visible_states)
    rows = store.conn.execute(
        f"""
        SELECT c.conversation_id, ch.chunk_id, ch.chunk_title, c.title AS conv_title,
               c.state, c.source,
               -bm25(fts_chunks) AS score,
               snippet(fts_chunks, 2, '[', ']', '...', 16) AS snippet
          FROM fts_chunks
          JOIN chunks ch ON ch.rowid = fts_chunks.rowid
          JOIN conversations c ON c.conversation_id = ch.conversation_id
         WHERE fts_chunks MATCH ? AND c.{state_clause}
         ORDER BY score DESC
         LIMIT ?
        """,
        [qclean, *state_params, limit],
    ).fetchall()
    return [
        SearchHit(
            conversation_id=r["conversation_id"],
            chunk_id=r["chunk_id"],
            title=f"{r['conv_title']} :: {r['chunk_title']}",
            snippet=r["snippet"] or "",
            score=float(r["score"]),
            state=r["state"],
            source=r["source"],
        )
        for r in rows
    ]


# --- listings of derived objects --------------------------------------------

def _list_kind(
    store: Store, kind: str, visible_states: Iterable[str], limit: int
) -> list[dict]:
    state_clause, state_params = _state_in_clause(visible_states)
    rows = store.conn.execute(
        f"""
        SELECT DISTINCT o.object_id, o.title, o.description, o.kind,
               o.project_id, o.updated_at
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE o.kind = ? AND o.state = 'active' AND c.{state_clause}
         ORDER BY o.updated_at DESC
         LIMIT ?
        """,
        [kind, *state_params, limit],
    ).fetchall()
    return [dict(r) for r in rows]


def list_open_loops(store: Store, *, visible_states=MCP_VISIBLE_STATES, limit: int = 100):
    return _list_kind(store, DerivedKind.OPEN_LOOP.value, visible_states, limit)


def list_decisions(store: Store, *, visible_states=MCP_VISIBLE_STATES, limit: int = 100):
    return _list_kind(store, DerivedKind.DECISION.value, visible_states, limit)


def list_entities(store: Store, *, visible_states=MCP_VISIBLE_STATES, limit: int = 200):
    return _list_kind(store, DerivedKind.ENTITY.value, visible_states, limit)


def list_projects(store: Store, *, visible_states=MCP_VISIBLE_STATES, limit: int = 200):
    return _list_kind(store, DerivedKind.PROJECT.value, visible_states, limit)


# --- project synthesis ------------------------------------------------------

def project_view(
    store: Store, project_id: str, *, visible_states=MCP_VISIBLE_STATES
) -> dict | None:
    """Build a project page.

    Includes: project metadata, linked conversations, key chunks, linked
    entities, decisions, open loops, artifacts, and a provenance summary.
    Excludes anything not in ``visible_states``.
    """
    proj = store.get_derived_object(project_id)
    if proj is None or proj.kind != DerivedKind.PROJECT.value:
        return None

    state_clause, state_params = _state_in_clause(visible_states)

    # Conversations linked to this project via provenance.
    conv_rows = store.conn.execute(
        f"""
        SELECT DISTINCT c.conversation_id, c.title, c.source, c.state,
               c.created_at, c.updated_at, c.importance_score, c.has_open_loops
          FROM provenance_links p
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE p.object_id = ? AND c.{state_clause}
         ORDER BY COALESCE(c.updated_at, c.created_at) DESC
        """,
        [project_id, *state_params],
    ).fetchall()

    convs = [dict(r) for r in conv_rows]

    # Other derived objects co-occurring in those conversations.
    if convs:
        cv_placeholders = ",".join("?" for _ in convs)
        co_rows = store.conn.execute(
            f"""
            SELECT DISTINCT o.object_id, o.kind, o.title, o.description
              FROM derived_objects o
              JOIN provenance_links p ON p.object_id = o.object_id
             WHERE p.conversation_id IN ({cv_placeholders})
               AND o.state = 'active'
               AND o.object_id != ?
             ORDER BY o.kind, o.title
            """,
            [*[c["conversation_id"] for c in convs], project_id],
        ).fetchall()
    else:
        co_rows = []

    by_kind: dict[str, list[dict]] = {}
    for r in co_rows:
        by_kind.setdefault(r["kind"], []).append(dict(r))

    return {
        "project": {
            "object_id": proj.object_id,
            "title": proj.title,
            "description": proj.description,
        },
        "conversations": convs,
        "decisions": by_kind.get(DerivedKind.DECISION.value, []),
        "open_loops": by_kind.get(DerivedKind.OPEN_LOOP.value, []),
        "entities": by_kind.get(DerivedKind.ENTITY.value, []),
        "artifacts": by_kind.get(DerivedKind.ARTIFACT.value, []),
        "preferences": by_kind.get(DerivedKind.PREFERENCE.value, []),
    }


def project_timeline(
    store: Store, project_id: str, *, visible_states=MCP_VISIBLE_STATES
) -> list[dict]:
    """Chronological timeline of conversations and notable derived objects."""
    state_clause, state_params = _state_in_clause(visible_states)
    rows = store.conn.execute(
        f"""
        SELECT c.conversation_id, c.title, c.source,
               COALESCE(c.created_at, c.imported_at) AS event_at
          FROM provenance_links p
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE p.object_id = ? AND c.{state_clause}
         GROUP BY c.conversation_id
         ORDER BY event_at ASC
        """,
        [project_id, *state_params],
    ).fetchall()
    return [
        {
            "kind": "conversation",
            "conversation_id": r["conversation_id"],
            "title": r["title"],
            "source": r["source"],
            "at": r["event_at"],
        }
        for r in rows
    ]
