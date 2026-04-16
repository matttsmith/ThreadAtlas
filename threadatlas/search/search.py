"""Keyword + semantic search with reciprocal rank fusion.

We always filter by visibility state. Callers must pass the set of states
they consider "visible". The MCP layer always passes ``MCP_VISIBLE_STATES``
(indexed only); the CLI passes a wider set when the user is searching their
own private material.

v2 adds:
- Hybrid semantic + keyword search via reciprocal rank fusion.
- Date-range and register filtering.
- match_type field on search results.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable

from ..core.models import DEFAULT_REGISTER_EXCLUDES, DerivedKind, MCP_VISIBLE_STATES
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
    match_type: str = "keyword"  # keyword | semantic | both


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


def _apply_register_filter(
    store: Store,
    conversation_ids: set[str],
    register: list[str] | None,
    exclude_registers: frozenset[str] | None = None,
) -> set[str]:
    """Filter conversation IDs by register tags from conversation_llm_meta."""
    if not register and not exclude_registers:
        return conversation_ids
    if not conversation_ids:
        return conversation_ids

    filtered = set()
    for cid in conversation_ids:
        row = store.conn.execute(
            "SELECT dominant_register FROM conversation_llm_meta WHERE conversation_id = ?",
            (cid,),
        ).fetchone()
        dom_reg = row["dominant_register"] if row else None

        if register and dom_reg and dom_reg not in register:
            continue
        if exclude_registers and dom_reg and dom_reg in exclude_registers:
            continue
        filtered.add(cid)
    return filtered


def _apply_date_filter(
    store: Store,
    conversation_ids: set[str],
    after: float | None,
    before: float | None,
) -> set[str]:
    """Filter conversation IDs by date range."""
    if not after and not before:
        return conversation_ids
    if not conversation_ids:
        return conversation_ids

    filtered = set()
    for cid in conversation_ids:
        conv = store.get_conversation(cid)
        if conv is None:
            continue
        ts = conv.updated_at or conv.created_at or 0.0
        if after and ts < after:
            continue
        if before and ts > before:
            continue
        filtered.add(cid)
    return filtered


def search_conversations(
    store: Store,
    query: str,
    *,
    visible_states: Iterable[str] = MCP_VISIBLE_STATES,
    limit: int = 25,
    after: float | None = None,
    before: float | None = None,
    register: list[str] | None = None,
    source_filter: str | None = None,
) -> list[SearchHit]:
    """Hybrid keyword + semantic search over conversations.

    Ranking signals:
      * BM25 lexical score (keyword path)
      * Semantic similarity via embeddings (semantic path)
      * Reciprocal rank fusion to merge both paths
      * +importance_score * 0.05 (capped)
      * +recency bonus (max +0.5)
      * +0.75 exact-phrase bonus (multi-word query appearing verbatim)

    Returns results with ``match_type`` indicating how they were found.
    """
    qclean = _sanitize_query(query)
    if not qclean:
        return []

    state_clause, state_params = _state_in_clause(visible_states)
    now = time.time()

    # --- Keyword path (BM25) ---
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

    keyword_hits: dict[str, SearchHit] = {}
    for r in title_rows:
        keyword_hits[r["conversation_id"]] = SearchHit(
            conversation_id=r["conversation_id"],
            chunk_id=None,
            title=r["title"],
            snippet=(r["snippet"] or r["title"])[:240],
            score=_score_row(r, 1.5),
            state=r["state"],
            source=r["source"],
            match_type="keyword",
        )
    for r in msg_rows:
        cid = r["conversation_id"]
        new_score = _score_row(r, 1.0)
        prev = keyword_hits.get(cid)
        if prev is None or new_score > prev.score:
            keyword_hits[cid] = SearchHit(
                conversation_id=cid,
                chunk_id=None,
                title=r["title"],
                snippet=r["snippet"] or "",
                score=new_score,
                state=r["state"],
                source=r["source"],
                match_type="keyword",
            )

    # --- Semantic path (embedding similarity) ---
    semantic_hits: dict[str, SearchHit] = {}
    try:
        from .embeddings import (
            TFIDFEmbedder,
            bytes_to_embedding,
            cosine_similarity,
            fit_embedder_from_corpus,
        )
        embedder = fit_embedder_from_corpus(store)
        query_vec = embedder.embed(query)

        vs_tuple = tuple(visible_states)
        all_embeddings = store.get_all_chunk_embeddings(visible_states=vs_tuple)

        if all_embeddings and any(v != 0.0 for v in query_vec):
            scored_chunks: list[tuple[str, str, float]] = []
            for chunk_id, conv_id, emb_bytes in all_embeddings:
                emb_vec = bytes_to_embedding(emb_bytes)
                sim = cosine_similarity(query_vec, emb_vec)
                scored_chunks.append((chunk_id, conv_id, sim))

            scored_chunks.sort(key=lambda x: x[2], reverse=True)
            for chunk_id, conv_id, sim in scored_chunks[:limit * 2]:
                if sim < 0.05:
                    continue
                if conv_id not in semantic_hits or sim > semantic_hits[conv_id].score:
                    conv = store.get_conversation(conv_id)
                    if conv is None:
                        continue
                    semantic_hits[conv_id] = SearchHit(
                        conversation_id=conv_id,
                        chunk_id=chunk_id,
                        title=conv.title,
                        snippet=conv.summary_short[:240] if conv.summary_short else conv.title,
                        score=sim,
                        state=conv.state,
                        source=conv.source,
                        match_type="semantic",
                    )
    except Exception:
        pass  # Embeddings not available; keyword-only.

    # --- Reciprocal rank fusion ---
    if semantic_hits:
        from .embeddings import reciprocal_rank_fusion

        kw_ranked = sorted(keyword_hits.items(), key=lambda x: x[1].score, reverse=True)
        sem_ranked = sorted(semantic_hits.items(), key=lambda x: x[1].score, reverse=True)

        kw_list = [(cid, h.score) for cid, h in kw_ranked]
        sem_list = [(cid, h.score) for cid, h in sem_ranked]

        fused = reciprocal_rank_fusion(kw_list, sem_list)

        # Build merged hit list.
        all_hits_map = {}
        for cid, hit in keyword_hits.items():
            all_hits_map[cid] = hit
        for cid, hit in semantic_hits.items():
            if cid in all_hits_map:
                all_hits_map[cid].match_type = "both"
            else:
                all_hits_map[cid] = hit

        by_conv = {}
        for cid, fused_score in fused:
            if cid in all_hits_map:
                hit = all_hits_map[cid]
                hit.score = fused_score
                by_conv[cid] = hit
    else:
        by_conv = keyword_hits

    # Apply filters.
    result_cids = set(by_conv.keys())
    if register is not None:
        result_cids = _apply_register_filter(store, result_cids, register)
    if after or before:
        result_cids = _apply_date_filter(store, result_cids, after, before)
    if source_filter:
        result_cids = {cid for cid in result_cids
                       if by_conv[cid].source.lower() == source_filter.lower()}

    hits = [by_conv[cid] for cid in result_cids if cid in by_conv]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def search_chunks(
    store: Store,
    query: str,
    *,
    visible_states: Iterable[str] = MCP_VISIBLE_STATES,
    limit: int = 25,
    after: float | None = None,
    before: float | None = None,
    register: list[str] | None = None,
    source_filter: str | None = None,
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
        [qclean, *state_params, limit * 2],
    ).fetchall()
    hits = [
        SearchHit(
            conversation_id=r["conversation_id"],
            chunk_id=r["chunk_id"],
            title=f"{r['conv_title']} :: {r['chunk_title']}",
            snippet=r["snippet"] or "",
            score=float(r["score"]),
            state=r["state"],
            source=r["source"],
            match_type="keyword",
        )
        for r in rows
    ]

    # Apply filters.
    if register is not None:
        valid_cids = _apply_register_filter(
            store, {h.conversation_id for h in hits}, register)
        hits = [h for h in hits if h.conversation_id in valid_cids]
    if after or before:
        valid_cids = _apply_date_filter(
            store, {h.conversation_id for h in hits}, after, before)
        hits = [h for h in hits if h.conversation_id in valid_cids]
    if source_filter:
        hits = [h for h in hits if h.source.lower() == source_filter.lower()]

    return hits[:limit]


# --- listings of derived objects --------------------------------------------

def _list_kind(
    store: Store,
    kind: str,
    visible_states: Iterable[str],
    limit: int,
    *,
    after: float | None = None,
    before: float | None = None,
    register: list[str] | None = None,
    source_filter: str | None = None,
) -> list[dict]:
    state_clause, state_params = _state_in_clause(visible_states)

    # Build extra join + filter conditions for register and date.
    extra_joins = ""
    extra_conditions = ""
    extra_params: list = []

    if register is not None:
        extra_conditions += " AND o.source_register IN ({})".format(
            ",".join("?" for _ in register))
        extra_params.extend(register)
    elif kind in (DerivedKind.DECISION.value, DerivedKind.OPEN_LOOP.value, DerivedKind.PROJECT.value):
        # Default: exclude roleplay and jailbreak from substantive types.
        excl = list(DEFAULT_REGISTER_EXCLUDES)
        if excl:
            extra_conditions += " AND (o.source_register IS NULL OR o.source_register NOT IN ({}))".format(
                ",".join("?" for _ in excl))
            extra_params.extend(excl)

    if after:
        extra_conditions += " AND COALESCE(c.updated_at, c.created_at) >= ?"
        extra_params.append(after)
    if before:
        extra_conditions += " AND COALESCE(c.updated_at, c.created_at) <= ?"
        extra_params.append(before)
    if source_filter:
        extra_conditions += " AND c.source = ?"
        extra_params.append(source_filter)

    rows = store.conn.execute(
        f"""
        SELECT DISTINCT o.object_id, o.title, o.description, o.kind,
               o.project_id, o.updated_at, o.entity_type, o.source_register,
               o.source_reality_mode, o.paraphrase, o.status
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE o.kind = ? AND o.state = 'active' AND c.{state_clause}
               {extra_conditions}
         ORDER BY o.updated_at DESC
         LIMIT ?
        """,
        [kind, *state_params, *extra_params, limit],
    ).fetchall()
    return [dict(r) for r in rows]


def list_open_loops(store: Store, *, visible_states=MCP_VISIBLE_STATES, limit: int = 100,
                    after=None, before=None, register=None, source_filter=None):
    return _list_kind(store, DerivedKind.OPEN_LOOP.value, visible_states, limit,
                      after=after, before=before, register=register, source_filter=source_filter)


def list_decisions(store: Store, *, visible_states=MCP_VISIBLE_STATES, limit: int = 100,
                   after=None, before=None, register=None, source_filter=None):
    return _list_kind(store, DerivedKind.DECISION.value, visible_states, limit,
                      after=after, before=before, register=register, source_filter=source_filter)


def list_entities(store: Store, *, visible_states=MCP_VISIBLE_STATES, limit: int = 200,
                  after=None, before=None, register=None, source_filter=None):
    return _list_kind(store, DerivedKind.ENTITY.value, visible_states, limit,
                      after=after, before=before, register=register, source_filter=source_filter)


def list_projects(store: Store, *, visible_states=MCP_VISIBLE_STATES, limit: int = 200,
                  after=None, before=None, register=None, source_filter=None):
    return _list_kind(store, DerivedKind.PROJECT.value, visible_states, limit,
                      after=after, before=before, register=register, source_filter=source_filter)


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
