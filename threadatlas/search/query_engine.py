"""Structured query engine for ThreadAtlas.

Provides a single entry-point that accepts a query string with optional
filter prefixes, dispatches across conversations, chunks, and derived
objects, and returns a unified ranked result set.

Supported filter prefixes (case-insensitive)::

    source:chatgpt          — restrict to a conversation source
    tag:productivity        — require a manual or auto tag
    kind:decision           — search only derived objects of this kind
    project:<id>            — restrict to a project id
    after:2024-01-01        — conversations created/updated after date
    before:2024-12-31       — conversations created/updated before date
    has:open_loops           — conversations with open loops
    has:chunks               — conversations with chunks

Anything not matching a prefix is treated as free-text and fed to FTS.

Examples::

    "migration plan source:chatgpt"
    "kind:decision after:2024-06-01"
    "tag:architecture project:proj_abc123"
    "kubernetes deployment"
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Iterable

from ..core.models import DerivedKind, MCP_VISIBLE_STATES
from ..store import Store
from .search import (
    SearchHit,
    _sanitize_query,
    _state_in_clause,
    _recency_bonus,
    _exact_phrase_bonus,
    search_conversations,
    search_chunks,
)


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

# Matches key:value or key:"quoted value" tokens.
_PREFIX_RE = re.compile(
    r"""
    (?P<key>[a-z_]+)            # prefix key
    :                           # separator
    (?:
        "(?P<qval>[^"]*)"      # quoted value
      | (?P<val>\S+)           # unquoted value
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_KNOWN_PREFIXES = frozenset({
    "source", "tag", "kind", "project", "after", "before", "has",
})

_VALID_KINDS = frozenset(k.value for k in DerivedKind)

_HAS_FLAGS = frozenset({"open_loops", "chunks"})


def _parse_date(s: str) -> float | None:
    """Parse YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS to POSIX seconds."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return None


@dataclass
class QueryFilter:
    """Parsed filter predicates extracted from the query string."""

    sources: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    after: float | None = None
    before: float | None = None
    has_flags: list[str] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop empty/None values for a compact representation.
        return {k: v for k, v in d.items() if v}


def parse_query(raw: str) -> QueryFilter:
    """Parse a raw query string into structured filters + free text."""
    filt = QueryFilter()
    remaining_parts: list[str] = []
    pos = 0
    raw = (raw or "").strip()

    for m in _PREFIX_RE.finditer(raw):
        # Collect text between the previous match end and this match start.
        gap = raw[pos:m.start()].strip()
        if gap:
            remaining_parts.append(gap)
        pos = m.end()

        key = m.group("key").lower()
        val = m.group("qval") if m.group("qval") is not None else m.group("val")

        if key not in _KNOWN_PREFIXES:
            # Unknown prefix — treat the whole token as text.
            remaining_parts.append(m.group(0))
            continue

        if key == "source":
            filt.sources.append(val.lower())
        elif key == "tag":
            filt.tags.append(val)
        elif key == "kind":
            v = val.lower()
            if v in _VALID_KINDS:
                filt.kinds.append(v)
        elif key == "project":
            filt.projects.append(val)
        elif key == "after":
            ts = _parse_date(val)
            if ts is not None:
                filt.after = ts
        elif key == "before":
            ts = _parse_date(val)
            if ts is not None:
                filt.before = ts
        elif key == "has":
            v = val.lower()
            if v in _HAS_FLAGS:
                filt.has_flags.append(v)

    # Trailing text after the last prefix match.
    trailing = raw[pos:].strip()
    if trailing:
        remaining_parts.append(trailing)

    filt.text = " ".join(remaining_parts).strip()
    return filt


# ---------------------------------------------------------------------------
# Unified hit type
# ---------------------------------------------------------------------------

@dataclass
class QueryHit:
    """A single result from the query engine."""

    hit_type: str       # "conversation", "chunk", "derived_object"
    id: str             # conversation_id, chunk_id, or object_id
    title: str
    snippet: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryResult:
    """Complete result set from a query engine run."""

    raw_query: str
    filters: dict
    hits: list[QueryHit]
    total_by_type: dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Filter application (post-FTS SQL or Python)
# ---------------------------------------------------------------------------

def _apply_conversation_filters(
    store: Store,
    hits: list[SearchHit],
    filt: QueryFilter,
) -> list[SearchHit]:
    """Post-filter search hits by source, tag, date, has-flags."""
    if not (filt.sources or filt.tags or filt.after or filt.before or filt.has_flags):
        return hits

    out = []
    for h in hits:
        conv = store.get_conversation(h.conversation_id)
        if conv is None:
            continue
        if filt.sources and conv.source.lower() not in filt.sources:
            continue
        if filt.tags:
            all_tags = set(t.lower() for t in (conv.manual_tags or []) + (conv.auto_tags or []))
            if not any(t.lower() in all_tags for t in filt.tags):
                continue
        ts = conv.updated_at or conv.created_at or 0.0
        if filt.after and ts < filt.after:
            continue
        if filt.before and ts > filt.before:
            continue
        if "open_loops" in filt.has_flags and not conv.has_open_loops:
            continue
        if "chunks" in filt.has_flags:
            chunk_count = store.conn.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE conversation_id = ?",
                (conv.conversation_id,),
            ).fetchone()["c"]
            if chunk_count == 0:
                continue
        out.append(h)
    return out


def _apply_project_filter_to_conversations(
    store: Store,
    filt: QueryFilter,
    visible_states: Iterable[str],
) -> set[str] | None:
    """If project filters are set, return the set of conversation IDs linked
    to those projects. Returns None if no project filter is active."""
    if not filt.projects:
        return None
    state_clause, state_params = _state_in_clause(visible_states)
    cids: set[str] = set()
    for pid in filt.projects:
        rows = store.conn.execute(
            f"""
            SELECT DISTINCT p.conversation_id
              FROM provenance_links p
              JOIN conversations c ON c.conversation_id = p.conversation_id
             WHERE p.object_id = ? AND c.{state_clause}
            """,
            [pid, *state_params],
        ).fetchall()
        for r in rows:
            cids.add(r["conversation_id"])
    return cids


# ---------------------------------------------------------------------------
# Derived object search (LIKE-based, no FTS)
# ---------------------------------------------------------------------------

def _search_derived_objects(
    store: Store,
    text: str,
    *,
    kinds: list[str] | None = None,
    visible_states: Iterable[str] = MCP_VISIBLE_STATES,
    limit: int = 50,
) -> list[QueryHit]:
    """Search derived objects by title/description using LIKE.

    Only returns objects that have at least one provenance link to a
    conversation in ``visible_states``.
    """
    state_clause, state_params = _state_in_clause(visible_states)

    conditions = ["o.state = 'active'"]
    params: list = []

    if text:
        conditions.append("(o.title LIKE ? OR o.description LIKE ?)")
        like_pat = f"%{text}%"
        params.extend([like_pat, like_pat])

    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        conditions.append(f"o.kind IN ({placeholders})")
        params.extend(kinds)

    where = " AND ".join(conditions)
    params.extend(state_params)

    rows = store.conn.execute(
        f"""
        SELECT DISTINCT o.object_id, o.kind, o.title, o.description,
               o.project_id, o.updated_at
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE {where} AND c.{state_clause}
         ORDER BY o.updated_at DESC
         LIMIT ?
        """,
        [*params, limit],
    ).fetchall()

    hits: list[QueryHit] = []
    text_lower = (text or "").lower()
    for r in rows:
        title = r["title"] or ""
        desc = r["description"] or ""
        # Simple relevance: title match > description match.
        score = 0.0
        if text_lower:
            if text_lower in title.lower():
                score = 2.0
            elif text_lower in desc.lower():
                score = 1.0
        else:
            score = 0.5  # listing mode, no text
        hits.append(QueryHit(
            hit_type="derived_object",
            id=r["object_id"],
            title=title,
            snippet=desc[:240] if desc else title,
            score=score,
            metadata={
                "kind": r["kind"],
                "project_id": r["project_id"],
                "updated_at": r["updated_at"],
            },
        ))
    return hits


# ---------------------------------------------------------------------------
# List-only queries (no text, just filters)
# ---------------------------------------------------------------------------

def _list_conversations(
    store: Store,
    filt: QueryFilter,
    *,
    visible_states: Iterable[str],
    limit: int,
) -> list[QueryHit]:
    """List conversations matching filters without a keyword search."""
    vs = list(visible_states)
    state_clause, state_params = _state_in_clause(vs)

    conditions = [f"c.{state_clause}"]
    params: list = list(state_params)

    if filt.sources:
        src_ph = ",".join("?" for _ in filt.sources)
        conditions.append(f"c.source IN ({src_ph})")
        params.extend(filt.sources)
    if filt.after:
        conditions.append("COALESCE(c.updated_at, c.created_at) >= ?")
        params.append(filt.after)
    if filt.before:
        conditions.append("COALESCE(c.updated_at, c.created_at) <= ?")
        params.append(filt.before)
    if "open_loops" in filt.has_flags:
        conditions.append("c.has_open_loops = 1")

    where = " AND ".join(conditions)
    rows = store.conn.execute(
        f"""
        SELECT c.conversation_id, c.title, c.summary_short, c.state,
               c.source, c.manual_tags, c.auto_tags,
               c.importance_score, c.updated_at, c.created_at
          FROM conversations c
         WHERE {where}
         ORDER BY COALESCE(c.updated_at, c.created_at) DESC
         LIMIT ?
        """,
        [*params, limit * 2],  # over-fetch for post-filter
    ).fetchall()

    import json
    hits: list[QueryHit] = []
    now = time.time()
    for r in rows:
        # Post-filter tags (stored as JSON).
        if filt.tags:
            manual = set(t.lower() for t in json.loads(r["manual_tags"] or "[]"))
            auto = set(t.lower() for t in json.loads(r["auto_tags"] or "[]"))
            all_tags = manual | auto
            if not any(t.lower() in all_tags for t in filt.tags):
                continue

        # Post-filter has:chunks.
        if "chunks" in filt.has_flags:
            cc = store.conn.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE conversation_id = ?",
                (r["conversation_id"],),
            ).fetchone()["c"]
            if cc == 0:
                continue

        score = (float(r["importance_score"] or 0.0) * 0.1
                 + _recency_bonus(r["updated_at"] or r["created_at"], now))
        hits.append(QueryHit(
            hit_type="conversation",
            id=r["conversation_id"],
            title=r["title"],
            snippet=(r["summary_short"] or r["title"] or "")[:240],
            score=score,
            metadata={
                "source": r["source"],
                "state": r["state"],
                "updated_at": r["updated_at"],
                "created_at": r["created_at"],
            },
        ))
        if len(hits) >= limit:
            break
    return hits


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def query(
    store: Store,
    raw_query: str,
    *,
    visible_states: Iterable[str] = MCP_VISIBLE_STATES,
    limit: int = 25,
) -> QueryResult:
    """Execute a structured query against the ThreadAtlas store.

    Parses filter prefixes from ``raw_query``, dispatches to the
    appropriate search surfaces, and returns a unified result set.

    Search surfaces:
    * **Conversations** — FTS over title/summary/tags/messages. Skipped
      if ``kind:`` filters request only derived object types.
    * **Chunks** — FTS over chunk titles/bodies. Skipped if ``kind:``
      is set to non-chunk types.
    * **Derived objects** — LIKE search over title/description. Always
      included unless kind filters exclude all derived types.
    """
    t0 = time.monotonic()
    filt = parse_query(raw_query)
    vs = tuple(visible_states)
    all_hits: list[QueryHit] = []

    # Decide which surfaces to search based on kind filters.
    search_conversations_flag = True
    search_chunks_flag = True
    search_derived_flag = True

    if filt.kinds:
        # If the user asked for specific kinds, only search derived objects
        # of those kinds. Skip conversation/chunk FTS unless a kind maps to
        # something conversation-level (there is no "conversation" kind; this
        # is purely derived object types).
        search_conversations_flag = False
        search_chunks_flag = False
        search_derived_flag = True

    # Project-scoped conversation IDs.
    project_cids = _apply_project_filter_to_conversations(store, filt, vs)

    has_text = bool(filt.text)

    # --- Conversation search ---
    if search_conversations_flag:
        if has_text:
            conv_hits = search_conversations(
                store, filt.text, visible_states=vs, limit=limit * 2,
            )
            conv_hits = _apply_conversation_filters(store, conv_hits, filt)
            if project_cids is not None:
                conv_hits = [h for h in conv_hits if h.conversation_id in project_cids]
            for h in conv_hits[:limit]:
                all_hits.append(QueryHit(
                    hit_type="conversation",
                    id=h.conversation_id,
                    title=h.title,
                    snippet=h.snippet,
                    score=h.score,
                    metadata={"source": h.source, "state": h.state},
                ))
        else:
            # No free text — list mode with filters only.
            list_hits = _list_conversations(
                store, filt, visible_states=vs, limit=limit,
            )
            if project_cids is not None:
                list_hits = [h for h in list_hits if h.id in project_cids]
            all_hits.extend(list_hits[:limit])

    # --- Chunk search ---
    if search_chunks_flag and has_text:
        chunk_hits = search_chunks(
            store, filt.text, visible_states=vs, limit=limit,
        )
        if project_cids is not None:
            chunk_hits = [h for h in chunk_hits if h.conversation_id in project_cids]
        if filt.sources or filt.tags or filt.after or filt.before:
            chunk_hits = _apply_conversation_filters(store, chunk_hits, filt)
        for h in chunk_hits[:limit]:
            all_hits.append(QueryHit(
                hit_type="chunk",
                id=h.chunk_id or h.conversation_id,
                title=h.title,
                snippet=h.snippet,
                score=h.score,
                metadata={
                    "conversation_id": h.conversation_id,
                    "source": h.source,
                    "state": h.state,
                },
            ))

    # --- Derived object search ---
    if search_derived_flag:
        derived_hits = _search_derived_objects(
            store,
            filt.text,
            kinds=filt.kinds or None,
            visible_states=vs,
            limit=limit,
        )
        if filt.projects:
            # Filter derived objects to those linked to the target projects.
            proj_set = set(filt.projects)
            derived_hits = [
                h for h in derived_hits
                if h.metadata.get("project_id") in proj_set
            ]
        all_hits.extend(derived_hits)

    # --- Rank + dedupe ---
    all_hits.sort(key=lambda h: h.score, reverse=True)

    # Dedupe by (hit_type, id).
    seen: set[tuple[str, str]] = set()
    deduped: list[QueryHit] = []
    for h in all_hits:
        key = (h.hit_type, h.id)
        if key not in seen:
            seen.add(key)
            deduped.append(h)

    final = deduped[:limit]

    # Compute totals by type.
    totals: dict[str, int] = {}
    for h in final:
        totals[h.hit_type] = totals.get(h.hit_type, 0) + 1

    elapsed = (time.monotonic() - t0) * 1000
    return QueryResult(
        raw_query=raw_query,
        filters=filt.to_dict(),
        hits=final,
        total_by_type=totals,
        elapsed_ms=round(elapsed, 2),
    )
