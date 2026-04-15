"""MCP-safe group labels.

Grouping runs over the mixed ``indexed + private`` pool so clusters have
enough data to be useful. But that means the persisted
``keyword_label`` / ``llm_label`` on a group may be derived from
``private`` content. Exposing those labels via MCP would leak summaries
of private conversations into the MCP surface indirectly.

This module computes *safe* labels: labels derived from ONLY the
indexed members of a group. If the group has fewer than
``MIN_INDEXED_FOR_SAFE_LABEL`` indexed members, we refuse to produce a
label (returns ``None``) - the group is simply summarized as "N indexed
conversations".
"""

from __future__ import annotations

import json

from ..store import Store
from .tfidf import build_tfidf, distinctive_terms


MIN_INDEXED_FOR_SAFE_LABEL = 3


def _document_for_row(row) -> str:
    parts: list[str] = []
    t = row["title"] or ""
    if t:
        parts.append(t)
        parts.append(t)  # double-weight the title
    for tags_col in ("manual_tags", "auto_tags"):
        for tag in json.loads(row[tags_col] or "[]"):
            parts.append(tag)
    if row["summary_short"]:
        parts.append(row["summary_short"])
    if row["summary_long"]:
        parts.append(row["summary_long"])
    return "\n".join(parts)


def compute_safe_keyword_label(
    store: Store, group_id: str, *, top_k: int = 5
) -> str | None:
    """Compute a keyword label from indexed members of this group only.

    Returns ``None`` if fewer than ``MIN_INDEXED_FOR_SAFE_LABEL`` indexed
    members exist (too few to make a credible label that isn't just one
    conversation's text).
    """
    rows = store.conn.execute(
        """
        SELECT c.title, c.summary_short, c.summary_long, c.manual_tags, c.auto_tags
          FROM conversation_group_memberships m
          JOIN conversations c ON c.conversation_id = m.conversation_id
         WHERE m.group_id = ? AND c.state = 'indexed'
        """,
        (group_id,),
    ).fetchall()
    if len(rows) < MIN_INDEXED_FOR_SAFE_LABEL:
        return None

    # Build a full corpus context: use the other groups' indexed members as
    # the "other clusters" reference for distinctive-term ranking.
    other_rows = store.conn.execute(
        """
        SELECT c.title, c.summary_short, c.summary_long, c.manual_tags, c.auto_tags
          FROM conversations c
          LEFT JOIN conversation_group_memberships m
                 ON m.conversation_id = c.conversation_id AND m.group_id = ?
         WHERE c.state = 'indexed' AND m.conversation_id IS NULL
        """,
        (group_id,),
    ).fetchall()

    this_docs = [_document_for_row(r) for r in rows]
    other_docs = [_document_for_row(r) for r in other_rows]
    all_docs = this_docs + other_docs
    if not all_docs:
        return None

    vectors, _, _ = build_tfidf(all_docs, min_df=1)
    this_vecs = vectors[: len(this_docs)]
    other_vecs = vectors[len(this_docs):]

    # Centroid of this group's indexed members.
    from .kmeans import _centroid_mean
    centroid = _centroid_mean(this_vecs)
    if not centroid:
        return None
    other_centroid = _centroid_mean(other_vecs) if other_vecs else {}
    terms = distinctive_terms(centroid, [other_centroid] if other_centroid else [], top_k=top_k)
    if not terms:
        return None
    return ", ".join(terms)
