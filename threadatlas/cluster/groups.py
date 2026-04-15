"""Thematic grouping pipeline.

1. Collect conversation summaries for eligible states (indexed + private).
2. TF-IDF vectorize.
3. K-means at two levels (broad, fine).
4. Keyword labels from distinctive terms per centroid.
5. Persist (replacing any prior groups at each level).

Deterministic end-to-end: given the same input corpus, the same
``broad_k``, ``fine_k``, and ``seed`` always produce identical groups and
labels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.models import EXTRACTABLE_STATES, new_id
from ..store import Store
from .kmeans import cosine_similarity, kmeans
from .tfidf import build_tfidf, distinctive_terms


@dataclass
class GroupingResult:
    broad_groups: int = 0
    fine_groups: int = 0
    members: int = 0
    skipped_empty_corpus: bool = False
    broad_generation_id: str = ""
    fine_generation_id: str = ""


def _document_for(conv) -> str:
    """Build the text we feed TF-IDF for one conversation.

    Priority: title, manual tags, summary_short. We intentionally do NOT
    include raw message bodies here - that would swamp the vocabulary with
    chat filler. The summary is the analyst-curated distillation and is
    the right level of abstraction for grouping.
    """
    parts: list[str] = []
    if conv.title:
        parts.append(conv.title)
        # Double-weight titles by including them twice.
        parts.append(conv.title)
    parts.extend(conv.manual_tags or [])
    parts.extend(conv.auto_tags or [])
    if conv.summary_short:
        parts.append(conv.summary_short)
    if conv.summary_long:
        parts.append(conv.summary_long)
    return "\n".join(parts)


def _top_members_by_similarity(
    vectors: list[dict[str, float]],
    assignments: list[int],
    centroids: list[dict[str, float]],
    group_index: int,
    conversation_ids: list[str],
    top_n: int,
) -> list[str]:
    """Return conversation_ids of the top-N members in ``group_index`` by
    cosine similarity to that centroid.

    Used by downstream LLM naming: we want the *most representative* members
    to describe the cluster.
    """
    members: list[tuple[float, str]] = []
    for i, a in enumerate(assignments):
        if a != group_index:
            continue
        sim = cosine_similarity(vectors[i], centroids[group_index])
        members.append((sim, conversation_ids[i]))
    members.sort(reverse=True)
    return [cid for _, cid in members[:top_n]]


def _format_keyword_label(terms: list[str]) -> str:
    """Render distinctive terms into a stable keyword label.

    Example: ``['chs', 'staffing', 'q2']`` -> ``"chs, staffing, q2"``. We
    keep commas so operators can tell these are keywords, not prose.
    """
    return ", ".join(terms) or "(unlabeled)"


def regroup_all(
    store: Store,
    *,
    broad_k: int = 10,
    fine_k: int = 100,
    seed: int = 42,
    min_corpus_size: int = 4,
) -> GroupingResult:
    """Regenerate broad + fine groups across the FTS-indexed corpus.

    Only runs over conversations in ``EXTRACTABLE_STATES`` (indexed +
    private). Quarantined and pending_review content never participates in
    grouping.
    """
    state_placeholders = ",".join(f"'{s}'" for s in EXTRACTABLE_STATES)
    rows = store.conn.execute(
        f"""
        SELECT conversation_id, title, summary_short, summary_long,
               manual_tags, auto_tags
          FROM conversations
         WHERE state IN ({state_placeholders})
         ORDER BY conversation_id
        """
    ).fetchall()

    # Eager in-memory; single-user vault scale.
    import json as _json

    class _C:  # Minimal duck type for _document_for
        def __init__(self, row):
            self.title = row["title"] or ""
            self.summary_short = row["summary_short"] or ""
            self.summary_long = row["summary_long"]
            self.manual_tags = _json.loads(row["manual_tags"] or "[]")
            self.auto_tags = _json.loads(row["auto_tags"] or "[]")

    conversations = [(r["conversation_id"], _C(r)) for r in rows]
    if len(conversations) < min_corpus_size:
        # Too small to cluster usefully. Wipe any stale groups and return.
        store.conn.execute("DELETE FROM conversation_group_memberships")
        store.conn.execute("DELETE FROM conversation_groups")
        store.conn.commit()
        return GroupingResult(skipped_empty_corpus=True)

    conversation_ids = [cid for cid, _ in conversations]
    docs = [_document_for(c) for _, c in conversations]
    vectors, vocab, idf = build_tfidf(docs)

    now = time.time()
    broad_gen = new_id("gen")
    fine_gen = new_id("gen")

    def _run_level(level: str, k: int, gen_id: str):
        k_eff = min(k, len(conversations))
        assignments, centroids = kmeans(vectors, k_eff, seed=seed)
        group_rows = []
        for j in range(len(centroids)):
            member_ids = [
                conversation_ids[i] for i, a in enumerate(assignments) if a == j
            ]
            if not member_ids:
                # Degenerate empty cluster - skip rather than persist it.
                continue
            distinctive = distinctive_terms(
                centroids[j],
                [centroids[x] for x in range(len(centroids)) if x != j],
                top_k=5,
            )
            group_rows.append({
                "group_id": new_id("grp"),
                "keyword_label": _format_keyword_label(distinctive),
                "llm_label": None,
                "member_count": len(member_ids),
                "member_ids": member_ids,
                "created_at": now,
                "generation_id": gen_id,
            })
        store.replace_groups(level, group_rows)
        return group_rows, assignments, centroids

    broad_rows, broad_assign, broad_centroids = _run_level("broad", broad_k, broad_gen)
    fine_rows, fine_assign, fine_centroids = _run_level("fine", fine_k, fine_gen)
    store.conn.commit()

    return GroupingResult(
        broad_groups=len(broad_rows),
        fine_groups=len(fine_rows),
        members=len(conversations),
        broad_generation_id=broad_gen,
        fine_generation_id=fine_gen,
    )


def top_members_for_group(
    store: Store, group_id: str, top_n: int = 15
) -> list[str]:
    """Return the top-N member conversation_ids of a group by centroid cosine.

    We recompute the centroid on the fly from the group's current members so
    we don't need to persist vectors.
    """
    member_ids = store.list_group_members(group_id)
    if not member_ids:
        return []
    placeholders = ",".join("?" for _ in member_ids)
    rows = store.conn.execute(
        f"""
        SELECT conversation_id, title, summary_short, summary_long,
               manual_tags, auto_tags
          FROM conversations
         WHERE conversation_id IN ({placeholders})
        """,
        member_ids,
    ).fetchall()

    import json as _json

    class _C:
        def __init__(self, row):
            self.title = row["title"] or ""
            self.summary_short = row["summary_short"] or ""
            self.summary_long = row["summary_long"]
            self.manual_tags = _json.loads(row["manual_tags"] or "[]")
            self.auto_tags = _json.loads(row["auto_tags"] or "[]")

    conv_by_id = {r["conversation_id"]: _C(r) for r in rows}
    ordered_ids = [cid for cid in member_ids if cid in conv_by_id]
    docs = [_document_for(conv_by_id[cid]) for cid in ordered_ids]
    vectors, _, _ = build_tfidf(docs)
    if not vectors:
        return ordered_ids[:top_n]
    # Centroid = mean of member vectors.
    from .kmeans import _centroid_mean
    centroid = _centroid_mean(vectors)
    ranked = sorted(
        zip(ordered_ids, vectors),
        key=lambda pair: cosine_similarity(pair[1], centroid),
        reverse=True,
    )
    return [cid for cid, _ in ranked[:top_n]]
