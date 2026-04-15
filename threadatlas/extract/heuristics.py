"""Heuristic, no-LLM extraction of high-value derived objects.

Per spec, v1 extracts a small set of durable types:
  * project
  * entity (person or organization)
  * decision
  * open_loop
  * artifact
  * preference

Each extracted object stores provenance back to the source conversation and
chunk. Quality bar: store an object only if it crosses a practical threshold
- e.g., a project should look like a recurring workstream, not a one-off
mention.

Determinism is intentional. These rules are inspectable, debuggable, and run
without any network or model.
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..core.models import (
    Chunk,
    Conversation,
    DerivedKind,
    DerivedObject,
    EXTRACTABLE_STATES,
    Message,
    ProvenanceLink,
    new_id,
)
from ..store import Store, transaction


# --- regexes ----------------------------------------------------------------

# Project mentions: an explicit "project X" phrase is a high-precision signal.
# The keyword is case-insensitive (inline ``(?i:...)``) but the project name
# itself must start with a capital letter to keep precision up. A trailing
# word boundary prevents the group from swallowing trailing clauses.
_PROJECT_RX = re.compile(
    r"\b(?i:project|initiative|workstream)\s+([A-Z][\w&\-/]{1,40}(?:\s+[A-Z][\w&\-/]{1,40}){0,3})\b",
)

# A single-token acronym in ALL CAPS that recurs across messages is a likely
# project/code-name. We harvest then filter by recurrence.
_ACRONYM_RX = re.compile(r"\b([A-Z]{2,6}(?:-?[A-Z0-9]{0,4})?)\b")

# People/orgs: conservative proper-noun bigram match. We deliberately keep
# this conservative and bias towards under-merging.
_NAMED_ENTITY_RX = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
)

# Decisions: we want clearly committed or decided phrasing. "I will" paired
# with a weak verb ("see", "try") is too noisy; require an action-y verb
# afterwards, or a decision keyword like "decided", "chose", "picked",
# "went with", or an explicit "decision:" / "verdict:" prefix.
_DECISION_PATTERNS = [
    # Commit verbs that imply a choice was made.
    re.compile(
        r"\b(?:I|we)\s+(?:decided to|chose to|picked|went with|are going with|settled on|agreed to)\s+\w{2,}[^.\n]{2,160}",
        re.I,
    ),
    # "I will <verb>" - require the next token to be a plausible action verb
    # (2+ chars), and the sentence to carry at least a few more words.
    re.compile(
        r"\b(?:I|we)\s+(?:will|am going to|are going to)\s+\w{2,}[^.\n]{6,160}",
        re.I,
    ),
    re.compile(r"\b(?:final(?:ly)? )?(?:decision|verdict)\s*[:\-]\s*[^.\n]{4,160}", re.I),
]

# Open loops: precision signals only.
# * Explicit TODO markers (case-insensitive but word-bounded).
# * First-person "I still need to ...", "I need to ..." (not third-person).
# * "remember to ...", "don't forget to ..." action phrases.
# * "revisit" / "circle back" with an object.
# * Explicit "open question" / "unresolved" prefixes.
#
# Intentionally dropped: loose "follow up" (matched "as a follow up
# question" in casual prose), and second-person "you need to" prescriptions
# (those are advice, not the operator's open loops).
_OPEN_LOOP_PATTERNS = [
    re.compile(r"\bTODO\b[^.\n]{0,160}"),
    re.compile(r"\bTO[\s-]?DO\b[^.\n]{0,160}"),
    re.compile(r"\b(?:I|we)\s+(?:still\s+)?need\s+to\b[^.\n]{4,160}", re.I),
    re.compile(r"\b(?:remember to|don't forget to|reminder to)\b[^.\n]{4,160}", re.I),
    re.compile(r"\b(?:revisit|come back to|circle back to)\b\s+\w{2,}[^.\n]{0,160}", re.I),
    re.compile(r"\bopen question\b[^.\n]{0,160}", re.I),
    re.compile(r"\bunresolved\b[^.\n]{0,160}", re.I),
    re.compile(r"\b(?:haven't|have not)\s+decided\b[^.\n]{0,160}", re.I),
    re.compile(r"\b(?:follow[\s-]?up)\s+(?:on|with|about)\b[^.\n]{4,160}", re.I),
]

# Artifacts: documents produced or drafted.
_ARTIFACT_PATTERNS = [
    re.compile(r"\b(?:drafted|wrote|outline (?:of|for)|memo (?:on|about|for)|spec (?:for|on)|deck (?:for|on)|doc (?:on|for))\s+(?:a\s+|the\s+)?[a-z0-9 \-/_]{3,80}", re.I),
]

# Preferences: stable personal patterns. We explicitly drop "I like"/"I hate"
# because they fire constantly on casual remarks ("I like that idea"). Require
# stability-signalling words: prefer/always/usually/tend, or an explicit
# "my preference is ..." / "my rule is ...".
_PREFERENCE_PATTERNS = [
    re.compile(r"\bI\s+(?:prefer|always|usually|tend to)\b[^.\n]{4,160}", re.I),
    re.compile(r"\bmy\s+(?:preference|style|approach|rule)\s+(?:is|for)\b[^.\n]{4,160}", re.I),
]

# Bigram entity false-positive starters. When the first word of an
# "entity" is one of these, we skip the match. This does not try to be a
# full NER system - it just kills the top noisy sources for AI chat prose.
_ENTITY_START_STOPWORDS = frozenset({
    "Hi", "Hello", "Hey", "Thanks", "Thank", "Dear", "Sincerely",
    "The", "And", "But", "Or", "If", "When", "While", "After", "Before",
    "However", "Because", "Since", "Although", "Though", "Also",
    "You", "Your", "We", "Our", "They", "Their", "Let", "Here", "There",
    "As", "At", "In", "On", "For", "With", "From", "To", "By", "About",
    "This", "That", "These", "Those",
    # Common AI-prose starters that look like names when capitalized:
    "Sure", "Sorry", "Please", "Okay", "Great", "Good", "Excellent",
    "Got", "Done", "Noted", "Yes", "No", "Maybe",
})


def _excerpt(text: str, match_text: str, before: int = 60, after: int = 120) -> str:
    idx = text.find(match_text)
    if idx < 0:
        return match_text.strip()[:240]
    start = max(0, idx - before)
    end = min(len(text), idx + len(match_text) + after)
    snippet = text[start:end].replace("\n", " ")
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet.strip()[:400]


def _canon_key(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


@dataclass
class _Hit:
    kind: str
    title: str
    description: str
    excerpt: str
    chunk_id: str | None
    canonical_key: str = ""


def _find_chunk_for_message(chunks: list[Chunk], ordinal: int) -> Chunk | None:
    for ch in chunks:
        if ch.start_message_ordinal <= ordinal <= ch.end_message_ordinal:
            return ch
    return None


def _harvest_decisions(messages: list[Message], chunks: list[Chunk]) -> list[_Hit]:
    """Collect decisions.

    Overlapping patterns on the same sentence are deduplicated by canonical
    key so a single phrasing does not spawn multiple "decision" objects.
    """
    hits: list[_Hit] = []
    seen_keys: set[str] = set()
    for m in messages:
        if m.role not in {"user", "assistant"}:
            continue
        for rx in _DECISION_PATTERNS:
            for match in rx.finditer(m.content_text or ""):
                text = match.group(0).strip()
                key = _canon_key(text)[:120]
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                ch = _find_chunk_for_message(chunks, m.ordinal)
                hits.append(_Hit(
                    kind=DerivedKind.DECISION.value,
                    title=text[:80],
                    description=text,
                    excerpt=_excerpt(m.content_text, text),
                    chunk_id=ch.chunk_id if ch else None,
                    canonical_key=key,
                ))
    return hits


def _harvest_open_loops(messages: list[Message], chunks: list[Chunk]) -> list[_Hit]:
    hits: list[_Hit] = []
    seen_keys: set[str] = set()
    for m in messages:
        if m.role not in {"user", "assistant"}:
            continue
        for rx in _OPEN_LOOP_PATTERNS:
            for match in rx.finditer(m.content_text or ""):
                text = match.group(0).strip()
                key = _canon_key(text)[:120]
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                ch = _find_chunk_for_message(chunks, m.ordinal)
                hits.append(_Hit(
                    kind=DerivedKind.OPEN_LOOP.value,
                    title=text[:80],
                    description=text,
                    excerpt=_excerpt(m.content_text, text),
                    chunk_id=ch.chunk_id if ch else None,
                    canonical_key=key,
                ))
    return hits


def _harvest_preferences(messages: list[Message], chunks: list[Chunk]) -> list[_Hit]:
    hits: list[_Hit] = []
    # Track canonical keys to suppress repeated phrasings within the same
    # conversation - a preference repeated by the user is still just one
    # preference.
    seen_keys: set[str] = set()
    for m in messages:
        if m.role != "user":
            continue
        for rx in _PREFERENCE_PATTERNS:
            for match in rx.finditer(m.content_text or ""):
                text = match.group(0).strip()
                key = _canon_key(text)[:120]
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                ch = _find_chunk_for_message(chunks, m.ordinal)
                hits.append(_Hit(
                    kind=DerivedKind.PREFERENCE.value,
                    title=text[:80],
                    description=text,
                    excerpt=_excerpt(m.content_text, text),
                    chunk_id=ch.chunk_id if ch else None,
                    canonical_key=key,
                ))
    return hits


def _harvest_artifacts(messages: list[Message], chunks: list[Chunk]) -> list[_Hit]:
    hits: list[_Hit] = []
    for m in messages:
        for rx in _ARTIFACT_PATTERNS:
            for match in rx.finditer(m.content_text or ""):
                text = match.group(0).strip()
                ch = _find_chunk_for_message(chunks, m.ordinal)
                hits.append(_Hit(
                    kind=DerivedKind.ARTIFACT.value,
                    title=text[:80],
                    description=text,
                    excerpt=_excerpt(m.content_text, text),
                    chunk_id=ch.chunk_id if ch else None,
                ))
    return hits


def _harvest_projects(
    conv: Conversation, messages: list[Message], chunks: list[Chunk]
) -> list[_Hit]:
    """Project candidates come from explicit "project X" or recurring acronyms."""
    hits: list[_Hit] = []
    seen: set[str] = set()
    # Explicit "project X" mentions are high-confidence.
    for m in messages:
        for match in _PROJECT_RX.finditer(m.content_text or ""):
            label = match.group(1).strip()
            key = _canon_key(label)
            if not key or key in seen:
                continue
            seen.add(key)
            ch = _find_chunk_for_message(chunks, m.ordinal)
            hits.append(_Hit(
                kind=DerivedKind.PROJECT.value,
                title=label[:80],
                description=f"Mentioned as project '{label}' in '{conv.title}'",
                excerpt=_excerpt(m.content_text, match.group(0)),
                chunk_id=ch.chunk_id if ch else None,
                canonical_key=key,
            ))
    # Recurring acronyms (>= 3 occurrences) become project candidates if not
    # already an explicit project.
    counts: Counter[str] = Counter()
    first_seen: dict[str, Message] = {}
    for m in messages:
        for match in _ACRONYM_RX.finditer(m.content_text or ""):
            tok = match.group(1)
            counts[tok] += 1
            first_seen.setdefault(tok, m)
    for tok, n in counts.items():
        key = _canon_key(tok)
        if n < 3 or key in seen:
            continue
        # Skip very generic acronyms.
        if tok in {"OK", "HTTP", "JSON", "API", "URL", "AI", "LLM", "PDF", "CSV", "XLSX", "TODO", "FAQ", "UI", "UX", "CTO", "CEO", "CFO", "USA", "EU", "UK"}:
            continue
        seen.add(key)
        m = first_seen[tok]
        ch = _find_chunk_for_message(chunks, m.ordinal)
        hits.append(_Hit(
            kind=DerivedKind.PROJECT.value,
            title=tok,
            description=f"Recurring acronym '{tok}' ({n}x) in '{conv.title}'",
            excerpt=_excerpt(m.content_text, tok),
            chunk_id=ch.chunk_id if ch else None,
            canonical_key=key,
        ))
    return hits


def _harvest_entities(messages: list[Message], chunks: list[Chunk]) -> list[_Hit]:
    """Conservative proper-noun bigram extraction with a recurrence threshold.

    Only emit if the same name appears >= 2 times, to keep noise down. We
    additionally drop matches that start with a stopword ("Hi Claude",
    "Thanks John", "Sure John") since those are overwhelmingly noise in AI
    chat prose.
    """
    counts: Counter[str] = Counter()
    first_seen: dict[str, Message] = {}
    for m in messages:
        for match in _NAMED_ENTITY_RX.finditer(m.content_text or ""):
            name = match.group(1).strip()
            first = name.split()[0]
            if first in _ENTITY_START_STOPWORDS:
                continue
            # Also drop bigrams where any word is a stopword-start (catches
            # "John And Jane" and similar structural tokens).
            if any(w in _ENTITY_START_STOPWORDS for w in name.split()):
                continue
            counts[name] += 1
            first_seen.setdefault(name, m)
    hits: list[_Hit] = []
    for name, n in counts.items():
        if n < 2:
            continue
        m = first_seen[name]
        ch = _find_chunk_for_message(chunks, m.ordinal)
        hits.append(_Hit(
            kind=DerivedKind.ENTITY.value,
            title=name,
            description=f"Repeated proper noun '{name}' ({n}x).",
            excerpt=_excerpt(m.content_text, name),
            chunk_id=ch.chunk_id if ch else None,
            canonical_key=_canon_key(name),
        ))
    return hits


def _persist_hits(
    store: Store, conv: Conversation, hits: list[_Hit]
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    now = time.time()
    for h in hits:
        canonical_key = h.canonical_key or _canon_key(h.title)
        obj = DerivedObject(
            object_id=new_id("obj"),
            kind=h.kind,
            title=h.title,
            description=h.description,
            project_id=None,
            state="active",
            canonical_key=canonical_key,
            created_at=now,
            updated_at=now,
        )
        actual_id = store.upsert_derived_object(obj)
        link = ProvenanceLink(
            link_id=new_id("prov"),
            object_id=actual_id,
            conversation_id=conv.conversation_id,
            chunk_id=h.chunk_id,
            excerpt=h.excerpt,
            created_at=now,
        )
        store.insert_provenance(link)
        counts[h.kind] += 1
    return dict(counts)


def extract_for_conversation(store: Store, conversation_id: str) -> dict[str, int]:
    """Run heuristic extraction for a single conversation. Replaces previous
    provenance for this conversation (so re-running is idempotent)."""
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise KeyError(conversation_id)
    if conv.state not in EXTRACTABLE_STATES:
        return {}

    msgs = store.list_messages(conversation_id)
    chunks = store.list_chunks(conversation_id)

    hits: list[_Hit] = []
    hits += _harvest_projects(conv, msgs, chunks)
    hits += _harvest_decisions(msgs, chunks)
    hits += _harvest_open_loops(msgs, chunks)
    hits += _harvest_preferences(msgs, chunks)
    hits += _harvest_artifacts(msgs, chunks)
    hits += _harvest_entities(msgs, chunks)

    has_open_loops = any(h.kind == DerivedKind.OPEN_LOOP.value for h in hits)

    # importance score: simple, inspectable, non-magical.
    importance = 0.0
    importance += min(len(msgs) / 20.0, 2.0)            # length
    importance += sum(1 for h in hits if h.kind == DerivedKind.DECISION.value) * 0.5
    importance += sum(1 for h in hits if h.kind == DerivedKind.PROJECT.value) * 0.7
    importance += sum(1 for h in hits if h.kind == DerivedKind.OPEN_LOOP.value) * 0.3
    importance = round(min(importance, 10.0), 3)

    resurfacing = 0.0
    resurfacing += sum(1 for h in hits if h.kind == DerivedKind.OPEN_LOOP.value) * 0.7
    resurfacing += sum(1 for h in hits if h.kind == DerivedKind.PROJECT.value) * 0.2
    resurfacing = round(min(resurfacing, 10.0), 3)

    with transaction(store):
        # Reset prior provenance for this conversation; orphan derived objects
        # get cleaned up afterwards.
        store.conn.execute(
            "DELETE FROM provenance_links WHERE conversation_id = ?",
            (conversation_id,),
        )
        counts = _persist_hits(store, conv, hits)
        store.delete_orphan_derived_objects()
        store.update_conversation_meta(
            conversation_id,
            has_open_loops=has_open_loops,
            importance_score=importance,
            resurfacing_score=resurfacing,
            summary_short=_make_short_summary(conv, msgs, chunks),
        )
        store.reindex_conversation_fts(conversation_id)
    return counts


def _make_short_summary(conv: Conversation, msgs: list[Message], chunks: list[Chunk]) -> str:
    if conv.summary_short:
        return conv.summary_short
    first_user = next((m for m in msgs if m.role == "user"), None)
    n_chunks = len(chunks)
    n_msgs = len(msgs)
    base = f"{n_msgs} msgs, {n_chunks} chunks."
    if first_user:
        snippet = (first_user.content_text or "").strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        return f"{base} Opened with: {snippet}"
    return base


def extract_all_eligible(store: Store) -> dict[str, dict[str, int]]:
    """Run extraction for every extractable conversation."""
    out: dict[str, dict[str, int]] = {}
    eligible_states = ",".join(f"'{s}'" for s in EXTRACTABLE_STATES)
    rows = store.conn.execute(
        f"SELECT conversation_id FROM conversations WHERE state IN ({eligible_states})"
    ).fetchall()
    for r in rows:
        cid = r["conversation_id"]
        out[cid] = extract_for_conversation(store, cid)
    return out
