"""Deterministic, heuristic-only chunking.

Per spec:
* Don't chunk by token count alone.
* Preserve message boundaries.
* Prefer a small number of meaningful chunks over many tiny fragments.
* Use chunk titles and summaries that help a human understand what changed.

Strategy
--------
1. Collect message records in order.
2. Walk forward, accumulating into a candidate chunk. Mark a topic boundary
   when:
   * the user role asks a clearly new question (heuristic: starts a new turn
     with low Jaccard overlap to the running keyword set), AND
   * the candidate already has a reasonable size (>= MIN_MESSAGES messages
     or >= MIN_CHARS chars).
3. After all boundaries are placed, merge any adjacent chunk smaller than
   MIN_MESSAGES with its larger neighbor. This avoids dust-sized fragments.
4. Title each chunk from the first user prompt; summary is the first ~200
   chars of the first user prompt + first assistant reply.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ..core.models import Chunk, EXTRACTABLE_STATES, Message, new_id
from ..core.vault import Vault
from ..store import Store


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "into", "have", "has",
    "your", "you", "i'm", "i've", "i'll", "are", "was", "were", "but", "not",
    "any", "all", "can", "could", "would", "should", "what", "when", "where",
    "which", "while", "about", "they", "them", "their", "there", "here",
    "then", "than", "also", "just", "like", "want", "need", "make", "made",
    "use", "using", "used", "one", "two", "three", "lot", "more", "most",
    "some", "such", "very", "well", "yes", "no", "ok", "okay", "thanks",
    "thank", "please", "really", "actually", "maybe", "got",
})

# Tunables. These are meant to feel right for typical chat lengths and
# explicitly trade away precision for a small number of useful chunks.
MIN_MESSAGES = 4      # min messages in a chunk before we'll cut
MIN_CHARS = 1200      # min chars in a chunk before we'll cut
MAX_CHARS_HINT = 9000  # hint to consider cutting even without a topic shift
JACCARD_NEW_TOPIC = 0.10  # below this overlap, treat as a new topic


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "") if w.lower() not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _title_from_user_message(text: str) -> str:
    if not text:
        return "Untitled chunk"
    # First non-empty line, trimmed to ~80 chars.
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text.strip())
    if len(line) > 80:
        line = line[:77].rstrip() + "..."
    return line


def _summary_for(messages: list[Message]) -> str:
    user_msg = next((m for m in messages if m.role == "user"), None)
    asst_msg = next((m for m in messages if m.role == "assistant"), None)
    parts: list[str] = []
    if user_msg:
        parts.append(_truncate(user_msg.content_text, 240))
    if asst_msg:
        parts.append("-> " + _truncate(asst_msg.content_text, 240))
    return " | ".join(parts)


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _build_boundaries(messages: list[Message]) -> list[tuple[int, int]]:
    """Return (start_ordinal, end_ordinal) ranges that partition the messages."""
    if not messages:
        return []
    ranges: list[list[int]] = []
    cur_start = messages[0].ordinal
    cur_end = messages[0].ordinal
    cur_chars = len(messages[0].content_text or "")
    cur_count = 1
    cur_tokens: set[str] = _tokens(messages[0].content_text)

    for prev, m in zip(messages, messages[1:]):
        m_tokens = _tokens(m.content_text)
        is_new_user_question = (
            m.role == "user" and prev.role == "assistant"
            and _jaccard(cur_tokens, m_tokens) < JACCARD_NEW_TOPIC
        )
        big_enough = cur_count >= MIN_MESSAGES and cur_chars >= MIN_CHARS
        too_big = cur_chars >= MAX_CHARS_HINT and m.role == "user" and prev.role == "assistant"

        if (is_new_user_question and big_enough) or too_big:
            ranges.append([cur_start, cur_end])
            cur_start = m.ordinal
            cur_end = m.ordinal
            cur_chars = len(m.content_text or "")
            cur_count = 1
            cur_tokens = m_tokens
        else:
            cur_end = m.ordinal
            cur_chars += len(m.content_text or "")
            cur_count += 1
            # Merge tokens softly so each new question is compared to recent
            # context, not just the previous message.
            cur_tokens |= m_tokens
    ranges.append([cur_start, cur_end])

    # Merge tiny adjacent chunks into their neighbor.
    by_ordinal = {m.ordinal: m for m in messages}
    def _count(r):
        return sum(1 for o in by_ordinal if r[0] <= o <= r[1])
    i = 0
    while i < len(ranges) and len(ranges) > 1:
        if _count(ranges[i]) < MIN_MESSAGES:
            if i == len(ranges) - 1:
                # merge with previous
                ranges[i - 1][1] = ranges[i][1]
                ranges.pop(i)
                i = max(i - 1, 0)
            else:
                ranges[i][1] = ranges[i + 1][1]
                ranges.pop(i + 1)
        else:
            i += 1
    return [(s, e) for s, e in ranges]


def chunk_conversation(store: Store, conversation_id: str) -> list[Chunk]:
    """(Re)compute chunks for a single conversation. Replaces existing chunks."""
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise KeyError(conversation_id)
    if conv.state not in EXTRACTABLE_STATES:
        # Quarantined and pending_review: clear chunks rather than create.
        store.replace_chunks(conversation_id, [])
        store.reindex_conversation_fts(conversation_id)
        store.conn.commit()
        return []
    msgs = store.list_messages(conversation_id)
    boundaries = _build_boundaries(msgs)
    by_ordinal = {m.ordinal: m for m in msgs}
    chunks: list[Chunk] = []
    for idx, (start, end) in enumerate(boundaries):
        body_msgs = [by_ordinal[o] for o in sorted(by_ordinal) if start <= o <= end]
        first_user = next((m for m in body_msgs if m.role == "user"), body_msgs[0] if body_msgs else None)
        title = _title_from_user_message(first_user.content_text) if first_user else f"Chunk {idx+1}"
        summary = _summary_for(body_msgs)
        chunks.append(Chunk(
            chunk_id=new_id("chk"),
            conversation_id=conversation_id,
            chunk_index=idx,
            start_message_ordinal=start,
            end_message_ordinal=end,
            chunk_title=title,
            summary_short=summary,
        ))
    store.replace_chunks(conversation_id, chunks)
    store.reindex_conversation_fts(conversation_id)
    store.conn.commit()
    return chunks


def chunk_all_eligible(store: Store) -> dict[str, int]:
    """Run chunking on every conversation in an extractable state.

    Returns ``{conversation_id: chunk_count}``.
    """
    out: dict[str, int] = {}
    eligible_states = ",".join(f"'{s}'" for s in EXTRACTABLE_STATES)
    rows = store.conn.execute(
        f"SELECT conversation_id FROM conversations WHERE state IN ({eligible_states})"
    ).fetchall()
    for r in rows:
        cid = r["conversation_id"]
        chunks = chunk_conversation(store, cid)
        out[cid] = len(chunks)
    return out
