"""Embedding generation and semantic search.

Uses a simple TF-IDF-based embedding as the default (no external model needed).
When a local embedding model is available via the LLM runner, it can be used
instead for higher-quality results.

The TF-IDF approach works entirely offline with no dependencies beyond the
standard library, which aligns with ThreadAtlas's design principles.
"""

from __future__ import annotations

import math
import re
import struct
import time
from collections import Counter
from typing import Iterable

from ..core.models import EXTRACTABLE_STATES, MCP_VISIBLE_STATES
from ..store import Store


# ---------------------------------------------------------------------------
# Simple TF-IDF embedding (no external dependencies)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]{2,}")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "into", "have", "has",
    "your", "you", "are", "was", "were", "but", "not", "any", "all", "can",
    "could", "would", "should", "what", "when", "where", "which", "while",
    "about", "they", "them", "their", "there", "here", "then", "than", "also",
    "just", "like", "want", "need", "make", "made", "use", "using", "used",
    "one", "two", "three", "lot", "more", "most", "some", "such", "very",
    "well", "yes", "okay", "thanks", "thank", "please", "really", "actually",
    "maybe", "got", "been", "being", "will", "would", "could", "should",
    "does", "did", "had", "how", "its", "let", "may", "our", "own", "say",
    "she", "too", "way", "who", "get", "each", "see", "her", "him", "his",
    "new", "now", "old", "only", "other", "over", "still", "think", "try",
    "help", "know", "come", "much", "take", "because", "good", "give",
})

# Fixed vocabulary size for the embedding vectors.
EMBEDDING_DIM = 256


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "") if w.lower() not in _STOPWORDS]


class TFIDFEmbedder:
    """Build a fixed-dimension TF-IDF embedding from a corpus of documents."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._fitted = False

    def fit(self, documents: list[str]) -> None:
        """Build vocabulary and IDF from a corpus of documents."""
        doc_freq: Counter[str] = Counter()
        word_freq: Counter[str] = Counter()
        n_docs = len(documents)

        for doc in documents:
            tokens = _tokenize(doc)
            word_freq.update(tokens)
            doc_freq.update(set(tokens))

        # Pick top EMBEDDING_DIM terms by document frequency.
        top_terms = [t for t, _ in doc_freq.most_common(EMBEDDING_DIM)]
        self.vocab = {t: i for i, t in enumerate(top_terms)}

        for term, idx in self.vocab.items():
            df = doc_freq.get(term, 0)
            self.idf[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

        self._fitted = True

    def embed(self, text: str) -> list[float]:
        """Generate a TF-IDF embedding vector for a single document."""
        if not self._fitted:
            # Return zero vector if not fitted.
            return [0.0] * EMBEDDING_DIM

        tokens = _tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1

        vec = [0.0] * EMBEDDING_DIM
        for term, idx in self.vocab.items():
            if term in tf:
                vec[idx] = (tf[term] / total) * self.idf.get(term, 1.0)

        # L2 normalize.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def embedding_to_bytes(vec: list[float]) -> bytes:
    """Pack a float vector into raw bytes (float32)."""
    return struct.pack(f"{len(vec)}f", *vec)


def bytes_to_embedding(data: bytes) -> list[float]:
    """Unpack raw bytes into a float vector."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each ranked list is a list of (id, score) tuples sorted by score descending.
    Returns a merged list of (id, fused_score) sorted by fused_score descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Index-time embedding generation
# ---------------------------------------------------------------------------

def build_embeddings_for_conversation(
    store: Store,
    conversation_id: str,
    embedder: TFIDFEmbedder,
) -> int:
    """Generate and store embeddings for all chunks in a conversation.

    Returns the number of chunks embedded.
    """
    chunks = store.list_chunks(conversation_id)
    if not chunks:
        return 0

    now = time.time()
    count = 0
    for chunk in chunks:
        # Build text from chunk messages.
        msg_rows = store.conn.execute(
            "SELECT content_text FROM messages WHERE conversation_id = ? "
            "AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
            (conversation_id, chunk.start_message_ordinal, chunk.end_message_ordinal),
        ).fetchall()
        text = " ".join((r["content_text"] or "") for r in msg_rows)
        if not text.strip():
            continue

        vec = embedder.embed(text)
        store.upsert_chunk_embedding(
            chunk.chunk_id, embedding_to_bytes(vec), "tfidf-256", now
        )
        count += 1
    return count


def fit_embedder_from_corpus(store: Store) -> TFIDFEmbedder:
    """Build a TF-IDF embedder fitted on the entire visible corpus."""
    eligible_states = ",".join(f"'{s}'" for s in EXTRACTABLE_STATES)
    rows = store.conn.execute(
        f"""
        SELECT m.content_text
          FROM messages m
          JOIN conversations c ON c.conversation_id = m.conversation_id
         WHERE c.state IN ({eligible_states})
           AND m.role IN ('user', 'assistant')
        """,
    ).fetchall()

    # Group messages by conversation for document-level IDF.
    conv_rows = store.conn.execute(
        f"""
        SELECT c.conversation_id, c.title, c.summary_short
          FROM conversations c
         WHERE c.state IN ({eligible_states})
        """,
    ).fetchall()

    documents = []
    for cr in conv_rows:
        msg_rows = store.conn.execute(
            "SELECT content_text FROM messages WHERE conversation_id = ? ORDER BY ordinal",
            (cr["conversation_id"],),
        ).fetchall()
        doc_text = " ".join((r["content_text"] or "") for r in msg_rows)
        documents.append(doc_text)

    embedder = TFIDFEmbedder()
    if documents:
        embedder.fit(documents)
    return embedder


def build_all_embeddings(store: Store) -> int:
    """Build embeddings for all chunks across all eligible conversations."""
    embedder = fit_embedder_from_corpus(store)

    eligible_states = ",".join(f"'{s}'" for s in EXTRACTABLE_STATES)
    rows = store.conn.execute(
        f"SELECT conversation_id FROM conversations WHERE state IN ({eligible_states})"
    ).fetchall()

    total = 0
    for r in rows:
        total += build_embeddings_for_conversation(store, r["conversation_id"], embedder)
    store.conn.commit()
    return total
