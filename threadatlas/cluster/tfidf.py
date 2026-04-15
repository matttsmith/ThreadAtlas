"""Pure-Python TF-IDF.

Stdlib only. Vectors are ``dict[str, float]`` so memory stays sparse and we
can run without numpy. For a single-user vault (thousands of conversations,
not millions), this is fast enough and easy to reason about.

Notes
-----
* All vectors are L2-normalized so cosine similarity degrades to a dot
  product.
* ``idf`` uses the conventional +1 smoothing:
  ``idf(t) = log((1 + N) / (1 + df(t))) + 1``.
* Vocabulary is pruned by document frequency: we keep the top
  ``max_features`` terms by descending df.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


_TOKEN_RX = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


# Intentionally small. We reuse the chunking stopword list as a starting
# point and add a few more common chat-prose tokens.
STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "into", "have", "has",
    "your", "you", "are", "was", "were", "but", "not", "any", "all", "can",
    "could", "would", "should", "what", "when", "where", "which", "while",
    "about", "they", "them", "their", "there", "here", "then", "than",
    "also", "just", "like", "want", "need", "make", "made", "use", "using",
    "used", "one", "two", "three", "lot", "more", "most", "some", "such",
    "very", "well", "yes", "no", "ok", "okay", "thanks", "thank", "please",
    "really", "actually", "maybe", "got", "will", "shall", "may", "might",
    "does", "did", "doing", "done", "been", "being", "were", "its", "it's",
    "my", "me", "we", "our", "us", "i'm", "i've", "i'll", "don't", "isn't",
    "msg", "msgs", "chunk", "chunks", "message", "messages",
    "user", "assistant", "system",
})


def tokenize(text: str) -> list[str]:
    """Unicode-lowercase tokens, 3+ chars, no stopwords.

    Deliberately simple. We do NOT stem: stemming obscures origin terms and
    makes distinctive labels harder to read.
    """
    return [
        t.lower()
        for t in _TOKEN_RX.findall(text or "")
        if t.lower() not in STOPWORDS
    ]


def build_tfidf(
    docs: list[str],
    *,
    max_features: int = 4000,
    min_df: int = 2,
) -> tuple[list[dict[str, float]], list[str], dict[str, float]]:
    """Compute TF-IDF for a list of documents.

    Returns ``(vectors, vocab, idf)``:
      * ``vectors[i]`` is the L2-normalized TF-IDF vector for ``docs[i]``,
        as ``dict[term, weight]``.
      * ``vocab`` is the pruned vocabulary (list of kept terms).
      * ``idf`` is the computed IDF per kept term.
    """
    tokenized = [tokenize(d) for d in docs]

    # Document frequency.
    df: Counter[str] = Counter()
    for toks in tokenized:
        for t in set(toks):
            df[t] += 1

    # Prune: keep only terms with df >= min_df, then top-N by df.
    kept_items = [(t, n) for t, n in df.most_common() if n >= min_df]
    vocab_items = kept_items[:max_features]
    vocab = [t for t, _ in vocab_items]
    vocab_set = set(vocab)

    n_docs = max(len(docs), 1)
    idf = {
        t: math.log((1 + n_docs) / (1 + df[t])) + 1.0
        for t in vocab
    }

    vectors: list[dict[str, float]] = []
    for toks in tokenized:
        tf: Counter[str] = Counter(t for t in toks if t in vocab_set)
        vec: dict[str, float] = {}
        norm_sq = 0.0
        for t, count in tf.items():
            w = float(count) * idf[t]
            if w:
                vec[t] = w
                norm_sq += w * w
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            vec = {t: w / norm for t, w in vec.items()}
        vectors.append(vec)
    return vectors, vocab, idf


def distinctive_terms(
    centroid: dict[str, float],
    other_centroids: Iterable[dict[str, float]],
    *,
    top_k: int = 5,
) -> list[str]:
    """Top terms in ``centroid`` that are NOT generic across other centroids.

    Simple deterministic rule: for each term in the centroid, compute the
    centroid's weight minus the mean weight of that term across the other
    centroids. Rank by this margin; return the top ``top_k`` surviving terms.
    """
    others = list(other_centroids)
    n_other = max(len(others), 1)
    margins: list[tuple[str, float]] = []
    for term, w in centroid.items():
        mean_other = sum(c.get(term, 0.0) for c in others) / n_other
        margins.append((term, w - mean_other))
    margins.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in margins[:top_k]]
