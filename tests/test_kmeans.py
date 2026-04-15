"""K-means determinism + edge-case tests."""

from __future__ import annotations

from threadatlas.cluster.kmeans import cosine_similarity, kmeans
from threadatlas.cluster.tfidf import build_tfidf


def _cluster_sets(assignments: list[int]) -> list[set[int]]:
    """Return a canonical cluster representation (frozenset per cluster)."""
    buckets: dict[int, set[int]] = {}
    for i, a in enumerate(assignments):
        buckets.setdefault(a, set()).add(i)
    # Sort by min element for canonical ordering.
    return sorted(buckets.values(), key=lambda s: min(s))


def test_kmeans_empty_input():
    a, c = kmeans([], 3, seed=1)
    assert a == [] and c == []


def test_kmeans_k_larger_than_n():
    v = [{"a": 1.0}, {"b": 1.0}]
    a, c = kmeans(v, 5, seed=1)
    assert set(a) == {0, 1}
    assert len(c) == 2


def test_kmeans_k_one():
    v = [{"a": 1.0}, {"b": 1.0}]
    a, c = kmeans(v, 1, seed=1)
    assert a == [0, 0]
    assert len(c) == 1


def test_kmeans_separates_obvious_topics():
    # Topic A docs share 'chs' and 'planning'; Topic B docs share 'risotto'
    # and 'cooking'. Two distinct clusters should result.
    docs = [
        "CHS planning staffing",
        "CHS planning budget",
        "CHS planning quarterly",
        "CHS planning program",
        "risotto cooking arborio",
        "risotto cooking mushroom",
        "risotto cooking italian",
        "risotto cooking technique",
    ]
    vectors, _, _ = build_tfidf(docs, min_df=1)
    a, _ = kmeans(vectors, k=2, seed=42)
    clusters = _cluster_sets(a)
    a_indices = {0, 1, 2, 3}
    b_indices = {4, 5, 6, 7}
    assert any(c == a_indices for c in clusters)
    assert any(c == b_indices for c in clusters)


def test_kmeans_deterministic_with_fixed_seed():
    docs = [
        "CHS staffing plan",
        "Risotto recipe arborio rice",
        "CHS quarterly budget",
        "Italian cooking tips risotto",
        "CHS roadmap planning",
        "Pasta sauce Italian",
    ]
    vectors, _, _ = build_tfidf(docs, min_df=1)
    a1, _ = kmeans(vectors, k=2, seed=42)
    a2, _ = kmeans(vectors, k=2, seed=42)
    assert _cluster_sets(a1) == _cluster_sets(a2)


def test_cosine_similarity_of_identical_vectors_is_one():
    v = {"chs": 0.6, "staffing": 0.8}
    # Normalize it: sqrt(0.36 + 0.64) = 1.0
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9
