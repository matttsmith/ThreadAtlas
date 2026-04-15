"""TF-IDF correctness + stability tests."""

from __future__ import annotations

import math

from threadatlas.cluster.tfidf import build_tfidf, distinctive_terms, tokenize


def test_tokenize_drops_stopwords_and_short_tokens():
    toks = tokenize("The CHS staffing plan is about next quarter.")
    assert "the" not in toks
    assert "is" not in toks
    assert "chs" in toks
    assert "staffing" in toks
    assert "quarter" in toks


def test_tfidf_vectors_are_unit_length():
    docs = [
        "CHS staffing plan",
        "CHS quarterly numbers",
        "Recipe for risotto",
        "Italian cooking risotto rice",
    ]
    vectors, vocab, idf = build_tfidf(docs, min_df=1)
    for v in vectors:
        norm = math.sqrt(sum(w * w for w in v.values()))
        assert norm == 0 or abs(norm - 1.0) < 1e-9


def test_tfidf_is_deterministic():
    docs = [
        "CHS staffing plan",
        "Risotto recipe arborio rice",
        "CHS quarterly budget",
        "Italian cooking tips risotto",
    ]
    v1, _, _ = build_tfidf(docs, min_df=1)
    v2, _, _ = build_tfidf(docs, min_df=1)
    assert v1 == v2


def test_tfidf_vocab_pruning_by_min_df():
    docs = [
        "alpha beta beta gamma",
        "alpha delta",
        "alpha epsilon",
    ]
    # 'alpha' has df=3, 'beta' df=1, 'gamma' df=1, 'delta' df=1, 'epsilon' df=1
    _, vocab, _ = build_tfidf(docs, min_df=2)
    assert "alpha" in vocab
    assert "beta" not in vocab
    assert "delta" not in vocab


def test_distinctive_terms_prefers_cluster_specific():
    cluster = {"chs": 0.8, "staffing": 0.5, "the": 0.1, "quarter": 0.3}
    other = {"chs": 0.05, "staffing": 0.05, "the": 0.1, "quarter": 0.05, "risotto": 0.7}
    terms = distinctive_terms(cluster, [other], top_k=3)
    # "chs" should score highest; "the" should score low because it's equally common.
    assert terms[0] == "chs"
    assert "the" not in terms
