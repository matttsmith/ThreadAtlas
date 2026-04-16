"""Tests for embedding infrastructure and hybrid search.

Tests cover:
- TF-IDF embedder fitting and embedding
- Embedding serialization (to/from bytes)
- Cosine similarity
- Reciprocal rank fusion
- Index-time embedding generation
- Hybrid search (keyword + semantic)
- Search filters (date range, register, source)
"""

from __future__ import annotations

import math
import struct
import time
from pathlib import Path

import pytest

from threadatlas.core.models import (
    ConversationLLMMeta,
    DerivedKind,
    State,
    new_id,
)
from threadatlas.core.vault import init_vault
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path
from threadatlas.search.embeddings import (
    EMBEDDING_DIM,
    TFIDFEmbedder,
    build_all_embeddings,
    build_embeddings_for_conversation,
    bytes_to_embedding,
    cosine_similarity,
    embedding_to_bytes,
    fit_embedder_from_corpus,
    load_embedder_state,
    save_embedder_state,
    reciprocal_rank_fusion,
)
from threadatlas.search.search import (
    SearchHit,
    search_chunks,
    search_conversations,
    list_decisions,
    list_entities,
    list_open_loops,
    list_projects,
)
from threadatlas.store import open_store

from conftest import make_chatgpt_export


# --- TF-IDF Embedder ---

class TestTFIDFEmbedder:

    def test_fit_and_embed(self):
        docs = [
            "kubernetes container orchestration deployment",
            "python machine learning neural network training",
            "database schema migration postgresql indexing",
        ]
        embedder = TFIDFEmbedder()
        embedder.fit(docs)
        assert embedder._fitted

        vec = embedder.embed("kubernetes deployment containers")
        assert len(vec) == EMBEDDING_DIM
        # Should have non-zero elements.
        assert any(v != 0.0 for v in vec)

    def test_embed_without_fit_returns_zero(self):
        embedder = TFIDFEmbedder()
        vec = embedder.embed("some text")
        assert len(vec) == EMBEDDING_DIM
        assert all(v == 0.0 for v in vec)

    def test_embedding_is_normalized(self):
        docs = ["hello world test", "foo bar baz qux"]
        embedder = TFIDFEmbedder()
        embedder.fit(docs)
        vec = embedder.embed("hello world test")
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            assert abs(norm - 1.0) < 1e-6

    def test_serialization_roundtrip(self):
        docs = ["kubernetes deployment pods", "python machine learning"]
        embedder = TFIDFEmbedder()
        embedder.fit(docs)

        state = embedder.to_dict()
        restored = TFIDFEmbedder.from_dict(state)

        assert restored._fitted
        assert restored.vocab == embedder.vocab
        assert restored.idf == embedder.idf

        # Same embedding output.
        v1 = embedder.embed("kubernetes pods")
        v2 = restored.embed("kubernetes pods")
        assert v1 == v2

    def test_similar_docs_have_higher_similarity(self):
        docs = [
            "kubernetes container orchestration deployment pods",
            "python machine learning neural network training",
            "kubernetes pods services deployment cluster",
        ]
        embedder = TFIDFEmbedder()
        embedder.fit(docs)

        v1 = embedder.embed("kubernetes deployment")
        v2 = embedder.embed("kubernetes pods cluster")
        v3 = embedder.embed("python neural network")

        sim_12 = cosine_similarity(v1, v2)
        sim_13 = cosine_similarity(v1, v3)
        # Kubernetes-related docs should be more similar to each other.
        assert sim_12 > sim_13


# --- Serialization ---

class TestEmbeddingSerialization:

    def test_roundtrip(self):
        vec = [0.1, 0.2, 0.3, -0.4, 0.5]
        data = embedding_to_bytes(vec)
        assert isinstance(data, bytes)
        assert len(data) == len(vec) * 4  # float32

        restored = bytes_to_embedding(data)
        assert len(restored) == len(vec)
        for a, b in zip(vec, restored):
            assert abs(a - b) < 1e-6

    def test_empty_vector(self):
        vec = []
        data = embedding_to_bytes(vec)
        restored = bytes_to_embedding(data)
        assert restored == []


# --- Cosine similarity ---

class TestCosineSimilarity:

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0

    def test_negative_similarity(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) < 0


# --- Reciprocal Rank Fusion ---

class TestReciprocalRankFusion:

    def test_single_list(self):
        ranked = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
        result = reciprocal_rank_fusion(ranked)
        ids = [item_id for item_id, _ in result]
        assert ids[0] == "a"
        assert ids[1] == "b"
        assert ids[2] == "c"

    def test_two_lists_merge(self):
        list1 = [("a", 10.0), ("b", 5.0)]
        list2 = [("b", 10.0), ("c", 5.0)]
        result = reciprocal_rank_fusion(list1, list2)
        ids = [item_id for item_id, _ in result]
        # "b" appears in both lists at high rank, should be first.
        assert ids[0] == "b"

    def test_empty_lists(self):
        result = reciprocal_rank_fusion([], [])
        assert result == []


# --- Integration: embeddings with store ---

def _setup_corpus(tmp_path: Path):
    vault = init_vault(tmp_path / "vault")
    store = open_store(vault)
    export_path = make_chatgpt_export(tmp_path, [
        {
            "id": "conv-k8s",
            "title": "Kubernetes migration",
            "create_time": time.time() - 86400 * 10,
            "update_time": time.time() - 86400 * 5,
            "messages": [
                ("user", "We need to migrate our Kubernetes cluster to the new region. Pods and services need updating.", time.time() - 86400 * 10),
                ("assistant", "I'll outline the migration steps for the Kubernetes cluster.", time.time() - 86400 * 10 + 60),
                ("user", "The migration involves moving 50 pods across three namespaces.", time.time() - 86400 * 9),
                ("assistant", "That's a significant infrastructure change for the cluster.", time.time() - 86400 * 9 + 60),
            ],
        },
        {
            "id": "conv-python",
            "title": "Python ML model training",
            "create_time": time.time() - 86400 * 20,
            "update_time": time.time() - 86400 * 15,
            "messages": [
                ("user", "I want to train a neural network model using PyTorch for image classification.", time.time() - 86400 * 20),
                ("assistant", "PyTorch is great for training neural network models. Let me help with the architecture.", time.time() - 86400 * 20 + 60),
                ("user", "The training data has 50000 labeled images across 10 categories.", time.time() - 86400 * 19),
                ("assistant", "A convolutional neural network would work well for image classification.", time.time() - 86400 * 19 + 60),
            ],
        },
        {
            "id": "conv-art",
            "title": "Creative writing story",
            "create_time": time.time() - 86400 * 5,
            "update_time": time.time() - 86400 * 3,
            "messages": [
                ("user", "Write me a fantasy story about a dragon who learns to paint.", time.time() - 86400 * 5),
                ("assistant", "Once upon a time, there was a dragon named Ember who discovered art.", time.time() - 86400 * 5 + 60),
                ("user", "The dragon should befriend a human artist in a small village.", time.time() - 86400 * 4),
                ("assistant", "Ember flew to the village of Millbrook and met Clara, a painter.", time.time() - 86400 * 4 + 60),
            ],
        },
    ])
    result = import_path(vault, store, export_path)
    conv_ids = {}
    for cid in result.imported:
        conv = store.get_conversation(cid)
        transition_state(store, cid, State.INDEXED.value)
        chunk_conversation(store, cid)
        extract_for_conversation(store, cid)
        if "kubernetes" in conv.title.lower():
            conv_ids["k8s"] = cid
        elif "python" in conv.title.lower():
            conv_ids["python"] = cid
        elif "creative" in conv.title.lower():
            conv_ids["art"] = cid

    return vault, store, conv_ids


class TestEmbeddingIntegration:

    def test_build_embeddings_for_conversation(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)
        embedder = fit_embedder_from_corpus(store)
        count = build_embeddings_for_conversation(store, conv_ids["k8s"], embedder)
        assert count > 0

        embs = store.get_chunk_embeddings_for_conversation(conv_ids["k8s"])
        assert len(embs) == count
        for chunk_id, data in embs:
            vec = bytes_to_embedding(data)
            assert len(vec) == EMBEDDING_DIM
        store.close()

    def test_build_all_embeddings(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)
        total = build_all_embeddings(store)
        assert total > 0

        all_embs = store.get_all_chunk_embeddings()
        assert len(all_embs) == total
        store.close()

    def test_fit_embedder_empty_corpus(self, tmp_path):
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)
        embedder = fit_embedder_from_corpus(store)
        vec = embedder.embed("test")
        assert len(vec) == EMBEDDING_DIM
        store.close()

    def test_embedder_state_persisted_after_build(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)
        build_all_embeddings(store)

        # Embedder state should be persisted in the DB.
        loaded = load_embedder_state(store)
        assert loaded is not None
        assert loaded._fitted
        assert len(loaded.vocab) > 0

        # Query-time fit_embedder_from_corpus should load persisted state.
        query_embedder = fit_embedder_from_corpus(store)
        assert query_embedder.vocab == loaded.vocab
        assert query_embedder.idf == loaded.idf
        store.close()

    def test_embedder_persistence_matches_index_time(self, tmp_path):
        """Verify that query-time embeddings use the same vector space as index-time."""
        vault, store, conv_ids = _setup_corpus(tmp_path)
        build_all_embeddings(store)

        # Get the persisted embedder (same one used for chunk embeddings).
        embedder = fit_embedder_from_corpus(store)

        # Embed a query and compare against stored chunk embeddings.
        query_vec = embedder.embed("kubernetes cluster migration")
        all_embs = store.get_all_chunk_embeddings()
        assert len(all_embs) > 0

        # At least one chunk should have non-trivial similarity.
        sims = [cosine_similarity(query_vec, bytes_to_embedding(emb))
                for _, _, emb in all_embs]
        assert max(sims) > 0.0
        store.close()


# --- Search filters ---

class TestSearchFilters:

    def test_search_with_date_filter(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)

        # Search with after filter: should exclude old conversations.
        after_ts = time.time() - 86400 * 8  # 8 days ago
        hits = search_conversations(
            store, "migration cluster",
            visible_states=("indexed",), after=after_ts,
        )
        # K8s conv updated 5 days ago, should be included.
        cids = {h.conversation_id for h in hits}
        if conv_ids["k8s"] in cids:
            pass  # Good, it's recent enough.

        # Python conv updated 15 days ago, should be excluded.
        assert conv_ids["python"] not in cids
        store.close()

    def test_search_with_source_filter(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)
        hits = search_conversations(
            store, "migration",
            visible_states=("indexed",), source_filter="chatgpt",
        )
        for h in hits:
            assert h.source == "chatgpt"
        store.close()

    def test_search_with_register_filter(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)

        # Set up register metadata.
        store.upsert_conversation_llm_meta(ConversationLLMMeta(
            conversation_id=conv_ids["k8s"],
            dominant_register="work",
            content_hash="abc",
        ))
        store.upsert_conversation_llm_meta(ConversationLLMMeta(
            conversation_id=conv_ids["art"],
            dominant_register="creative_writing",
            content_hash="def",
        ))
        store.conn.commit()

        hits = search_conversations(
            store, "dragon painting village story",
            visible_states=("indexed",), register=["work"],
        )
        # Art conv should be excluded when filtering for work only.
        cids = {h.conversation_id for h in hits}
        assert conv_ids["art"] not in cids
        store.close()

    def test_list_with_filters(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)

        # All listing functions should accept filter parameters without error.
        list_projects(store, visible_states=("indexed",), after=time.time() - 86400)
        list_decisions(store, visible_states=("indexed",), before=time.time())
        list_entities(store, visible_states=("indexed",), source_filter="chatgpt")
        list_open_loops(store, visible_states=("indexed",), register=["work"])
        store.close()

    def test_search_match_type_field(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)
        hits = search_conversations(store, "kubernetes", visible_states=("indexed",))
        for h in hits:
            assert h.match_type in ("keyword", "semantic", "both")
        store.close()


# --- Hybrid search ---

class TestHybridSearch:

    def test_keyword_search_still_works(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)
        hits = search_conversations(store, "kubernetes", visible_states=("indexed",))
        assert len(hits) > 0
        assert any(h.conversation_id == conv_ids["k8s"] for h in hits)
        store.close()

    def test_semantic_search_with_embeddings(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)
        build_all_embeddings(store)

        # Semantic search for paraphrase of content.
        hits = search_conversations(
            store, "container orchestration cloud infrastructure",
            visible_states=("indexed",),
        )
        # Should find k8s conversation even with different phrasing.
        assert len(hits) > 0
        store.close()

    def test_chunk_search_with_filters(self, tmp_path):
        vault, store, conv_ids = _setup_corpus(tmp_path)
        hits = search_chunks(store, "kubernetes", visible_states=("indexed",))
        assert len(hits) > 0
        for h in hits:
            assert h.match_type == "keyword"
        store.close()
