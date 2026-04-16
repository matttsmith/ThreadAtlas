"""Search and retrieval."""

from .search import (
    SearchHit,
    search_conversations,
    search_chunks,
    list_open_loops,
    list_decisions,
    list_entities,
    project_view,
    project_timeline,
)

from .query_engine import (
    QueryFilter,
    QueryHit,
    QueryResult,
    parse_query,
    query,
)

from .embeddings import (
    TFIDFEmbedder,
    build_all_embeddings,
    build_embeddings_for_conversation,
    cosine_similarity,
    embedding_to_bytes,
    bytes_to_embedding,
    fit_embedder_from_corpus,
    load_embedder_state,
    save_embedder_state,
    reciprocal_rank_fusion,
)

__all__ = [
    "SearchHit",
    "search_conversations",
    "search_chunks",
    "list_open_loops",
    "list_decisions",
    "list_entities",
    "project_view",
    "project_timeline",
    "QueryFilter",
    "QueryHit",
    "QueryResult",
    "parse_query",
    "query",
    "TFIDFEmbedder",
    "build_all_embeddings",
    "build_embeddings_for_conversation",
    "cosine_similarity",
    "embedding_to_bytes",
    "bytes_to_embedding",
    "fit_embedder_from_corpus",
    "reciprocal_rank_fusion",
]
