"""Deterministic clustering layer.

Pure-Python TF-IDF + k-means. Stdlib only. No optional dependencies.
"""

from .tfidf import tokenize, build_tfidf, distinctive_terms
from .kmeans import kmeans, cosine_similarity
from .groups import regroup_all, GroupingResult

__all__ = [
    "tokenize",
    "build_tfidf",
    "distinctive_terms",
    "kmeans",
    "cosine_similarity",
    "regroup_all",
    "GroupingResult",
]
