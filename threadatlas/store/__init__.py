"""Persistence layer: SQLite metadata + FTS5 + normalized file IO."""

from .db import Store, open_store, transaction
from .normalized import write_normalized, read_normalized, delete_normalized

__all__ = [
    "Store",
    "open_store",
    "transaction",
    "write_normalized",
    "read_normalized",
    "delete_normalized",
]
