"""Extraction layer.

Two responsibilities:

* :mod:`.chunking` - split a conversation into thematic chunks
* :mod:`.heuristics` - extract durable derived objects (projects, decisions,
  open loops, etc) without an LLM

Both are deterministic and offline by design.
"""

from .chunking import chunk_conversation, chunk_all_eligible
from .heuristics import extract_for_conversation, extract_all_eligible

__all__ = [
    "chunk_conversation",
    "chunk_all_eligible",
    "extract_for_conversation",
    "extract_all_eligible",
]
