"""Canonical domain models for ThreadAtlas.

The model is intentionally small. Fields exist either to support
search/synthesis, to support deletion correctness, or to support audit. Fields
that exist purely because they "might be analytically interesting" are out.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import uuid


def new_id(prefix: str = "id") -> str:
    """Generate a stable, vault-local identifier with a small type prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class State(str, Enum):
    """Visibility / lifecycle state for a conversation.

    The value strings are persisted to SQLite, so do not rename casually.
    """

    PENDING_REVIEW = "pending_review"
    INDEXED = "indexed"
    PRIVATE = "private"
    QUARANTINED = "quarantined"
    # NOTE: ``DELETED`` is a transient marker only. Hard delete physically
    # removes records; we never keep tombstones in the conversations table.
    DELETED = "deleted"


# States that are eligible for MCP exposure.
MCP_VISIBLE_STATES = frozenset({State.INDEXED.value})

# States that may participate in global synthesis / project pages.
SYNTHESIS_STATES = frozenset({State.INDEXED.value})

# States that may have chunks, embeddings, and extracted derived objects.
EXTRACTABLE_STATES = frozenset({State.INDEXED.value, State.PRIVATE.value})

# States whose content may be present in the FTS5 indexes at all. This is the
# narrowest set and the ground truth for "searchable surface exists" -
# ``pending_review`` and ``quarantined`` content must never have FTS rows, even
# if a caller accidentally passes their state in ``visible_states``.
FTS_INDEXED_STATES = frozenset({State.INDEXED.value, State.PRIVATE.value})


class Source(str, Enum):
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    OTHER = "other"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    OTHER = "other"


class DerivedKind(str, Enum):
    PROJECT = "project"
    ENTITY = "entity"
    DECISION = "decision"
    OPEN_LOOP = "open_loop"
    ARTIFACT = "artifact"
    PREFERENCE = "preference"


@dataclass
class Message:
    message_id: str
    conversation_id: str
    ordinal: int
    role: str  # one of Role values
    content_text: str
    timestamp: float | None = None  # POSIX seconds, may be missing
    content_structured: dict | None = None
    source_message_id: str | None = None
    visibility_state_inherited: str = State.PENDING_REVIEW.value


@dataclass
class Conversation:
    conversation_id: str
    source: str  # one of Source values
    title: str
    created_at: float | None
    updated_at: float | None
    imported_at: float
    state: str = State.PENDING_REVIEW.value
    source_conversation_id: str | None = None
    source_export_fingerprint: str | None = None
    message_count: int = 0
    summary_short: str = ""
    summary_long: str | None = None
    # Provenance of the summary: "deterministic" (default) or "llm". Lets
    # operators filter or re-run summarization selectively.
    summary_source: str = "deterministic"
    manual_tags: list[str] = field(default_factory=list)
    auto_tags: list[str] = field(default_factory=list)
    primary_project_id: str | None = None
    importance_score: float = 0.0
    resurfacing_score: float = 0.0
    has_open_loops: bool = False
    schema_version: int = 1
    parser_version: int = 1
    notes_local: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    conversation_id: str
    chunk_index: int
    start_message_ordinal: int
    end_message_ordinal: int
    chunk_title: str = ""
    summary_short: str = ""
    project_id: str | None = None
    importance_score: float = 0.0
    has_open_loops: bool = False


@dataclass
class DerivedObject:
    object_id: str
    kind: str  # one of DerivedKind values
    title: str
    description: str = ""
    project_id: str | None = None
    state: str = "active"  # active | suppressed
    canonical_key: str = ""  # used for dedupe within a kind, e.g. lower(title)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class ProvenanceLink:
    """Connects a derived object to a source conversation/chunk and an excerpt.

    Excerpts are stored short to keep the database compact; full text always
    lives in the messages/chunks tables.
    """

    link_id: str
    object_id: str
    conversation_id: str
    chunk_id: str | None
    excerpt: str
    created_at: float
