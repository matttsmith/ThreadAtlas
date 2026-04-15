"""Disaster recovery: rebuild the DB from ``vault/normalized/*.json``.

The normalized JSON files are the on-disk source of truth. If the
SQLite DB is lost or corrupted, this command reads every normalized
file back into a fresh DB.

What's rebuilt:
  * conversations, messages
  * chunks (via re-run of the deterministic chunker for eligible states)
  * FTS indexes
  * extraction (via re-run of heuristics for eligible states)

What's NOT automatically rebuilt:
  * LLM-generated summaries (the normalized files store the summary the
    conversation had at last write, so they come back if they were
    persisted there)
  * thematic groups (run ``threadatlas group`` after)

The old DB file is moved aside to ``vault/db/threadatlas.sqlite3.bak.<ts>``
before the rebuild, so nothing is silently destroyed.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .core.models import Conversation, Message, State
from .core.vault import Vault
from .extract import chunk_conversation, extract_for_conversation
from .store import Store, open_store


@dataclass
class RebuildResult:
    conversations_restored: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    chunks_built: int = 0
    extraction_ran: int = 0
    backup_path: Path | None = None


def _load_normalized_files(vault: Vault) -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []
    for p in vault.normalized.rglob("*.json"):
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out


def _conv_from_payload(payload: dict) -> Conversation:
    c = payload["conversation"]
    return Conversation(
        conversation_id=c["conversation_id"],
        source=c["source"],
        source_conversation_id=c.get("source_conversation_id"),
        source_export_fingerprint=c.get("source_export_fingerprint"),
        title=c.get("title") or "Untitled",
        created_at=c.get("created_at"),
        updated_at=c.get("updated_at"),
        imported_at=c.get("imported_at") or time.time(),
        state=c.get("state") or State.PENDING_REVIEW.value,
        message_count=int(c.get("message_count") or 0),
        summary_short=c.get("summary_short") or "",
        summary_long=c.get("summary_long"),
        manual_tags=c.get("manual_tags") or [],
        auto_tags=c.get("auto_tags") or [],
        primary_project_id=c.get("primary_project_id"),
        importance_score=float(c.get("importance_score") or 0.0),
        resurfacing_score=float(c.get("resurfacing_score") or 0.0),
        has_open_loops=bool(c.get("has_open_loops")),
        schema_version=int(c.get("schema_version") or 1),
        parser_version=int(c.get("parser_version") or 1),
        notes_local=c.get("notes_local") or "",
    )


def _msgs_from_payload(payload: dict, conversation_id: str, state: str) -> list[Message]:
    out: list[Message] = []
    for m in payload.get("messages", []):
        out.append(Message(
            message_id=m.get("message_id") or f"msg_{conversation_id}_{m.get('ordinal', 0)}",
            conversation_id=conversation_id,
            ordinal=int(m.get("ordinal", 0)),
            role=m.get("role") or "other",
            content_text=m.get("content_text") or "",
            timestamp=m.get("timestamp"),
            content_structured=m.get("content_structured"),
            source_message_id=m.get("source_message_id"),
            visibility_state_inherited=state,
        ))
    return out


def rebuild_from_normalized(vault: Vault) -> RebuildResult:
    """Rebuild the DB from normalized JSON files. Non-destructive to normalized files."""
    result = RebuildResult()

    # Backup the current DB.
    if vault.db_path.exists():
        backup = vault.db_path.with_name(
            vault.db_path.name + f".bak.{int(time.time())}"
        )
        shutil.copy2(vault.db_path, backup)
        result.backup_path = backup

    # Wipe the existing DB so we start clean.
    if vault.db_path.exists():
        vault.db_path.unlink()
    # WAL sidecars:
    for suffix in ("-wal", "-shm"):
        p = vault.db_path.with_name(vault.db_path.name + suffix)
        if p.exists():
            p.unlink()

    store = open_store(vault)
    try:
        payloads = _load_normalized_files(vault)
        for path, payload in payloads:
            try:
                conv = _conv_from_payload(payload)
                msgs = _msgs_from_payload(payload, conv.conversation_id, conv.state)
                store.insert_conversation(conv)
                if msgs:
                    store.insert_messages(msgs)
                result.conversations_restored += 1
            except Exception as e:
                result.skipped.append((str(path), repr(e)))
        store.conn.commit()

        # Re-run chunking + extraction for eligible states, rebuild FTS.
        from .core.models import EXTRACTABLE_STATES
        rows = store.conn.execute(
            f"SELECT conversation_id FROM conversations WHERE state IN ({','.join(['?']*len(EXTRACTABLE_STATES))})",
            tuple(EXTRACTABLE_STATES),
        ).fetchall()
        for r in rows:
            cid = r["conversation_id"]
            chunks = chunk_conversation(store, cid)
            result.chunks_built += len(chunks)
            extract_for_conversation(store, cid)
            result.extraction_ran += 1
        store.rebuild_all_fts()
        store.conn.commit()
    finally:
        store.close()
    return result
