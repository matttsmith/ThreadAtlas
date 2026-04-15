"""Import pipeline.

Glues a parser to the SQLite store + normalized vault. Lands every imported
conversation in ``pending_review``. Conservatively dedupes by
``source_export_fingerprint`` (per-conversation hash, not per-archive) so
re-importing the same export does not create duplicates.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.models import Conversation, Message, State, new_id
from ..core.vault import Vault
from ..store import Store, transaction, write_normalized
from .base import ParsedConversation, Parser, registry


def _get_parser(source: str) -> Parser:
    if source == "auto":
        return registry.get_autodetect()
    return registry.get(source)


@dataclass
class ImportResult:
    imported: list[str] = field(default_factory=list)
    deduped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    raw_path: Path | None = None

    @property
    def total(self) -> int:
        return len(self.imported) + len(self.deduped) + len(self.failed)


def _fingerprint_conversation(parsed: ParsedConversation) -> str:
    """Stable per-conversation hash: source + source_id + content snapshot.

    We include message ids and a hash of the joined content so that an
    edited (but same-source-id) conversation is still recognised as a new
    record.
    """
    h = hashlib.sha256()
    h.update((parsed.source or "").encode())
    h.update(b"\x00")
    h.update((parsed.source_conversation_id or "").encode())
    h.update(b"\x00")
    h.update((parsed.title or "").encode())
    h.update(b"\x00")
    for m in parsed.messages:
        h.update((m.source_message_id or "").encode())
        h.update(b"|")
        h.update((m.role or "").encode())
        h.update(b"|")
        h.update(m.content_text.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()


def _copy_raw_input(vault: Vault, src: Path) -> Path | None:
    """Copy the raw export into ``raw_imports/`` for provenance.

    Skips the copy if the path is already inside the vault (e.g., during
    tests).
    """
    src = src.resolve()
    if not src.exists():
        return None
    if vault.root in src.parents or src == vault.root:
        return src
    vault.raw_imports.mkdir(parents=True, exist_ok=True)
    base = f"{int(time.time())}_{src.name}"
    dest = vault.raw_imports / base
    # Disambiguate if the same name already exists (re-imports, fast loops).
    suffix = 1
    while dest.exists():
        dest = vault.raw_imports / f"{base}_{suffix}"
        suffix += 1
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    return dest


def import_path(
    vault: Vault,
    store: Store,
    path: Path,
    *,
    source: str = "auto",
    copy_raw: bool = True,
) -> ImportResult:
    """Run a full import. The path may be a file, directory, or zip archive."""
    parser: Parser = _get_parser(source)
    raw_dest = _copy_raw_input(vault, path) if copy_raw else None
    result = ImportResult(raw_path=raw_dest)

    for parsed in parser.iter_conversations(path):
        try:
            cid = _import_one(vault, store, parsed)
            if cid is None:
                result.deduped.append(parsed.source_conversation_id or parsed.title)
            else:
                result.imported.append(cid)
        except Exception as e:  # pragma: no cover - defensive
            result.failed.append((parsed.title or "?", repr(e)))
    return result


def _import_one(vault: Vault, store: Store, parsed: ParsedConversation) -> str | None:
    fingerprint = _fingerprint_conversation(parsed)
    existing = store.find_conversation_by_fingerprint(fingerprint)
    if existing:
        return None  # exact duplicate

    conv_id = new_id("conv")
    now = time.time()
    conv = Conversation(
        conversation_id=conv_id,
        source=parsed.source,
        source_conversation_id=parsed.source_conversation_id or None,
        source_export_fingerprint=fingerprint,
        title=parsed.title or "Untitled",
        created_at=parsed.created_at,
        updated_at=parsed.updated_at,
        imported_at=now,
        state=State.PENDING_REVIEW.value,
        message_count=parsed.message_count,
    )

    messages: list[Message] = []
    for pm in parsed.messages:
        messages.append(Message(
            message_id=new_id("msg"),
            conversation_id=conv_id,
            ordinal=pm.ordinal,
            role=pm.role,
            content_text=pm.content_text,
            timestamp=pm.timestamp,
            content_structured=pm.content_structured,
            source_message_id=pm.source_message_id,
            visibility_state_inherited=conv.state,
        ))

    with transaction(store):
        store.insert_conversation(conv)
        if messages:
            store.insert_messages(messages)
        # No FTS for pending_review (kept out of search indexes by design).

    write_normalized(vault, conv, messages)
    return conv_id
