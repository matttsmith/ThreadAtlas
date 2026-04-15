"""Import pipeline.

Glues a parser to the SQLite store + normalized vault. Default initial
state is ``pending_review``. Two escape hatches:

* ``auto_rules.json`` in the vault may DOWN-classify matching
  conversations to ``private`` or ``quarantined`` at import time.
* ``--auto-approve`` lifts the default initial state to ``indexed`` for
  conversations that DIDN'T match any auto-rule. Rules always win over
  auto-approve: a sensitive thread never transits through ``indexed``.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.models import (
    EXTRACTABLE_STATES,
    Conversation,
    Message,
    State,
    new_id,
)
from ..core.vault import Vault
from ..rules import RuleSet, evaluate, load_rules, summarize_matches
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
    empty_skipped: list[str] = field(default_factory=list)
    by_source: dict[str, int] = field(default_factory=dict)
    # Initial-state histogram: counts how many conversations landed in each
    # state after auto-rules + auto-approve. Keys are State values.
    by_initial_state: dict[str, int] = field(default_factory=dict)
    auto_rule_matches: int = 0
    pending_review_count_after: int = 0
    raw_path: Path | None = None

    @property
    def total(self) -> int:
        return (len(self.imported) + len(self.deduped)
                + len(self.failed) + len(self.empty_skipped))


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
    skip_empty: bool = True,
    auto_approve: bool = False,
) -> ImportResult:
    """Run a full import. The path may be a file, directory, or zip archive.

    Initial-state selection per conversation:
      1. Load ``<vault>/auto_rules.json`` (may be empty).
      2. Evaluate each conversation against the ruleset. If anything
         matches, set the state to ``private`` or ``quarantined``.
      3. Otherwise, if ``auto_approve`` is True, set state to ``indexed``.
      4. Otherwise, default ``pending_review``.

    Conversations with zero messages are reported as ``empty_skipped``
    (override with ``skip_empty=False``).
    """
    parser: Parser = _get_parser(source)
    raw_dest = _copy_raw_input(vault, path) if copy_raw else None
    result = ImportResult(raw_path=raw_dest)

    try:
        ruleset = load_rules(vault.root)
    except ValueError:
        # Fail loudly: refusing to import keeps sensitive content out of
        # a wrongly-classified indexed state.
        raise

    try:
        stream = list(parser.iter_conversations(path))
    except Exception as e:
        result.failed.append(("<parser>", repr(e)))
        stream = []

    for parsed in stream:
        try:
            if skip_empty and parsed.message_count == 0:
                result.empty_skipped.append(
                    parsed.title or parsed.source_conversation_id or "?"
                )
                continue
            cid, initial_state, matched = _import_one(
                vault, store, parsed, ruleset=ruleset, auto_approve=auto_approve,
            )
            if cid is None:
                result.deduped.append(parsed.source_conversation_id or parsed.title)
            else:
                result.imported.append(cid)
                result.by_source[parsed.source] = result.by_source.get(parsed.source, 0) + 1
                result.by_initial_state[initial_state] = (
                    result.by_initial_state.get(initial_state, 0) + 1
                )
                if matched:
                    result.auto_rule_matches += 1
        except Exception as e:
            result.failed.append((parsed.title or "?", repr(e)))

    result.pending_review_count_after = store.conn.execute(
        "SELECT COUNT(*) AS c FROM conversations WHERE state = ?",
        (State.PENDING_REVIEW.value,),
    ).fetchone()["c"]
    return result


def _choose_initial_state(
    parsed: ParsedConversation,
    ruleset: RuleSet,
    auto_approve: bool,
) -> tuple[str, str]:
    """Decide the initial state + any notes_local text from rule matches."""
    target, matches = evaluate(
        ruleset,
        title=parsed.title,
        summary="",
        messages=[m.content_text for m in parsed.messages],
    )
    notes = summarize_matches(matches)
    if target is not None:
        return target, notes
    if auto_approve:
        return State.INDEXED.value, notes
    return State.PENDING_REVIEW.value, notes


def _import_one(
    vault: Vault,
    store: Store,
    parsed: ParsedConversation,
    *,
    ruleset: RuleSet,
    auto_approve: bool,
) -> tuple[str | None, str, bool]:
    """Import a single conversation.

    Returns ``(conv_id | None, initial_state, auto_rule_matched)``.
    ``conv_id`` is None for an exact duplicate.
    """
    fingerprint = _fingerprint_conversation(parsed)
    existing = store.find_conversation_by_fingerprint(fingerprint)
    if existing:
        return None, existing.state, False

    initial_state, notes = _choose_initial_state(parsed, ruleset, auto_approve)
    matched = bool(notes)

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
        state=initial_state,
        message_count=parsed.message_count,
        notes_local=notes,
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
            visibility_state_inherited=initial_state,
        ))

    with transaction(store):
        store.insert_conversation(conv)
        if messages:
            store.insert_messages(messages)
        # Index FTS for states that allow it (indexed + private).
        if initial_state in EXTRACTABLE_STATES:
            store.reindex_conversation_fts(conv_id)

    write_normalized(vault, conv, messages)
    return conv_id, initial_state, matched
