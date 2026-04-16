"""Two-pass LLM extraction pipeline (v2).

Pass 1: Per-turn classification (register + reality_mode).
Pass 2: Per-conversation extraction (projects, decisions, open_loops, entities, summary).

This runs at index time, not query time. Results are cached in the database.
Only processes conversations in EXTRACTABLE_STATES.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass

from ..core.models import (
    EXTRACTABLE_STATES,
    ConversationLLMMeta,
    DerivedKind,
    DerivedObject,
    Message,
    MessageClassification,
    ProvenanceLink,
    Register,
    RealityMode,
    new_id,
)
from ..core.vault import Vault
from ..store import Store, transaction
from .common import LLMResponse, parse_json_response
from .prompt_loader import (
    COMBINED_PROMPT,
    EXTRACTION_PROMPT,
    TURN_CLASSIFIER_PROMPT,
    get_prompt_version,
    render_prompt,
)
from .runner import LLMRunner


_JSON_ARRAY_RX = re.compile(r"\[.*\]", re.DOTALL)

VALID_REGISTERS = frozenset(r.value for r in Register)
VALID_REALITY_MODES = frozenset(r.value for r in RealityMode)

# Short conversations (few user/assistant messages) can use a combined
# single-pass prompt instead of two separate LLM calls. This roughly
# halves the wall-clock time for the majority of conversations which tend
# to be short.
SINGLE_PASS_MAX_MESSAGES = 30

# Heuristic keywords for fast register classification without LLM.
# If the title + first few messages match one of these patterns strongly,
# we skip the LLM classification call entirely.
_ROLEPLAY_SIGNALS = frozenset({
    "roleplay", "you are a", "you're a", "i am a", "let's pretend",
    "in character", "stay in character", "your character",
    "act as", "play the role",
})
_JAILBREAK_SIGNALS = frozenset({
    "ignore previous", "ignore all previous", "ignore your instructions",
    "system prompt", "jailbreak", "dan mode", "developer mode",
    "bypass", "pretend you have no restrictions",
})
_CREATIVE_SIGNALS = frozenset({
    "write me a story", "write a story", "write a poem",
    "write me a poem", "write a script", "creative writing",
    "short story about", "fiction about",
})


def _content_hash(messages: list[Message]) -> str:
    h = hashlib.sha256()
    for m in messages:
        h.update(f"{m.ordinal}|{m.role}|{m.content_text or ''}\n".encode("utf-8", errors="replace"))
    return h.hexdigest()


def _render_messages_for_classification(messages: list[Message], max_chars: int = 600) -> str:
    lines = []
    for m in messages:
        if m.role not in ("user", "assistant"):
            continue
        text = (m.content_text or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
        lines.append(json.dumps({"role": m.role, "text": text}))
    return "\n".join(lines)


def _render_messages_with_tags(
    messages: list[Message],
    classifications: dict[str, MessageClassification],
    max_chars: int = 600,
) -> str:
    lines = []
    for m in messages:
        if m.role not in ("user", "assistant"):
            continue
        text = (m.content_text or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
        cls = classifications.get(m.message_id)
        reg = cls.register if cls else "other"
        rm = cls.reality_mode if cls else "literal"
        lines.append(f"[{m.role} | register={reg} | reality_mode={rm}] {text}")
    return "\n".join(lines)


def _parse_json_array(resp: LLMResponse) -> list[dict] | None:
    if not resp.success or not resp.raw:
        return None
    match = _JSON_ARRAY_RX.search(resp.raw)
    if not match:
        return None
    try:
        result = json.loads(match.group(0))
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Fast heuristic pre-classification (no LLM needed)
# ---------------------------------------------------------------------------

def _heuristic_register(title: str, messages: list[Message]) -> str | None:
    """Try to classify register from keywords alone.

    Returns a register string if confident, None if LLM is needed.
    Only fires on high-precision signals to avoid false positives.
    """
    text_sample = (title or "").lower()
    for m in messages[:6]:
        if m.role == "user":
            text_sample += " " + (m.content_text or "").lower()[:500]

    for signal in _JAILBREAK_SIGNALS:
        if signal in text_sample:
            return "jailbreak_experiment"
    for signal in _ROLEPLAY_SIGNALS:
        if signal in text_sample:
            return "roleplay"
    for signal in _CREATIVE_SIGNALS:
        if signal in text_sample:
            return "creative_writing"
    return None


# ---------------------------------------------------------------------------
# Pass 1: Turn classification
# ---------------------------------------------------------------------------

def classify_turns(
    vault: Vault,
    runner: LLMRunner,
    title: str,
    messages: list[Message],
) -> list[MessageClassification]:
    """Classify each message's register and reality_mode via LLM.

    Optimization: if the heuristic pre-classifier is confident about the
    register, we skip the LLM call entirely and assign the heuristic
    result to all messages. This saves ~15-25 seconds per conversation
    for the many conversations that are obviously work/roleplay/etc.
    """
    now = time.time()
    classifiable = [m for m in messages if m.role in ("user", "assistant")]
    prompt_version = get_prompt_version(TURN_CLASSIFIER_PROMPT)

    # Fast path: heuristic classification for obvious cases.
    heuristic_reg = _heuristic_register(title, messages)
    if heuristic_reg is not None:
        rm = "fictional" if heuristic_reg in ("roleplay",) else "literal"
        return [MessageClassification(
            message_id=m.message_id,
            register=heuristic_reg,
            reality_mode=rm,
            prompt_version=f"heuristic+{prompt_version}",
            classified_at=now,
        ) for m in classifiable]

    # LLM path.
    rendered = _render_messages_for_classification(messages)
    prompt = render_prompt(TURN_CLASSIFIER_PROMPT, TITLE=title or "(no title)", MESSAGES=rendered)

    resp = runner.run("turn_classification", prompt)
    parsed = _parse_json_array(resp)

    if parsed and len(parsed) >= len(classifiable):
        classifications = []
        for i, m in enumerate(classifiable):
            item = parsed[i] if i < len(parsed) else {}
            reg = item.get("register", "other")
            rm = item.get("reality_mode", "literal")
            if reg not in VALID_REGISTERS:
                reg = "other"
            if rm not in VALID_REALITY_MODES:
                rm = "literal"
            classifications.append(MessageClassification(
                message_id=m.message_id,
                register=reg,
                reality_mode=rm,
                prompt_version=prompt_version,
                classified_at=now,
            ))
        return classifications

    # Fallback: assign defaults.
    return [MessageClassification(
        message_id=m.message_id,
        register="other",
        reality_mode="literal",
        prompt_version=prompt_version,
        classified_at=now,
    ) for m in classifiable]


# ---------------------------------------------------------------------------
# Pass 2: Per-conversation extraction
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    summary: str
    projects: list[dict]
    decisions: list[dict]
    open_loops: list[dict]
    entities: list[dict]


def extract_conversation(
    vault: Vault,
    runner: LLMRunner,
    title: str,
    messages: list[Message],
    classifications: dict[str, MessageClassification],
    created_at: float | None = None,
    updated_at: float | None = None,
) -> ExtractionResult:
    """Extract structured data from a classified conversation via LLM."""
    rendered = _render_messages_with_tags(messages, classifications)

    def _fmt_ts(ts: float | None) -> str:
        if not ts:
            return "unknown"
        import datetime
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")

    prompt = render_prompt(
        EXTRACTION_PROMPT,
        TITLE=title or "(no title)",
        CREATED=_fmt_ts(created_at),
        UPDATED=_fmt_ts(updated_at),
        MESSAGES_WITH_TAGS=rendered,
    )

    resp = runner.run("extraction", prompt)
    parsed = parse_json_response(resp)

    if parsed:
        return ExtractionResult(
            summary=parsed.get("summary", ""),
            projects=parsed.get("projects", []),
            decisions=parsed.get("decisions", []),
            open_loops=parsed.get("open_loops", []),
            entities=parsed.get("entities", []),
        )

    # Fallback: empty extraction.
    return ExtractionResult(
        summary="",
        projects=[],
        decisions=[],
        open_loops=[],
        entities=[],
    )


# ---------------------------------------------------------------------------
# Combined single-pass mode (for short conversations)
# ---------------------------------------------------------------------------

def _run_combined_pass(
    vault: Vault,
    runner: LLMRunner,
    conv,
    messages: list[Message],
    classifiable: list[Message],
) -> tuple[list[MessageClassification], ExtractionResult]:
    """Run classification + extraction in a single LLM call.

    For conversations with <= SINGLE_PASS_MAX_MESSAGES user/assistant
    messages, this halves the wall-clock time by avoiding two separate
    LLM round-trips.

    If the heuristic pre-classifier fires, we skip the LLM classification
    entirely and only need one call for extraction — so this path only
    saves time when the heuristic doesn't fire.
    """
    now = time.time()
    prompt_version = get_prompt_version(COMBINED_PROMPT)

    # Check heuristic first — if it fires, we only need the extraction call.
    heuristic_reg = _heuristic_register(conv.title, messages)
    if heuristic_reg is not None:
        rm = "fictional" if heuristic_reg in ("roleplay",) else "literal"
        classifications = [MessageClassification(
            message_id=m.message_id,
            register=heuristic_reg,
            reality_mode=rm,
            prompt_version=f"heuristic+{prompt_version}",
            classified_at=now,
        ) for m in classifiable]
        cls_by_id = {c.message_id: c for c in classifications}
        result = extract_conversation(
            vault, runner, conv.title, messages, cls_by_id,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        )
        return classifications, result

    # Combined LLM call.
    rendered = _render_messages_for_classification(messages)

    import datetime as _dt
    def _fmt_ts(ts):
        if not ts:
            return "unknown"
        return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%d")

    prompt = render_prompt(
        COMBINED_PROMPT,
        TITLE=conv.title or "(no title)",
        CREATED=_fmt_ts(conv.created_at),
        UPDATED=_fmt_ts(conv.updated_at),
        MESSAGES=rendered,
    )

    # The combined prompt uses the "extraction" task since it's the more
    # complex of the two; the config needs extraction enabled.
    resp = runner.run("extraction", prompt)
    parsed = parse_json_response(resp)

    if parsed:
        # Parse classifications from combined response.
        raw_cls = parsed.get("classifications", [])
        classifications = []
        for i, m in enumerate(classifiable):
            item = raw_cls[i] if i < len(raw_cls) else {}
            reg = item.get("register", "other")
            rm = item.get("reality_mode", "literal")
            if reg not in VALID_REGISTERS:
                reg = "other"
            if rm not in VALID_REALITY_MODES:
                rm = "literal"
            classifications.append(MessageClassification(
                message_id=m.message_id,
                register=reg,
                reality_mode=rm,
                prompt_version=prompt_version,
                classified_at=now,
            ))

        result = ExtractionResult(
            summary=parsed.get("summary", ""),
            projects=parsed.get("projects", []),
            decisions=parsed.get("decisions", []),
            open_loops=parsed.get("open_loops", []),
            entities=parsed.get("entities", []),
        )
        return classifications, result

    # Fallback: default classifications + empty extraction.
    classifications = [MessageClassification(
        message_id=m.message_id,
        register="other",
        reality_mode="literal",
        prompt_version=prompt_version,
        classified_at=now,
    ) for m in classifiable]
    return classifications, ExtractionResult("", [], [], [], [])


# ---------------------------------------------------------------------------
# Full pipeline: both passes + persistence
# ---------------------------------------------------------------------------

def _find_chunk_for_ordinal(store: Store, conversation_id: str, ordinal: int) -> str | None:
    """Find the chunk containing a given message ordinal."""
    row = store.conn.execute(
        """
        SELECT chunk_id FROM chunks
         WHERE conversation_id = ?
           AND start_message_ordinal <= ?
           AND end_message_ordinal >= ?
         LIMIT 1
        """,
        (conversation_id, ordinal, ordinal),
    ).fetchone()
    return row["chunk_id"] if row else None


def run_pipeline(
    vault: Vault,
    store: Store,
    runner: LLMRunner,
    conversation_id: str,
    *,
    force: bool = False,
) -> dict:
    """Run the full two-pass extraction pipeline for a single conversation.

    Returns a summary dict with counts. Skips if the content hash hasn't
    changed since last extraction (unless ``force=True``).
    """
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise KeyError(conversation_id)
    if conv.state not in EXTRACTABLE_STATES:
        return {"skipped": True, "reason": "ineligible_state"}

    messages = store.list_messages(conversation_id)
    if not messages:
        return {"skipped": True, "reason": "no_messages"}

    # Check content hash for incremental indexing.
    current_hash = _content_hash(messages)
    if not force:
        existing_hash = store.get_conversation_content_hash(conversation_id)
        if existing_hash == current_hash:
            return {"skipped": True, "reason": "unchanged"}

    classifiable = [m for m in messages if m.role in ("user", "assistant")]
    use_combined = len(classifiable) <= SINGLE_PASS_MAX_MESSAGES

    if use_combined:
        # Single-pass: classification + extraction in one LLM call.
        classifications, result = _run_combined_pass(
            vault, runner, conv, messages, classifiable,
        )
    else:
        # Two-pass: separate classification and extraction calls.
        classifications = classify_turns(vault, runner, conv.title, messages)
        cls_by_id = {c.message_id: c for c in classifications}
        result = extract_conversation(
            vault, runner, conv.title, messages, cls_by_id,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        )

    # Compute dominant register from user messages.
    user_registers = [c.register for c in classifications
                      if any(m.message_id == c.message_id and m.role == "user" for m in messages)]
    reg_counts = Counter(user_registers)
    dominant_register = reg_counts.most_common(1)[0][0] if reg_counts else "other"

    prompt_version = get_prompt_version(EXTRACTION_PROMPT)
    now = time.time()

    with transaction(store):
        # Store classifications.
        store.upsert_message_classifications(classifications)

        # Clear old provenance for this conversation.
        store.conn.execute(
            "DELETE FROM provenance_links WHERE conversation_id = ?",
            (conversation_id,),
        )

        counts: dict[str, int] = {}

        # Persist projects.
        for proj in result.projects:
            title_str = (proj.get("title") or "").strip()
            if not title_str:
                continue
            canon = title_str.lower()[:120]
            obj = DerivedObject(
                object_id=new_id("obj"),
                kind=DerivedKind.PROJECT.value,
                title=title_str[:80],
                description=proj.get("description", ""),
                canonical_key=canon,
                created_at=now,
                updated_at=now,
                source_register=dominant_register,
                source_reality_mode="literal",
                first_seen=conv.created_at,
                last_seen=conv.updated_at,
                status=proj.get("status", "active"),
            )
            actual_id = store.upsert_derived_object(obj)
            store.insert_provenance(ProvenanceLink(
                link_id=new_id("prov"),
                object_id=actual_id,
                conversation_id=conversation_id,
                chunk_id=None,
                excerpt=f"Project: {title_str}",
                created_at=now,
            ))
            counts["project"] = counts.get("project", 0) + 1

        # Persist decisions.
        for dec in result.decisions:
            verbatim = (dec.get("verbatim") or "").strip()
            if not verbatim:
                continue
            canon = verbatim.lower()[:120]
            obj = DerivedObject(
                object_id=new_id("obj"),
                kind=DerivedKind.DECISION.value,
                title=verbatim[:80],
                description=verbatim,
                canonical_key=canon,
                created_at=now,
                updated_at=now,
                source_register=dominant_register,
                source_reality_mode="literal",
                paraphrase=dec.get("paraphrase", ""),
            )
            actual_id = store.upsert_derived_object(obj)
            store.insert_provenance(ProvenanceLink(
                link_id=new_id("prov"),
                object_id=actual_id,
                conversation_id=conversation_id,
                chunk_id=None,
                excerpt=verbatim[:400],
                created_at=now,
            ))
            counts["decision"] = counts.get("decision", 0) + 1

        # Persist open loops.
        for loop in result.open_loops:
            verbatim = (loop.get("verbatim") or "").strip()
            if not verbatim:
                continue
            canon = verbatim.lower()[:120]
            obj = DerivedObject(
                object_id=new_id("obj"),
                kind=DerivedKind.OPEN_LOOP.value,
                title=verbatim[:80],
                description=verbatim,
                canonical_key=canon,
                created_at=now,
                updated_at=now,
                source_register=dominant_register,
                source_reality_mode="literal",
                paraphrase=loop.get("paraphrase", ""),
            )
            actual_id = store.upsert_derived_object(obj)
            store.insert_provenance(ProvenanceLink(
                link_id=new_id("prov"),
                object_id=actual_id,
                conversation_id=conversation_id,
                chunk_id=None,
                excerpt=verbatim[:400],
                created_at=now,
            ))
            counts["open_loop"] = counts.get("open_loop", 0) + 1

        # Persist entities.
        for ent in result.entities:
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            canon = name.lower()[:120]
            etype = ent.get("type", "other")
            gloss = ent.get("gloss", "")
            obj = DerivedObject(
                object_id=new_id("obj"),
                kind=DerivedKind.ENTITY.value,
                title=name[:80],
                description=f"{name} ({gloss})" if gloss else name,
                canonical_key=canon,
                created_at=now,
                updated_at=now,
                entity_type=etype,
                source_register=dominant_register,
            )
            actual_id = store.upsert_derived_object(obj)
            store.insert_provenance(ProvenanceLink(
                link_id=new_id("prov"),
                object_id=actual_id,
                conversation_id=conversation_id,
                chunk_id=None,
                excerpt=f"Entity: {name}" + (f" - {gloss}" if gloss else ""),
                created_at=now,
            ))
            counts["entity"] = counts.get("entity", 0) + 1

        # Clean up orphaned derived objects.
        store.delete_orphan_derived_objects()

        # Update conversation metadata.
        has_open_loops = counts.get("open_loop", 0) > 0
        importance = 0.0
        importance += min(len(messages) / 20.0, 2.0)
        importance += counts.get("decision", 0) * 0.5
        importance += counts.get("project", 0) * 0.7
        importance += counts.get("open_loop", 0) * 0.3
        importance = round(min(importance, 10.0), 3)

        store.update_conversation_meta(
            conversation_id,
            has_open_loops=has_open_loops,
            importance_score=importance,
            summary_short=result.summary if result.summary else conv.summary_short,
        )

        if result.summary:
            store.update_conversation_summary(
                conversation_id,
                summary_short=result.summary,
                summary_source="llm",
            )

        # Store LLM meta.
        store.upsert_conversation_llm_meta(ConversationLLMMeta(
            conversation_id=conversation_id,
            llm_summary=result.summary or None,
            dominant_register=dominant_register,
            content_hash=current_hash,
            extraction_prompt_version=prompt_version,
            extracted_at=now,
        ))

        store.reindex_conversation_fts(conversation_id)

    return {"skipped": False, "counts": counts, "dominant_register": dominant_register}


def run_pipeline_all(
    vault: Vault,
    store: Store,
    runner: LLMRunner,
    *,
    force: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """Run the pipeline on all eligible conversations."""
    eligible_states = ",".join(f"'{s}'" for s in EXTRACTABLE_STATES)
    rows = store.conn.execute(
        f"SELECT conversation_id FROM conversations WHERE state IN ({eligible_states}) ORDER BY imported_at"
    ).fetchall()
    ids = [r["conversation_id"] for r in rows]
    if limit is not None:
        ids = ids[:limit]
    return [run_pipeline(vault, store, runner, cid, force=force) for cid in ids]
