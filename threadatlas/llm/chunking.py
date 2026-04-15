"""LLM-gated chunking.

Precision-first: the deterministic chunker (see
``threadatlas.extract.chunking``) proposes candidate boundaries. For each
adjacent pair of proposed chunks, the LLM is asked whether this is a clear
topic shift. If the LLM says "no" (the default bias), we merge the two
chunks. The LLM can therefore only ever REMOVE boundaries, never add them.

This means LLM-assisted chunking is strictly more conservative than the
deterministic chunker. If you want aggressive splitting, use the
deterministic chunker alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.models import EXTRACTABLE_STATES, Chunk, new_id
from ..core.vault import Vault
from ..extract.chunking import chunk_conversation as _deterministic_chunk
from ..store import Store
from .prompts import render_chunk_gate_prompt, render_messages
from .runner import LLMRunner, parse_json_response


# How many messages of context to include on each side of a boundary when
# asking the LLM to judge topic shift. Enough to give context, not enough
# to drown the model.
CONTEXT_MESSAGES_PER_SIDE = 3

# Safety bound: no more than this many adjacent-boundary evaluations per
# conversation. Prevents pathological loops if the LLM repeatedly flips.
MAX_GATE_ITERATIONS = 10


@dataclass
class GateOutcome:
    conversation_id: str
    before_chunks: int = 0
    after_chunks: int = 0
    merges: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    # Per-boundary log entries: (chunk_a_idx, chunk_b_idx, split:bool, reason, ok:bool)
    decisions: list[dict] = field(default_factory=list)


def _messages_in_range(
    all_messages: list, start_ordinal: int, end_ordinal: int
) -> list:
    return [m for m in all_messages if start_ordinal <= m.ordinal <= end_ordinal]


def _tail_head_text(
    messages: list, chunk_a: Chunk, chunk_b: Chunk, per_side: int
) -> tuple[str, str]:
    tail_msgs = _messages_in_range(
        messages, chunk_a.start_message_ordinal, chunk_a.end_message_ordinal
    )[-per_side:]
    head_msgs = _messages_in_range(
        messages, chunk_b.start_message_ordinal, chunk_b.end_message_ordinal
    )[:per_side]
    return render_messages(tail_msgs), render_messages(head_msgs)


def llm_chunk_conversation(
    vault: Vault,
    store: Store,
    runner: LLMRunner,
    conversation_id: str,
) -> GateOutcome:
    """Run deterministic chunking then apply the LLM gate to merge marginal boundaries."""
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise KeyError(conversation_id)
    outcome = GateOutcome(conversation_id=conversation_id)
    if conv.state not in EXTRACTABLE_STATES:
        outcome.before_chunks = 0
        outcome.after_chunks = 0
        return outcome

    # Step 1: deterministic chunking (already reindexes FTS / commits).
    det_chunks = _deterministic_chunk(store, conversation_id)
    outcome.before_chunks = len(det_chunks)
    if len(det_chunks) <= 1:
        outcome.after_chunks = outcome.before_chunks
        return outcome

    messages = store.list_messages(conversation_id)
    chunks: list[Chunk] = list(det_chunks)

    # Step 2: iterate adjacent boundaries, merging when LLM says "not a topic shift".
    iter_count = 0
    while iter_count < MAX_GATE_ITERATIONS:
        iter_count += 1
        merged_this_pass = False
        i = 0
        while i < len(chunks) - 1:
            a, b = chunks[i], chunks[i + 1]
            tail, head = _tail_head_text(messages, a, b, CONTEXT_MESSAGES_PER_SIDE)
            if not tail or not head:
                i += 1
                continue
            prompt = render_chunk_gate_prompt(tail, head)
            resp = runner.run("chunk_gating", prompt, conversation_ids=[conversation_id])
            outcome.llm_calls += 1
            parsed = parse_json_response(resp) if resp.success else None
            if parsed is None or not isinstance(parsed.get("split"), bool):
                outcome.llm_failures += 1
                outcome.decisions.append({
                    "chunk_a_idx": a.chunk_index, "chunk_b_idx": b.chunk_index,
                    "split": None, "reason": None, "ok": False,
                })
                # On LLM failure, keep the deterministic boundary (bias toward
                # preserving what the deterministic chunker chose when we
                # cannot get a clear answer).
                i += 1
                continue
            split = bool(parsed["split"])
            reason = parsed.get("reason") or ""
            outcome.decisions.append({
                "chunk_a_idx": a.chunk_index, "chunk_b_idx": b.chunk_index,
                "split": split, "reason": reason[:200], "ok": True,
            })
            if split:
                i += 1
                continue
            # Merge a and b into a single chunk.
            merged = Chunk(
                chunk_id=a.chunk_id,  # keep A's id; we'll assign new indexes at the end
                conversation_id=conversation_id,
                chunk_index=a.chunk_index,
                start_message_ordinal=a.start_message_ordinal,
                end_message_ordinal=b.end_message_ordinal,
                chunk_title=a.chunk_title,
                summary_short=a.summary_short or b.summary_short,
                project_id=a.project_id or b.project_id,
                importance_score=max(a.importance_score, b.importance_score),
                has_open_loops=a.has_open_loops or b.has_open_loops,
            )
            chunks[i] = merged
            chunks.pop(i + 1)
            outcome.merges += 1
            merged_this_pass = True
            # Do NOT advance; reconsider the new neighbor pair.
        if not merged_this_pass:
            break

    # Renumber chunk_index and assign fresh chunk_ids so downstream provenance
    # behaves deterministically.
    final_chunks: list[Chunk] = []
    for idx, c in enumerate(chunks):
        final_chunks.append(Chunk(
            chunk_id=new_id("chk"),
            conversation_id=conversation_id,
            chunk_index=idx,
            start_message_ordinal=c.start_message_ordinal,
            end_message_ordinal=c.end_message_ordinal,
            chunk_title=c.chunk_title,
            summary_short=c.summary_short,
            project_id=c.project_id,
            importance_score=c.importance_score,
            has_open_loops=c.has_open_loops,
        ))
    store.replace_chunks(conversation_id, final_chunks)
    store.reindex_conversation_fts(conversation_id)
    store.conn.commit()
    outcome.after_chunks = len(final_chunks)
    return outcome


def llm_chunk_all_eligible(
    vault: Vault, store: Store, runner: LLMRunner
) -> list[GateOutcome]:
    state_placeholders = ",".join(f"'{s}'" for s in EXTRACTABLE_STATES)
    rows = store.conn.execute(
        f"SELECT conversation_id FROM conversations WHERE state IN ({state_placeholders})"
    ).fetchall()
    return [llm_chunk_conversation(vault, store, runner, r["conversation_id"]) for r in rows]
