"""LLM-based conversation summarization.

* Only runs for conversations in ``EXTRACTABLE_STATES`` (indexed + private).
  Pending_review and quarantined content never reaches the LLM.
* Idempotent: re-running replaces the previous summary.
* Updates ``summary_short`` and ``summary_source='llm'`` on success.
* On LLM failure (timeout, bad JSON) the function leaves the existing
  summary untouched and returns ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.models import EXTRACTABLE_STATES
from ..core.vault import Vault
from ..store import Store
from .prompts import render_messages, render_summarize_prompt
from .runner import LLMRunner, parse_json_response


@dataclass
class SummarizeOutcome:
    conversation_id: str
    success: bool
    prompt_chars: int = 0
    response_chars: int = 0
    error: str | None = None
    new_summary: str | None = None


def summarize_conversation(
    vault: Vault,
    store: Store,
    runner: LLMRunner,
    conversation_id: str,
    *,
    max_messages: int = 40,
) -> SummarizeOutcome:
    conv = store.get_conversation(conversation_id)
    if conv is None:
        return SummarizeOutcome(conversation_id, False, error="unknown_conversation")
    if conv.state not in EXTRACTABLE_STATES:
        return SummarizeOutcome(conversation_id, False, error=f"ineligible_state:{conv.state}")

    msgs = store.list_messages(conversation_id)
    # Simple truncation strategy: take the first N messages so we always
    # include the opening. If the thread is longer, we tail-truncate since
    # the opening usually establishes the topic.
    msgs = msgs[:max_messages]
    rendered = render_messages(msgs)
    prompt = render_summarize_prompt(conv.title, rendered)

    resp = runner.run("summaries", prompt, conversation_ids=[conversation_id])
    if not resp.success:
        return SummarizeOutcome(
            conversation_id, False,
            prompt_chars=resp.prompt_chars, response_chars=resp.response_chars,
            error=resp.error,
        )
    parsed = parse_json_response(resp)
    if not parsed or not isinstance(parsed.get("summary"), str):
        return SummarizeOutcome(
            conversation_id, False,
            prompt_chars=resp.prompt_chars, response_chars=resp.response_chars,
            error="malformed_response",
        )
    summary = parsed["summary"].strip()
    if not summary:
        return SummarizeOutcome(
            conversation_id, False,
            prompt_chars=resp.prompt_chars, response_chars=resp.response_chars,
            error="empty_summary",
        )
    # Keep it short; trust-but-verify bound against runaway output.
    if len(summary) > 1200:
        summary = summary[:1200].rstrip() + "..."

    store.update_conversation_summary(
        conversation_id, summary_short=summary, summary_source="llm",
    )
    # Reindex so the new summary hits FTS.
    store.reindex_conversation_fts(conversation_id)
    store.conn.commit()
    return SummarizeOutcome(
        conversation_id, True,
        prompt_chars=resp.prompt_chars, response_chars=resp.response_chars,
        new_summary=summary,
    )


def summarize_all_eligible(
    vault: Vault, store: Store, runner: LLMRunner, *, limit: int | None = None
) -> list[SummarizeOutcome]:
    state_placeholders = ",".join(f"'{s}'" for s in EXTRACTABLE_STATES)
    rows = store.conn.execute(
        f"SELECT conversation_id FROM conversations WHERE state IN ({state_placeholders}) ORDER BY imported_at"
    ).fetchall()
    ids = [r["conversation_id"] for r in rows]
    if limit is not None:
        ids = ids[:limit]
    return [summarize_conversation(vault, store, runner, cid) for cid in ids]
