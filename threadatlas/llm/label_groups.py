"""Optional LLM naming for clusters.

Only ever SETS ``llm_label`` on an existing group row; never creates new
groups. Groups with fewer than :data:`MIN_MEMBERS_FOR_LLM_NAMING` members
are skipped (keyword label is more useful for small clusters anyway).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster.groups import top_members_for_group
from ..core.vault import Vault
from ..store import Store
from .prompts import render_group_name_prompt
from .runner import LLMRunner, parse_json_response


MIN_MEMBERS_FOR_LLM_NAMING = 3
REPRESENTATIVE_MEMBERS = 15


@dataclass
class LabelOutcome:
    group_id: str
    success: bool
    label: str | None = None
    error: str | None = None


def _best_summary_for(store: Store, conversation_id: str) -> str:
    row = store.conn.execute(
        "SELECT title, summary_short FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    if row is None:
        return ""
    title = (row["title"] or "").strip()
    summary = (row["summary_short"] or "").strip()
    if title and summary:
        return f"{title}: {summary}"
    return summary or title


def label_group(
    vault: Vault, store: Store, runner: LLMRunner, group_id: str
) -> LabelOutcome:
    group = store.get_group(group_id)
    if group is None:
        return LabelOutcome(group_id, False, error="unknown_group")
    if group["member_count"] < MIN_MEMBERS_FOR_LLM_NAMING:
        return LabelOutcome(group_id, False, error="below_min_members")
    top_ids = top_members_for_group(store, group_id, top_n=REPRESENTATIVE_MEMBERS)
    summaries = [s for s in (_best_summary_for(store, cid) for cid in top_ids) if s]
    if len(summaries) < MIN_MEMBERS_FOR_LLM_NAMING:
        return LabelOutcome(group_id, False, error="no_usable_summaries")
    prompt = render_group_name_prompt(summaries)
    resp = runner.run("group_naming", prompt, conversation_ids=top_ids)
    if not resp.success:
        return LabelOutcome(group_id, False, error=resp.error)
    parsed = parse_json_response(resp)
    if not parsed or not isinstance(parsed.get("name"), str):
        return LabelOutcome(group_id, False, error="malformed_response")
    name = parsed["name"].strip().rstrip(".")
    # Boring guard rails: reject empty / generic outputs.
    generic = {"various topics", "general discussion", "miscellaneous",
               "assorted threads", "mixed topics", "miscellaneous topics"}
    if not name or name.lower() in generic or len(name) > 120:
        return LabelOutcome(group_id, False, error="generic_or_invalid")
    store.set_group_llm_label(group_id, name)
    store.conn.commit()
    return LabelOutcome(group_id, True, label=name)


def label_all_groups(
    vault: Vault, store: Store, runner: LLMRunner, *, level: str | None = None
) -> list[LabelOutcome]:
    rows = store.list_groups(level)
    return [label_group(vault, store, runner, g["group_id"]) for g in rows]
