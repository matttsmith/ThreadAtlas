"""Re-apply auto-rules to the existing corpus.

Rules can evolve (you add a new keyword after a while and want to make
sure existing threads matching it get re-classified). This module
applies the current ruleset to every conversation in the vault.

Safety invariants
-----------------
* Only DOWN-classifies. A conversation can be moved from
  ``pending_review`` or ``indexed`` to ``private`` or ``quarantined``,
  or from ``private`` to ``quarantined``. Nothing is ever up-classified
  by rescan.
* ``notes_local`` is updated to record the matching rule(s).
* FTS rows for the conversation are refreshed (stripped on
  quarantine, rebuilt otherwise) to keep the visibility invariant.
* Deleted conversations are of course not touched (they don't exist).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core.models import FTS_INDEXED_STATES, State
from .core.vault import Vault
from .core.workflow import transition_state
from .rules import RuleSet, evaluate, load_rules, summarize_matches
from .store import Store


_DOWN_ORDER = {
    State.PENDING_REVIEW.value: 0,
    State.INDEXED.value: 0,
    State.PRIVATE.value: 1,
    State.QUARANTINED.value: 2,
}


@dataclass
class RescanResult:
    scanned: int = 0
    down_classified: int = 0
    per_transition: dict[str, int] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)


def rescan(vault: Vault, store: Store) -> RescanResult:
    ruleset = load_rules(vault.root)
    result = RescanResult()
    if ruleset.empty:
        return result

    rows = store.conn.execute(
        "SELECT conversation_id, state FROM conversations"
    ).fetchall()
    for row in rows:
        cid = row["conversation_id"]
        current = row["state"]
        # Load the conversation's text for evaluation.
        conv = store.get_conversation(cid)
        if conv is None:
            continue
        msgs = [m.content_text for m in store.list_messages(cid)]
        target, matches = evaluate(
            ruleset,
            title=conv.title or "",
            summary=conv.summary_short or "",
            messages=msgs,
        )
        result.scanned += 1
        if target is None:
            continue
        if _DOWN_ORDER.get(target, 0) <= _DOWN_ORDER.get(current, 0):
            # Not more restrictive than current state; leave alone.
            continue
        # Perform the down-classification through the normal workflow so
        # chunks / FTS / provenance get the right treatment, and update
        # notes_local on the conversation.
        new_notes = summarize_matches(matches)
        if conv.notes_local and conv.notes_local not in new_notes:
            new_notes = (conv.notes_local + " | " + new_notes).strip(" |")
        transition_state(store, cid, target, vault=vault)
        store.update_conversation_meta(cid, notes_local=new_notes)
        store.conn.commit()
        label = f"{current}->{target}"
        result.per_transition[label] = result.per_transition.get(label, 0) + 1
        result.down_classified += 1
        if len(result.examples) < 20:
            result.examples.append({
                "conversation_id": cid,
                "from": current,
                "to": target,
                "notes": new_notes,
            })
    return result
