"""Opt-in MCP write tools.

By default MCP is read-only. A vault can enable a **narrow** set of
write tools by dropping ``<vault>/mcp_config.json``:

.. code-block:: json

    {"allow_writes": true}

When enabled, the following tools become callable over MCP:

* ``set_group_label`` - correct a cluster's ``llm_label``.
* ``add_tag`` / ``remove_tag`` - manual tags on ``indexed`` conversations.
* ``rename_derived_object`` - correct a wrong project/entity title.

Explicitly NOT exposed, even with ``allow_writes: true``:

* State changes (approve / private / quarantine) - CLI + ``--yes`` only.
* Hard delete - CLI + ``--yes`` only.
* Merging / suppressing / deleting derived objects.
* Summaries, message content, or any raw content.
* Modifying the vault config itself.

Every successful write is appended to
``<vault>/logs/mcp_mutations.jsonl`` with timestamp, tool, args, and a
brief outcome. Operators can audit what Claude did.

Input validation caps every string argument at a conservative length.
Prompt injection that tries to stuff huge payloads into a label is
truncated, not executed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..core.models import MCP_VISIBLE_STATES
from ..core.vault import Vault
from ..store import Store


MCP_CONFIG_BASENAME = "mcp_config.json"
MUTATION_LOG_BASENAME = "mcp_mutations.jsonl"

MAX_LABEL_LEN = 120
MAX_TAG_LEN = 60
MAX_TITLE_LEN = 120
MAX_TAGS_PER_CALL = 10


def mcp_config_path(vault: Vault) -> Path:
    return vault.root / MCP_CONFIG_BASENAME


def writes_enabled(vault: Vault) -> bool:
    p = mcp_config_path(vault)
    if not p.exists():
        return False
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(cfg.get("allow_writes"))


def _log_mutation(vault: Vault, entry: dict) -> None:
    try:
        vault.logs.mkdir(parents=True, exist_ok=True)
        with (vault.logs / MUTATION_LOG_BASENAME).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # never let logging break the caller


def _truncate(s, n: int) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "\u2026"


# ---------------------------------------------------------------------------
# Handlers - each returns (ok: bool, payload: dict)
# ---------------------------------------------------------------------------

def set_group_label(vault: Vault, store: Store, args: dict) -> tuple[bool, dict]:
    gid = str(args.get("group_id") or "")
    label = _truncate(args.get("label") or "", MAX_LABEL_LEN)
    if not gid or not label:
        return False, {"error": "group_id and label are required"}
    g = store.get_group(gid)
    if g is None:
        return False, {"error": f"unknown group: {gid}"}
    store.set_group_llm_label(gid, label)
    store.conn.commit()
    _log_mutation(vault, {
        "ts": time.time(), "tool": "set_group_label",
        "group_id": gid, "new_label": label,
        "previous_label": g.get("llm_label"),
    })
    return True, {"group_id": gid, "label": label}


def _require_indexed_conversation(store: Store, cid: str) -> tuple[bool, dict | None]:
    c = store.get_conversation(cid)
    if c is None:
        return False, {"error": f"unknown conversation: {cid}"}
    if c.state not in MCP_VISIBLE_STATES:
        return False, {"error": f"conversation is not indexed: {cid}"}
    return True, None


def add_tag(vault: Vault, store: Store, args: dict) -> tuple[bool, dict]:
    cid = str(args.get("conversation_id") or "")
    tags_raw = args.get("tags") or []
    if not isinstance(tags_raw, list):
        return False, {"error": "tags must be a list of strings"}
    tags = [
        _truncate(t, MAX_TAG_LEN)
        for t in tags_raw
        if isinstance(t, str) and t.strip()
    ][:MAX_TAGS_PER_CALL]
    if not tags:
        return False, {"error": "at least one non-empty tag required"}
    ok, err = _require_indexed_conversation(store, cid)
    if not ok:
        return False, err
    updated = store.add_manual_tags(cid, tags)
    store.conn.commit()
    _log_mutation(vault, {
        "ts": time.time(), "tool": "add_tag",
        "conversation_id": cid, "tags_added": tags,
        "manual_tags_after": updated,
    })
    return True, {"conversation_id": cid, "manual_tags": updated}


def remove_tag(vault: Vault, store: Store, args: dict) -> tuple[bool, dict]:
    cid = str(args.get("conversation_id") or "")
    tags_raw = args.get("tags") or []
    if not isinstance(tags_raw, list):
        return False, {"error": "tags must be a list of strings"}
    tags = [
        _truncate(t, MAX_TAG_LEN)
        for t in tags_raw
        if isinstance(t, str) and t.strip()
    ][:MAX_TAGS_PER_CALL]
    if not tags:
        return False, {"error": "at least one non-empty tag required"}
    ok, err = _require_indexed_conversation(store, cid)
    if not ok:
        return False, err
    remaining = store.remove_manual_tags(cid, tags)
    store.conn.commit()
    _log_mutation(vault, {
        "ts": time.time(), "tool": "remove_tag",
        "conversation_id": cid, "tags_removed": tags,
        "manual_tags_after": remaining,
    })
    return True, {"conversation_id": cid, "manual_tags": remaining}


def rename_derived_object(vault: Vault, store: Store, args: dict) -> tuple[bool, dict]:
    oid = str(args.get("object_id") or "")
    new_title = _truncate(args.get("title") or "", MAX_TITLE_LEN)
    if not oid or not new_title:
        return False, {"error": "object_id and title are required"}
    obj = store.get_derived_object(oid)
    if obj is None:
        return False, {"error": f"unknown object: {oid}"}
    if obj.state != "active":
        return False, {"error": f"object {oid} is not active"}
    # Confirm the object is referenced from at least one indexed
    # conversation - a rename initiated by Claude should only apply to
    # objects Claude can legitimately see.
    visible = store.conn.execute(
        """
        SELECT 1 FROM provenance_links p
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE p.object_id = ? AND c.state = 'indexed' LIMIT 1
        """,
        (oid,),
    ).fetchone()
    if visible is None:
        return False, {"error": f"object {oid} has no indexed provenance; refusing"}
    old_title = obj.title
    store.rename_derived_object(oid, new_title)
    store.conn.commit()
    _log_mutation(vault, {
        "ts": time.time(), "tool": "rename_derived_object",
        "object_id": oid, "kind": obj.kind,
        "old_title": old_title, "new_title": new_title,
    })
    return True, {"object_id": oid, "kind": obj.kind, "title": new_title}
