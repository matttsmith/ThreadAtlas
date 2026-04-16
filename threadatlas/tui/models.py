"""Pure data builders for TUI screens.

No curses here. Each builder returns a :class:`ScreenModel` with columns
and rows ready to be rendered. Tests target these builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..core.models import State
from ..core.vault import Vault
from ..store import Store


@dataclass
class ScreenModel:
    """Canonical data structure for any screen.

    Fields
    ------
    title : human-readable screen name
    columns : list of column headers
    rows : list of row dicts. Each row has:
        - ``cells``: list[str] matching ``columns``
        - ``id``: optional id used when drilling into detail
        - ``state``: optional state string (drives color)
    footer : short status line text
    """

    title: str
    columns: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    footer: str = ""


def _iso(ts) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


def _trunc(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def build_overview(vault: Vault, store: Store) -> ScreenModel:
    """State histogram + derived-object counts + source mix + recent imports."""
    state_rows = store.conn.execute(
        "SELECT state, COUNT(*) AS c FROM conversations GROUP BY state"
    ).fetchall()
    state_counts = {r["state"]: r["c"] for r in state_rows}
    total = sum(state_counts.values())

    source_rows = store.conn.execute(
        "SELECT source, COUNT(*) AS c FROM conversations GROUP BY source"
    ).fetchall()

    derived_rows = store.conn.execute(
        """
        SELECT kind, COUNT(DISTINCT o.object_id) AS c
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE c.state = 'indexed' AND o.state = 'active'
         GROUP BY kind
        """
    ).fetchall()
    derived = {r["kind"]: r["c"] for r in derived_rows}

    group_rows = store.conn.execute(
        "SELECT level, COUNT(*) AS c FROM conversation_groups GROUP BY level"
    ).fetchall()
    groups = {r["level"]: r["c"] for r in group_rows}

    rows: list[dict] = []
    rows.append({"cells": ["=== Corpus ==="            , ""]})
    rows.append({"cells": ["Total conversations"       , str(total)]})
    for s in ["indexed", "private", "pending_review", "quarantined"]:
        rows.append({"cells": [f"  {s}"                , str(state_counts.get(s, 0))], "state": s})

    rows.append({"cells": ["", ""]})
    rows.append({"cells": ["=== Source ==="            , ""]})
    for r in source_rows:
        rows.append({"cells": [f"  {r['source']}"      , str(r["c"])]})

    rows.append({"cells": ["", ""]})
    rows.append({"cells": ["=== Indexed derived ===" , ""]})
    for k in ("project", "decision", "open_loop", "entity", "preference", "artifact"):
        rows.append({"cells": [f"  {k}"               , str(derived.get(k, 0))]})

    rows.append({"cells": ["", ""]})
    rows.append({"cells": ["=== Groups ===", ""]})
    rows.append({"cells": ["  broad"                   , str(groups.get("broad", 0))]})
    rows.append({"cells": ["  fine"                    , str(groups.get("fine", 0))]})

    footer = f"vault ok \u00b7 {total} conversations \u00b7 {sum(derived.values())} derived objects"
    return ScreenModel(title="Overview", columns=["Metric", "Count"], rows=rows, footer=footer)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

_STATE_ORDER = (
    None,  # all
    State.PENDING_REVIEW.value,
    State.INDEXED.value,
    State.PRIVATE.value,
    State.QUARANTINED.value,
)


def build_conversations(
    store: Store,
    *,
    state_filter: str | None = None,
    query: str | None = None,
    limit: int = 500,
) -> ScreenModel:
    sql = """
        SELECT conversation_id, source, title, state, message_count,
               COALESCE(updated_at, created_at, imported_at) AS ts,
               summary_short, importance_score
          FROM conversations
         WHERE 1=1
    """
    params: list = []
    if state_filter is not None:
        sql += " AND state = ?"
        params.append(state_filter)
    if query:
        sql += " AND (title LIKE ? OR summary_short LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like])
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    rows = store.conn.execute(sql, params).fetchall()
    out_rows: list[dict] = []
    for r in rows:
        out_rows.append({
            "id": r["conversation_id"],
            "state": r["state"],
            "cells": [
                r["conversation_id"],
                r["source"] or "",
                r["state"] or "",
                str(r["message_count"] or 0),
                _iso(r["ts"]),
                _trunc(r["title"] or "", 50),
            ],
        })
    title = "Conversations"
    if state_filter:
        title += f" [{state_filter}]"
    if query:
        title += f" /{query}/"
    footer = f"{len(out_rows)} shown \u00b7 Enter: detail \u00b7 s: cycle state \u00b7 /: filter"
    return ScreenModel(
        title=title,
        columns=["id", "source", "state", "msgs", "updated", "title"],
        rows=out_rows,
        footer=footer,
    )


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

def build_groups(store: Store, *, level: str | None = None) -> ScreenModel:
    sql = "SELECT * FROM conversation_groups"
    params: list = []
    if level is not None:
        sql += " WHERE level = ?"
        params.append(level)
    sql += " ORDER BY level, member_count DESC"
    rows = store.conn.execute(sql, params).fetchall()
    out_rows: list[dict] = []
    for r in rows:
        out_rows.append({
            "id": r["group_id"],
            "cells": [
                r["group_id"],
                r["level"],
                str(r["member_count"]),
                _trunc(r["llm_label"] or "", 30),
                _trunc(r["keyword_label"] or "", 50),
            ],
        })
    footer = f"{len(out_rows)} groups \u00b7 Enter: members \u00b7 l: toggle level filter"
    if not out_rows:
        footer = "No groups yet. Run `threadatlas group <vault>`."
    return ScreenModel(
        title="Groups" + (f" [{level}]" if level else ""),
        columns=["id", "level", "members", "llm label", "keyword label"],
        rows=out_rows,
        footer=footer,
    )


def build_group_members(store: Store, group_id: str) -> ScreenModel:
    g = store.get_group(group_id)
    if g is None:
        return ScreenModel(title=f"Group {group_id}", columns=["info"], rows=[
            {"cells": ["(unknown group)"]},
        ])
    member_ids = store.list_group_members(group_id)
    if not member_ids:
        return ScreenModel(
            title=f"Group {group_id}",
            columns=["info"],
            rows=[{"cells": ["(no members)"]}],
            footer=f"level={g['level']} keyword_label={g['keyword_label']!r}",
        )
    placeholders = ",".join("?" for _ in member_ids)
    rows = store.conn.execute(
        f"""
        SELECT conversation_id, source, state, title
          FROM conversations
         WHERE conversation_id IN ({placeholders})
         ORDER BY updated_at DESC
        """,
        member_ids,
    ).fetchall()
    out_rows = [
        {
            "id": r["conversation_id"],
            "state": r["state"],
            "cells": [r["conversation_id"], r["source"], r["state"],
                      _trunc(r["title"] or "", 60)],
        }
        for r in rows
    ]
    return ScreenModel(
        title=f"Group {group_id} [{g['level']}] ({g['member_count']} members)",
        columns=["id", "source", "state", "title"],
        rows=out_rows,
        footer=f"keyword={g['keyword_label']!r} llm={g['llm_label']!r}",
    )


# ---------------------------------------------------------------------------
# Derived-object listings
# ---------------------------------------------------------------------------

def _build_derived_kind(store: Store, kind: str, *, title: str | None = None) -> ScreenModel:
    rows = store.conn.execute(
        """
        SELECT o.object_id, o.title, o.description,
               COUNT(DISTINCT p.conversation_id) AS convs
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
         WHERE o.kind = ? AND o.state = 'active'
         GROUP BY o.object_id
         ORDER BY convs DESC, o.title
         LIMIT 500
        """,
        (kind,),
    ).fetchall()
    out_rows = [
        {
            "id": r["object_id"],
            "cells": [r["object_id"], str(r["convs"]), _trunc(r["title"] or "", 70)],
        }
        for r in rows
    ]
    footer = f"{len(out_rows)} {kind}s \u00b7 Enter: provenance"
    return ScreenModel(
        title=title or kind.title() + "s",
        columns=["id", "convs", "title"],
        rows=out_rows,
        footer=footer,
    )


def build_projects(store: Store) -> ScreenModel:
    return _build_derived_kind(store, "project")


def build_open_loops(store: Store) -> ScreenModel:
    return _build_derived_kind(store, "open_loop", title="Open loops")


def build_decisions(store: Store) -> ScreenModel:
    return _build_derived_kind(store, "decision")


def build_entities(store: Store) -> ScreenModel:
    return _build_derived_kind(store, "entity")


# ---------------------------------------------------------------------------
# Detail screens
# ---------------------------------------------------------------------------

def build_conversation_detail(vault: Vault, store: Store, conversation_id: str) -> ScreenModel:
    c = store.get_conversation(conversation_id)
    if c is None:
        return ScreenModel(title="Conversation", columns=["info"],
                           rows=[{"cells": [f"unknown: {conversation_id}"]}])
    msgs = store.list_messages(conversation_id)
    chunks = store.list_chunks(conversation_id)
    prov = store.list_provenance_for_conversation(conversation_id)
    groups = store.list_group_memberships_for_conversation(conversation_id)

    rows: list[dict] = []
    rows.append({"cells": [f"id:        {c.conversation_id}"]})
    rows.append({"cells": [f"source:    {c.source}"]})
    rows.append({"cells": [f"title:     {c.title}"]})
    rows.append({"cells": [f"state:     {c.state}"], "state": c.state})
    rows.append({"cells": [f"imported:  {_iso(c.imported_at)}"]})
    rows.append({"cells": [f"created:   {_iso(c.created_at)}"]})
    rows.append({"cells": [f"msgs:      {len(msgs)}   chunks: {len(chunks)}   provenance: {len(prov)}"]})
    rows.append({"cells": [f"importance:{c.importance_score}"]})
    rows.append({"cells": [f"open_loops: {'yes' if c.has_open_loops else 'no'}"]})
    if c.manual_tags:
        rows.append({"cells": [f"tags (manual): {', '.join(c.manual_tags)}"]})
    if c.auto_tags:
        rows.append({"cells": [f"tags (auto):   {', '.join(c.auto_tags)}"]})
    rows.append({"cells": [""]})
    rows.append({"cells": ["--- summary ---"]})
    summary = c.summary_short or "(none)"
    for line in _wrap(summary, 100):
        rows.append({"cells": [line]})
    rows.append({"cells": [""]})
    if groups:
        rows.append({"cells": ["--- groups ---"]})
        for g in groups:
            label = g["llm_label"] or g["keyword_label"] or "(unlabeled)"
            rows.append({"cells": [f"  [{g['level']}] {label}"]})
        rows.append({"cells": [""]})

    # Full message thread — the main content.
    rows.append({"cells": [f"--- messages ({len(msgs)}) ---"]})
    rows.append({"cells": [""]})
    for m in msgs:
        role = (m.role or "?").upper()
        ts = _iso(m.timestamp) if m.timestamp else ""
        header = f"[{role}]  {ts}"
        rows.append({"cells": [header], "state": "state_indexed" if m.role == "user" else None})
        text = (m.content_text or "").strip()
        if text:
            for line in _wrap(text, 100):
                rows.append({"cells": [f"  {line}"]})
        else:
            rows.append({"cells": ["  (empty)"]})
        rows.append({"cells": [""]})

    return ScreenModel(
        title=f"{_trunc(c.title, 50)}",
        columns=[""],
        rows=rows,
        footer="Esc: back  |  j/k: scroll  |  PgUp/PgDn: page",
    )


def build_object_detail(store: Store, object_id: str) -> ScreenModel:
    obj = store.get_derived_object(object_id)
    if obj is None:
        return ScreenModel(title="Object", columns=["info"],
                           rows=[{"cells": [f"unknown: {object_id}"]}])
    prov = store.list_provenance_for_object(object_id)
    rows: list[dict] = []
    rows.append({"cells": [f"id:          {obj.object_id}"]})
    rows.append({"cells": [f"kind:        {obj.kind}"]})
    rows.append({"cells": [f"title:       {obj.title}"]})
    rows.append({"cells": [f"description: {_trunc(obj.description, 120)}"]})
    rows.append({"cells": [f"state:       {obj.state}"]})
    rows.append({"cells": [f"provenance:  {len(prov)} link(s)"]})
    rows.append({"cells": [""]})
    rows.append({"cells": ["--- provenance excerpts ---"]})
    for p in prov[:40]:
        rows.append({"cells": [f"  [{p.conversation_id[:16]}] {_trunc(p.excerpt, 120)}"]})
    return ScreenModel(
        title=f"{obj.kind} {object_id}",
        columns=["field"],
        rows=rows,
        footer="Esc / Backspace: back",
    )


def _wrap(text: str, width: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    out: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            out.append("")
            continue
        line = ""
        for word in paragraph.split():
            if line and len(line) + 1 + len(word) > width:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            out.append(line)
    return out
