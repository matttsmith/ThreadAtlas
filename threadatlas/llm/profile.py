"""Profile generation for the indexed user.

Generates a narrative profile on demand, cached with a 7-day TTL.
Reads structured data from the database (projects, decisions, entities,
open loops, register distribution) and uses the LLM to synthesize a
coherent narrative.

If no LLM is configured, falls back to a structured data-only profile.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Iterable

from ..core.models import (
    DerivedKind,
    EXTRACTABLE_STATES,
    MCP_VISIBLE_STATES,
)
from ..core.vault import Vault
from ..store import Store

PROFILE_TTL_SECONDS = 7 * 86400  # 7 days


def generate_profile(
    vault: Vault,
    store: Store,
    *,
    focus: list[str] | None = None,
    visible_states: Iterable[str] = MCP_VISIBLE_STATES,
    force: bool = False,
) -> dict:
    """Generate or return cached profile.

    Returns a dict with ``profile`` (text), ``generated_at`` (timestamp),
    and ``focus`` (topics if specified).
    """
    vs = tuple(visible_states)

    # Check cache (any conversation's llm_meta can hold it; we use a special sentinel).
    if not force:
        cached = _get_cached_profile(store)
        if cached is not None:
            cached_data = json.loads(cached["profile_cache"])
            # If focus changed, regenerate.
            if cached_data.get("focus") == (focus or []):
                return cached_data

    # Gather structured data for the profile.
    projects = _list_kind_titles(store, DerivedKind.PROJECT.value, vs, 20)
    decisions = _list_kind_titles(store, DerivedKind.DECISION.value, vs, 20)
    open_loops = _list_kind_titles(store, DerivedKind.OPEN_LOOP.value, vs, 20)
    entities = _list_kind_with_type(store, DerivedKind.ENTITY.value, vs, 30)
    summaries = _list_conversation_summaries(store, vs, 30)
    register_dist = _register_distribution(store, vs)

    # Try LLM-based profile generation.
    profile_text = _generate_llm_profile(
        vault, projects, decisions, open_loops, entities, summaries,
        register_dist, focus,
    )

    if not profile_text:
        # Fallback: structured data profile.
        profile_text = _generate_structured_profile(
            projects, decisions, open_loops, entities, summaries, register_dist, focus,
        )

    now = time.time()
    result = {
        "profile": profile_text,
        "generated_at": now,
        "focus": focus or [],
        "data": {
            "project_count": len(projects),
            "decision_count": len(decisions),
            "open_loop_count": len(open_loops),
            "entity_count": len(entities),
            "conversation_count": len(summaries),
            "register_distribution": register_dist,
        },
    }

    # Cache the result.
    _cache_profile(store, result, now)

    return result


def _get_cached_profile(store: Store) -> dict | None:
    """Check for a cached profile that's still within TTL."""
    row = store.conn.execute(
        "SELECT profile_cache, profile_cached_at FROM conversation_llm_meta "
        "WHERE profile_cache IS NOT NULL ORDER BY profile_cached_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    cached_at = row["profile_cached_at"] or 0
    if time.time() - cached_at > PROFILE_TTL_SECONDS:
        return None
    return {"profile_cache": row["profile_cache"], "cached_at": cached_at}


def _cache_profile(store: Store, result: dict, now: float) -> None:
    """Store the profile cache in the first conversation's llm_meta row."""
    row = store.conn.execute(
        "SELECT conversation_id FROM conversation_llm_meta LIMIT 1"
    ).fetchone()
    if row:
        store.conn.execute(
            "UPDATE conversation_llm_meta SET profile_cache = ?, profile_cached_at = ? "
            "WHERE conversation_id = ?",
            (json.dumps(result, default=str), now, row["conversation_id"]),
        )
        store.conn.commit()


def _list_kind_titles(store: Store, kind: str, visible_states: tuple, limit: int) -> list[dict]:
    placeholders = ",".join("?" for _ in visible_states)
    rows = store.conn.execute(
        f"""
        SELECT DISTINCT o.title, o.description, o.status, o.paraphrase
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE o.kind = ? AND o.state = 'active' AND c.state IN ({placeholders})
         ORDER BY o.updated_at DESC
         LIMIT ?
        """,
        [kind, *visible_states, limit],
    ).fetchall()
    return [dict(r) for r in rows]


def _list_kind_with_type(store: Store, kind: str, visible_states: tuple, limit: int) -> list[dict]:
    placeholders = ",".join("?" for _ in visible_states)
    rows = store.conn.execute(
        f"""
        SELECT DISTINCT o.title, o.description, o.entity_type
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE o.kind = ? AND o.state = 'active' AND c.state IN ({placeholders})
         ORDER BY o.updated_at DESC
         LIMIT ?
        """,
        [kind, *visible_states, limit],
    ).fetchall()
    return [dict(r) for r in rows]


def _list_conversation_summaries(store: Store, visible_states: tuple, limit: int) -> list[dict]:
    placeholders = ",".join("?" for _ in visible_states)
    rows = store.conn.execute(
        f"""
        SELECT c.title, c.summary_short, c.source,
               COALESCE(c.updated_at, c.created_at) AS ts,
               m.dominant_register
          FROM conversations c
          LEFT JOIN conversation_llm_meta m ON m.conversation_id = c.conversation_id
         WHERE c.state IN ({placeholders})
         ORDER BY ts DESC
         LIMIT ?
        """,
        [*visible_states, limit],
    ).fetchall()
    return [dict(r) for r in rows]


def _register_distribution(store: Store, visible_states: tuple) -> dict[str, int]:
    placeholders = ",".join("?" for _ in visible_states)
    rows = store.conn.execute(
        f"""
        SELECT m.dominant_register, COUNT(*) AS cnt
          FROM conversation_llm_meta m
          JOIN conversations c ON c.conversation_id = m.conversation_id
         WHERE c.state IN ({placeholders}) AND m.dominant_register IS NOT NULL
         GROUP BY m.dominant_register
         ORDER BY cnt DESC
        """,
        list(visible_states),
    ).fetchall()
    return {r["dominant_register"]: r["cnt"] for r in rows}


def _generate_llm_profile(
    vault: Vault,
    projects: list[dict],
    decisions: list[dict],
    open_loops: list[dict],
    entities: list[dict],
    summaries: list[dict],
    register_dist: dict[str, int],
    focus: list[str] | None,
) -> str | None:
    """Try to generate profile via LLM. Returns None if LLM not available."""
    try:
        from .config import load_config
        from .runner import LLMRunner
        from .prompt_loader import render_prompt, PROFILE_PROMPT

        config = load_config(vault.root)
        if config is None or not config.is_enabled_for("profile"):
            return None

        runner = LLMRunner(vault, config)

        focus_instruction = ""
        if focus:
            focus_instruction = f"Focus the profile on these topics: {', '.join(focus)}. Be more detailed about these areas."

        def _fmt_list(items: list[dict], fields: list[str]) -> str:
            if not items:
                return "(none)"
            lines = []
            for item in items:
                parts = [str(item.get(f, "")) for f in fields if item.get(f)]
                lines.append("- " + " | ".join(parts))
            return "\n".join(lines)

        prompt = render_prompt(
            PROFILE_PROMPT,
            FOCUS_INSTRUCTION=focus_instruction,
            PROJECTS=_fmt_list(projects, ["title", "description", "status"]),
            DECISIONS=_fmt_list(decisions, ["title", "paraphrase"]),
            OPEN_LOOPS=_fmt_list(open_loops, ["title", "paraphrase"]),
            ENTITIES=_fmt_list(entities, ["title", "entity_type", "description"]),
            SUMMARIES=_fmt_list(summaries, ["title", "summary_short", "dominant_register"]),
            REGISTER_DIST=json.dumps(register_dist),
        )

        resp = runner.run("profile", prompt)
        if resp.success and resp.raw.strip():
            return resp.raw.strip()
    except Exception:
        pass
    return None


def _generate_structured_profile(
    projects: list[dict],
    decisions: list[dict],
    open_loops: list[dict],
    entities: list[dict],
    summaries: list[dict],
    register_dist: dict[str, int],
    focus: list[str] | None,
) -> str:
    """Generate a structured profile without LLM."""
    sections = []

    if projects:
        sections.append("Active Projects:")
        for p in projects[:10]:
            status = f" [{p.get('status', 'active')}]" if p.get("status") else ""
            sections.append(f"  - {p['title']}{status}: {p.get('description', '')}")

    if decisions:
        sections.append("\nRecent Decisions:")
        for d in decisions[:10]:
            sections.append(f"  - {d['title']}")

    if open_loops:
        sections.append("\nOpen Loops:")
        for ol in open_loops[:10]:
            sections.append(f"  - {ol['title']}")

    if entities:
        sections.append("\nKey Entities:")
        for e in entities[:15]:
            etype = f" ({e.get('entity_type', 'unknown')})" if e.get("entity_type") else ""
            sections.append(f"  - {e['title']}{etype}")

    if register_dist:
        sections.append("\nConversation Register Distribution:")
        total = sum(register_dist.values())
        for reg, count in sorted(register_dist.items(), key=lambda x: -x[1]):
            pct = round(100 * count / total, 1) if total else 0
            sections.append(f"  - {reg}: {count} ({pct}%)")

    if summaries:
        sections.append(f"\nRecent Conversations ({len(summaries)} shown):")
        for s in summaries[:10]:
            reg = f" [{s.get('dominant_register', '')}]" if s.get("dominant_register") else ""
            sections.append(f"  - {s.get('title', 'Untitled')}{reg}: {s.get('summary_short', '')[:100]}")

    return "\n".join(sections) if sections else "No indexed conversations found."
