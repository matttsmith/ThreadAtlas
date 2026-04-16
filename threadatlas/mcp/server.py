"""Minimal stdio JSON-RPC server speaking the Model Context Protocol surface.

Why not use the official MCP Python SDK? The spec forbids runtime
dependencies that silently fetch resources, and asks us to keep the MCP
attack surface tiny. The MCP wire format that Claude Desktop / Claude
Developer Tools use is JSON-RPC 2.0 over stdio with newline-framed
messages; we implement only the subset we need:

  * ``initialize`` -> server info
  * ``tools/list`` -> describe available tools
  * ``tools/call`` -> invoke a tool

Every tool below filters to ``MCP_VISIBLE_STATES`` (indexed only). There are
NO mutating tools in v1.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, IO

from .. import __version__ as TA_VERSION
from ..core.models import ALL_REGISTERS, DEFAULT_REGISTER_EXCLUDES, MCP_VISIBLE_STATES
from ..core.vault import Vault, open_vault
from ..search import (
    list_decisions,
    list_entities,
    list_open_loops,
    project_timeline,
    project_view,
    query as query_engine,
    search_chunks,
    search_conversations,
)
from ..search.search import list_projects
from ..store import open_store, Store
from . import writes as writes_mod


PROTOCOL_VERSION = "2024-11-05"


def _parse_date_param(val: str | None) -> float | None:
    """Parse ISO date string to POSIX timestamp."""
    if not val:
        return None
    from datetime import datetime, timezone
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _parse_register_param(val) -> list[str] | None:
    """Parse register filter: list of strings or None."""
    if val is None:
        return None
    if isinstance(val, list):
        return [r for r in val if r in ALL_REGISTERS]
    if isinstance(val, str):
        return [val] if val in ALL_REGISTERS else None
    return None


# Common filter schema properties for tool definitions.
_FILTER_PROPERTIES = {
    "after": {"type": "string", "description": "ISO date (YYYY-MM-DD). Only include items after this date."},
    "before": {"type": "string", "description": "ISO date (YYYY-MM-DD). Only include items before this date."},
    "register": {
        "type": "array",
        "items": {"type": "string", "enum": list(ALL_REGISTERS)},
        "description": "Filter by conversation register. Default excludes roleplay and jailbreak_experiment.",
    },
    "source": {
        "type": "string",
        "enum": ["chatgpt", "claude", "all"],
        "description": "Filter by conversation source. Default: all.",
    },
}


@dataclass
class _Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[[dict], Any]


def _ok(content: Any) -> dict:
    return {
        "content": [
            {"type": "text", "text": json.dumps(content, indent=2, default=str)}
        ]
    }


def _err(message: str) -> dict:
    return {
        "isError": True,
        "content": [{"type": "text", "text": message}],
    }


def build_tools(vault: Vault, store: Store) -> dict[str, _Tool]:
    """Construct the (read-mostly) tool registry."""
    visible = tuple(MCP_VISIBLE_STATES)

    def _common_filters(args: dict) -> dict:
        """Extract common filter parameters."""
        return {
            "after": _parse_date_param(args.get("after")),
            "before": _parse_date_param(args.get("before")),
            "register": _parse_register_param(args.get("register")),
            "source_filter": args.get("source") if args.get("source") != "all" else None,
        }

    def t_search_conversations(args: dict) -> dict:
        q = str(args.get("query") or "")
        limit = int(args.get("limit") or 25)
        filters = _common_filters(args)
        hits = search_conversations(store, q, visible_states=visible, limit=limit, **filters)
        return _ok([h.__dict__ for h in hits])

    def t_search_chunks(args: dict) -> dict:
        q = str(args.get("query") or "")
        limit = int(args.get("limit") or 25)
        filters = _common_filters(args)
        hits = search_chunks(store, q, visible_states=visible, limit=limit, **filters)
        return _ok([h.__dict__ for h in hits])

    def t_get_conversation_summary(args: dict) -> dict:
        cid = str(args.get("conversation_id") or "")
        c = store.get_conversation(cid)
        if c is None or c.state not in MCP_VISIBLE_STATES:
            return _err(f"Conversation not visible: {cid}")
        # Include LLM summary if available.
        llm_meta = store.get_conversation_llm_meta(cid)
        llm_summary = llm_meta.llm_summary if llm_meta else None
        dominant_register = llm_meta.dominant_register if llm_meta else None
        return _ok({
            "conversation_id": c.conversation_id,
            "source": c.source,
            "title": c.title,
            "summary_short": c.summary_short,
            "summary_long": c.summary_long,
            "llm_summary": llm_summary,
            "dominant_register": dominant_register,
            "message_count": c.message_count,
            "state": c.state,
            "manual_tags": c.manual_tags,
            "auto_tags": c.auto_tags,
            "importance_score": c.importance_score,
            "has_open_loops": c.has_open_loops,
        })

    def t_get_conversation_messages(args: dict) -> dict:
        cid = str(args.get("conversation_id") or "")
        limit = int(args.get("limit") or 200)
        c = store.get_conversation(cid)
        if c is None or c.state not in MCP_VISIBLE_STATES:
            return _err(f"Conversation not visible: {cid}")
        msgs = store.list_messages(cid)[:limit]
        return _ok([
            {
                "ordinal": m.ordinal, "role": m.role,
                "timestamp": m.timestamp, "content_text": m.content_text,
            }
            for m in msgs
        ])

    def t_get_conversation_chunks(args: dict) -> dict:
        cid = str(args.get("conversation_id") or "")
        c = store.get_conversation(cid)
        if c is None or c.state not in MCP_VISIBLE_STATES:
            return _err(f"Conversation not visible: {cid}")
        chunks = store.list_chunks(cid)
        return _ok([ch.__dict__ for ch in chunks])

    def t_list_projects(args: dict) -> dict:
        filters = _common_filters(args)
        return _ok(list_projects(store, visible_states=visible,
                                 limit=int(args.get("limit") or 200), **filters))

    def t_get_project(args: dict) -> dict:
        pid = str(args.get("project_id") or "")
        view = project_view(store, pid, visible_states=visible)
        if view is None:
            return _err(f"No project visible with id {pid}")
        return _ok(view)

    def t_get_project_timeline(args: dict) -> dict:
        pid = str(args.get("project_id") or "")
        return _ok(project_timeline(store, pid, visible_states=visible))

    def t_list_open_loops(args: dict) -> dict:
        filters = _common_filters(args)
        return _ok(list_open_loops(store, visible_states=visible,
                                   limit=int(args.get("limit") or 200), **filters))

    def t_list_decisions(args: dict) -> dict:
        filters = _common_filters(args)
        return _ok(list_decisions(store, visible_states=visible,
                                  limit=int(args.get("limit") or 200), **filters))

    def t_list_entities(args: dict) -> dict:
        filters = _common_filters(args)
        return _ok(list_entities(store, visible_states=visible,
                                 limit=int(args.get("limit") or 200), **filters))

    def t_generate_profile(args: dict) -> dict:
        """Generate a narrative profile of the indexed user."""
        focus = args.get("focus")
        from ..llm.profile import generate_profile
        result = generate_profile(vault, store, focus=focus, visible_states=visible)
        return _ok(result)

    def t_find_related(args: dict) -> dict:
        """Find related conversations by semantic similarity to free-text context."""
        context = str(args.get("context") or "")
        limit = int(args.get("limit") or 10)
        if not context.strip():
            return _err("context parameter is required")
        from ..search.embeddings import (
            bytes_to_embedding,
            cosine_similarity,
            fit_embedder_from_corpus,
        )
        try:
            embedder = fit_embedder_from_corpus(store)
            query_vec = embedder.embed(context)
            all_embeddings = store.get_all_chunk_embeddings(visible_states=visible)
            if not all_embeddings:
                return _ok([])

            # Score each conversation by max chunk similarity.
            conv_scores: dict[str, float] = {}
            for chunk_id, conv_id, emb_bytes in all_embeddings:
                emb_vec = bytes_to_embedding(emb_bytes)
                sim = cosine_similarity(query_vec, emb_vec)
                if conv_id not in conv_scores or sim > conv_scores[conv_id]:
                    conv_scores[conv_id] = sim

            ranked = sorted(conv_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            results = []
            for conv_id, score in ranked:
                if score < 0.01:
                    continue
                conv = store.get_conversation(conv_id)
                if conv is None:
                    continue
                results.append({
                    "conversation_id": conv_id,
                    "title": conv.title,
                    "source": conv.source,
                    "score": round(score, 4),
                    "summary_short": conv.summary_short[:240],
                })
            return _ok(results)
        except Exception as e:
            return _err(f"Semantic search not available: {e}")

    def t_list_groups(args: dict) -> dict:
        """MCP-safe group list.

        Groups are clustered over ``indexed + private`` but labels exposed
        via MCP are recomputed from ONLY the indexed members to avoid
        leaking private content into label text. Groups with fewer than
        ``MIN_INDEXED_FOR_SAFE_LABEL`` indexed members get no label (but
        are still listed with a count, so Claude knows they exist).
        """
        from ..cluster.safe_labels import compute_safe_keyword_label

        level = args.get("level")
        if level not in (None, "broad", "fine"):
            return _err("level must be 'broad' or 'fine'")
        groups = store.list_groups(level=level)
        out: list[dict] = []
        for g in groups:
            members = store.list_group_members(g["group_id"])
            if not members:
                continue
            placeholders = ",".join("?" for _ in members)
            visible = store.conn.execute(
                f"SELECT COUNT(*) AS c FROM conversations WHERE conversation_id IN ({placeholders}) AND state = 'indexed'",
                members,
            ).fetchone()["c"]
            if visible == 0:
                continue
            safe_label = compute_safe_keyword_label(store, g["group_id"])
            out.append({
                "group_id": g["group_id"],
                "level": g["level"],
                "member_count": g["member_count"],
                "visible_member_count": visible,
                # MCP NEVER exposes the mixed-pool keyword_label or llm_label
                # directly; only the safe label derived from indexed members.
                "label": safe_label,
            })
        return _ok(out)

    def t_get_group(args: dict) -> dict:
        from ..cluster.safe_labels import compute_safe_keyword_label

        gid = str(args.get("group_id") or "")
        g = store.get_group(gid)
        if g is None:
            return _err(f"Unknown group: {gid}")
        member_ids = store.list_group_members(gid)
        if member_ids:
            placeholders = ",".join("?" for _ in member_ids)
            rows = store.conn.execute(
                f"SELECT conversation_id, title FROM conversations "
                f"WHERE conversation_id IN ({placeholders}) AND state = 'indexed'",
                member_ids,
            ).fetchall()
            visible = [{"conversation_id": r["conversation_id"], "title": r["title"]} for r in rows]
        else:
            visible = []
        safe_label = compute_safe_keyword_label(store, gid)
        return _ok({
            "group": {
                "group_id": g["group_id"],
                "level": g["level"],
                "member_count": g["member_count"],
                "label": safe_label,
            },
            "indexed_members": visible,
        })

    def t_inspect_conversation_storage(args: dict) -> dict:
        """Audit hook: what does the system store about this conversation?

        Returns metadata only; refuses to expose content unless visible.
        """
        cid = str(args.get("conversation_id") or "")
        c = store.get_conversation(cid)
        if c is None:
            return _err(f"Unknown conversation: {cid}")
        # Always allow metadata-only inspection; never reveal text for non-visible.
        msg_count = store.conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE conversation_id = ?",
            (cid,),
        ).fetchone()["c"]
        chunk_count = store.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE conversation_id = ?",
            (cid,),
        ).fetchone()["c"]
        prov_count = store.conn.execute(
            "SELECT COUNT(*) AS c FROM provenance_links WHERE conversation_id = ?",
            (cid,),
        ).fetchone()["c"]
        return _ok({
            "conversation_id": cid,
            "title": c.title if c.state in MCP_VISIBLE_STATES else "[redacted]",
            "state": c.state,
            "message_count": msg_count,
            "chunk_count": chunk_count,
            "provenance_link_count": prov_count,
            "mcp_visible": c.state in MCP_VISIBLE_STATES,
        })

    def t_query(args: dict) -> dict:
        """Structured query across all indexed material.

        Accepts a single query string that may contain filter prefixes
        like ``source:chatgpt``, ``tag:architecture``, ``kind:decision``,
        ``project:<id>``, ``after:2024-01-01``, ``before:2024-12-31``,
        ``has:open_loops``, ``has:chunks``. Remaining text is used for
        keyword search. Returns a unified ranked result set spanning
        conversations, chunks, and derived objects.
        """
        q = str(args.get("query") or "")
        limit = int(args.get("limit") or 25)
        result = query_engine(store, q, visible_states=visible, limit=limit)
        return _ok({
            "raw_query": result.raw_query,
            "filters": result.filters,
            "hits": [
                {
                    "hit_type": h.hit_type,
                    "id": h.id,
                    "title": h.title,
                    "snippet": h.snippet,
                    "score": h.score,
                    "metadata": h.metadata,
                }
                for h in result.hits
            ],
            "total_by_type": result.total_by_type,
            "elapsed_ms": result.elapsed_ms,
        })

    # Build filter properties for reuse across tool schemas.
    fp = dict(_FILTER_PROPERTIES)

    tools = [
        _Tool("query", "Structured query across all indexed material. "
              "Supports filter prefixes: source:, tag:, kind:, project:, after:, before:, has:, register:. "
              "Remaining text is used for hybrid keyword + semantic search. Returns conversations, chunks, and derived objects.",
              {"type": "object",
               "properties": {
                   "query": {"type": "string",
                             "description": "Query string, optionally with filter prefixes "
                                            "(e.g. 'migration plan source:chatgpt', 'kind:decision after:2024-06-01', 'register:work')"},
                   "limit": {"type": "integer", "description": "Max results (default 25)"}},
               "required": ["query"]},
              t_query),
        _Tool("search_conversations", "Hybrid keyword + semantic search over indexed conversations. "
              "Finds both exact keyword matches and semantically similar content.",
              {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, **fp}, "required": ["query"]},
              t_search_conversations),
        _Tool("search_chunks", "Keyword search over chunks of indexed conversations.",
              {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, **fp}, "required": ["query"]},
              t_search_chunks),
        _Tool("get_conversation_summary", "Get LLM-generated summary and metadata for one indexed conversation. "
              "Includes register classification, importance score, and structured summary.",
              {"type": "object", "properties": {"conversation_id": {"type": "string"}}, "required": ["conversation_id"]},
              t_get_conversation_summary),
        _Tool("get_conversation_messages", "Get messages for one indexed conversation.",
              {"type": "object", "properties": {"conversation_id": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["conversation_id"]},
              t_get_conversation_messages),
        _Tool("get_conversation_chunks", "Get chunks for one indexed conversation.",
              {"type": "object", "properties": {"conversation_id": {"type": "string"}}, "required": ["conversation_id"]},
              t_get_conversation_chunks),
        _Tool("list_projects", "List active projects derived from indexed material. "
              "By default excludes roleplay and jailbreak content.",
              {"type": "object", "properties": {"limit": {"type": "integer"}, **fp}},
              t_list_projects),
        _Tool("get_project", "Get a project page (linked conversations + decisions + open loops + entities).",
              {"type": "object", "properties": {"project_id": {"type": "string"}}, "required": ["project_id"]},
              t_get_project),
        _Tool("get_project_timeline", "Timeline of conversations linked to a project.",
              {"type": "object", "properties": {"project_id": {"type": "string"}}, "required": ["project_id"]},
              t_get_project_timeline),
        _Tool("list_open_loops", "List currently open loops across indexed material. "
              "By default excludes roleplay and jailbreak content.",
              {"type": "object", "properties": {"limit": {"type": "integer"}, **fp}},
              t_list_open_loops),
        _Tool("list_decisions", "List decisions across indexed material. "
              "By default excludes roleplay and jailbreak content.",
              {"type": "object", "properties": {"limit": {"type": "integer"}, **fp}},
              t_list_decisions),
        _Tool("list_entities", "List recurring entities across indexed material with type disambiguation.",
              {"type": "object", "properties": {"limit": {"type": "integer"}, **fp}},
              t_list_entities),
        _Tool("list_groups", "List thematic conversation groups (broad/fine) with visible (indexed) member counts.",
              {"type": "object", "properties": {"level": {"type": "string", "enum": ["broad", "fine"]}}},
              t_list_groups),
        _Tool("get_group", "Show one group with its indexed members (no private/quarantined/pending leakage).",
              {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
              t_get_group),
        _Tool("inspect_conversation_storage", "Metadata-only audit of what is stored for a conversation.",
              {"type": "object", "properties": {"conversation_id": {"type": "string"}}, "required": ["conversation_id"]},
              t_inspect_conversation_storage),
        _Tool("generate_profile", "Generate a narrative profile of the indexed user organized by topic: "
              "active projects, recent interests, dormant threads, recurring preoccupations, stylistic tendencies. "
              "Cached with 7-day TTL.",
              {"type": "object", "properties": {
                  "focus": {"type": "array", "items": {"type": "string"},
                            "description": "Optional list of topics to focus the profile on."}}},
              t_generate_profile),
        _Tool("find_related", "Find past conversations most relevant to a given context description. "
              "Uses semantic similarity for proactive surfacing when current work resembles past conversations.",
              {"type": "object", "properties": {
                  "context": {"type": "string",
                              "description": "Free-text description of current context "
                                             "(e.g. 'working on a policy paper about AI governance')"},
                  "limit": {"type": "integer", "description": "Max results (default 10)"}},
               "required": ["context"]},
              t_find_related),
    ]

    # Opt-in write tools. Only registered when <vault>/mcp_config.json sets
    # allow_writes=true. Every successful call is audit-logged in
    # <vault>/logs/mcp_mutations.jsonl with metadata only (no content).
    if writes_mod.writes_enabled(vault):
        def _wrap(fn):
            def _handler(args):
                ok, payload = fn(vault, store, args)
                if ok:
                    return _ok(payload)
                return _err(str(payload.get("error", "write failed")))
            return _handler

        tools.extend([
            _Tool("set_group_label", "Update a thematic group's llm_label (e.g. correct a misnamed cluster).",
                  {"type": "object",
                   "properties": {"group_id": {"type": "string"},
                                  "label": {"type": "string"}},
                   "required": ["group_id", "label"]},
                  _wrap(writes_mod.set_group_label)),
            _Tool("add_tag", "Add one or more manual tags to an INDEXED conversation.",
                  {"type": "object",
                   "properties": {"conversation_id": {"type": "string"},
                                  "tags": {"type": "array",
                                           "items": {"type": "string"}}},
                   "required": ["conversation_id", "tags"]},
                  _wrap(writes_mod.add_tag)),
            _Tool("remove_tag", "Remove one or more manual tags from an INDEXED conversation.",
                  {"type": "object",
                   "properties": {"conversation_id": {"type": "string"},
                                  "tags": {"type": "array",
                                           "items": {"type": "string"}}},
                   "required": ["conversation_id", "tags"]},
                  _wrap(writes_mod.remove_tag)),
            _Tool("rename_derived_object", "Rename a project/entity/decision/etc. (title only; object id and kind unchanged).",
                  {"type": "object",
                   "properties": {"object_id": {"type": "string"},
                                  "title": {"type": "string"}},
                   "required": ["object_id", "title"]},
                  _wrap(writes_mod.rename_derived_object)),
        ])

    return {t.name: t for t in tools}


# --- JSON-RPC framing -------------------------------------------------------

def _make_rpc_response(rid, result=None, error=None) -> dict:
    msg = {"jsonrpc": "2.0", "id": rid}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    return msg


def _handle(message: dict, tools: dict[str, _Tool]) -> dict | None:
    method = message.get("method")
    rid = message.get("id")
    params = message.get("params") or {}
    if method == "initialize":
        return _make_rpc_response(rid, result={
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": "threadatlas", "version": TA_VERSION},
            "capabilities": {"tools": {"listChanged": False}},
        })
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _make_rpc_response(rid, result={
            "tools": [
                {"name": t.name, "description": t.description, "inputSchema": t.schema}
                for t in tools.values()
            ]
        })
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = tools.get(name)
        if tool is None:
            return _make_rpc_response(rid, error={"code": -32601, "message": f"Unknown tool: {name}"})
        try:
            return _make_rpc_response(rid, result=tool.fn(args))
        except Exception as e:
            tb = traceback.format_exc()
            return _make_rpc_response(rid, error={"code": -32000, "message": f"{e}\n{tb}"})
    if rid is None:
        return None  # Unhandled notification; ignore.
    return _make_rpc_response(rid, error={"code": -32601, "message": f"Method not supported: {method}"})


def serve(vault_path, *, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> int:
    """Run the stdio MCP loop until EOF.

    The function is synchronous and uses only the standard library. It does
    not open any sockets; it reads from stdin and writes to stdout.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    vault = open_vault(vault_path)
    store = open_store(vault)
    try:
        tools = build_tools(vault, store)
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = _handle(msg, tools)
            if response is not None:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()
    finally:
        store.close()
    return 0
