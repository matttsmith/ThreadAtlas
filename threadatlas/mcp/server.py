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
from ..core.models import MCP_VISIBLE_STATES
from ..core.vault import Vault, open_vault
from ..search import (
    list_decisions,
    list_entities,
    list_open_loops,
    project_timeline,
    project_view,
    search_chunks,
    search_conversations,
)
from ..search.search import list_projects
from ..store import open_store, Store


PROTOCOL_VERSION = "2024-11-05"


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

    def t_search_conversations(args: dict) -> dict:
        q = str(args.get("query") or "")
        limit = int(args.get("limit") or 25)
        hits = search_conversations(store, q, visible_states=visible, limit=limit)
        return _ok([h.__dict__ for h in hits])

    def t_search_chunks(args: dict) -> dict:
        q = str(args.get("query") or "")
        limit = int(args.get("limit") or 25)
        hits = search_chunks(store, q, visible_states=visible, limit=limit)
        return _ok([h.__dict__ for h in hits])

    def t_get_conversation_summary(args: dict) -> dict:
        cid = str(args.get("conversation_id") or "")
        c = store.get_conversation(cid)
        if c is None or c.state not in MCP_VISIBLE_STATES:
            return _err(f"Conversation not visible: {cid}")
        return _ok({
            "conversation_id": c.conversation_id,
            "source": c.source,
            "title": c.title,
            "summary_short": c.summary_short,
            "summary_long": c.summary_long,
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
        return _ok(list_projects(store, visible_states=visible, limit=int(args.get("limit") or 200)))

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
        return _ok(list_open_loops(store, visible_states=visible, limit=int(args.get("limit") or 200)))

    def t_list_decisions(args: dict) -> dict:
        return _ok(list_decisions(store, visible_states=visible, limit=int(args.get("limit") or 200)))

    def t_list_entities(args: dict) -> dict:
        return _ok(list_entities(store, visible_states=visible, limit=int(args.get("limit") or 200)))

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

    tools = [
        _Tool("search_conversations", "Keyword search over indexed conversations.",
              {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]},
              t_search_conversations),
        _Tool("search_chunks", "Keyword search over chunks of indexed conversations.",
              {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]},
              t_search_chunks),
        _Tool("get_conversation_summary", "Get summary metadata for one indexed conversation.",
              {"type": "object", "properties": {"conversation_id": {"type": "string"}}, "required": ["conversation_id"]},
              t_get_conversation_summary),
        _Tool("get_conversation_messages", "Get messages for one indexed conversation.",
              {"type": "object", "properties": {"conversation_id": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["conversation_id"]},
              t_get_conversation_messages),
        _Tool("get_conversation_chunks", "Get chunks for one indexed conversation.",
              {"type": "object", "properties": {"conversation_id": {"type": "string"}}, "required": ["conversation_id"]},
              t_get_conversation_chunks),
        _Tool("list_projects", "List active projects derived from indexed material.",
              {"type": "object", "properties": {"limit": {"type": "integer"}}},
              t_list_projects),
        _Tool("get_project", "Get a project page (linked conversations + decisions + open loops + entities).",
              {"type": "object", "properties": {"project_id": {"type": "string"}}, "required": ["project_id"]},
              t_get_project),
        _Tool("get_project_timeline", "Timeline of conversations linked to a project.",
              {"type": "object", "properties": {"project_id": {"type": "string"}}, "required": ["project_id"]},
              t_get_project_timeline),
        _Tool("list_open_loops", "List currently open loops across indexed material.",
              {"type": "object", "properties": {"limit": {"type": "integer"}}},
              t_list_open_loops),
        _Tool("list_decisions", "List decisions across indexed material.",
              {"type": "object", "properties": {"limit": {"type": "integer"}}},
              t_list_decisions),
        _Tool("list_entities", "List recurring entities across indexed material.",
              {"type": "object", "properties": {"limit": {"type": "integer"}}},
              t_list_entities),
        _Tool("inspect_conversation_storage", "Metadata-only audit of what is stored for a conversation.",
              {"type": "object", "properties": {"conversation_id": {"type": "string"}}, "required": ["conversation_id"]},
              t_inspect_conversation_storage),
    ]
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
