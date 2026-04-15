"""MCP server tests.

We exercise the JSON-RPC handler in-process (no actual stdio plumbing) and
verify that visibility rules are enforced at the tool level. The MCP layer
must NOT be a backdoor around state checks.
"""

from __future__ import annotations

import io
import json

from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path
from threadatlas.mcp.server import build_tools, _handle, serve


def _seed_indexed_and_private(tmp_vault, store, factory):
    path = factory([
        {"title": "Project Atlas planning",
         "messages": [
             ("user", "Project Atlas. Decision: I will pick the smaller team.", 1.0),
             ("assistant", "Sounds good. We agreed to ship in April.", 2.0),
             ("user", "TODO: follow up on staffing.", 3.0),
             ("assistant", "Noted.", 4.0),
         ]},
        {"title": "Therapy session",
         "messages": [
             ("user", "I feel anxious", 1.0),
             ("assistant", "Tell me more", 2.0),
         ]},
    ])
    res = import_path(tmp_vault, store, path)
    cid_a, cid_b = res.imported
    transition_state(store, cid_a, State.INDEXED.value)
    transition_state(store, cid_b, State.PRIVATE.value)
    chunk_conversation(store, cid_a)
    extract_for_conversation(store, cid_a)
    return cid_a, cid_b


def _call(tools, name, args):
    handler = tools[name]
    return handler.fn(args)


def test_search_conversations_only_returns_indexed(tmp_vault, store, chatgpt_export_factory):
    cid_a, cid_b = _seed_indexed_and_private(tmp_vault, store, chatgpt_export_factory)
    tools = build_tools(tmp_vault, store)
    result = _call(tools, "search_conversations", {"query": "anxious"})
    text = result["content"][0]["text"]
    payload = json.loads(text)
    # No private content should appear.
    assert all(item["conversation_id"] != cid_b for item in payload)


def test_get_conversation_summary_refuses_private(tmp_vault, store, chatgpt_export_factory):
    cid_a, cid_b = _seed_indexed_and_private(tmp_vault, store, chatgpt_export_factory)
    tools = build_tools(tmp_vault, store)
    result = _call(tools, "get_conversation_summary", {"conversation_id": cid_b})
    assert result.get("isError") is True


def test_get_conversation_summary_allows_indexed(tmp_vault, store, chatgpt_export_factory):
    cid_a, cid_b = _seed_indexed_and_private(tmp_vault, store, chatgpt_export_factory)
    tools = build_tools(tmp_vault, store)
    result = _call(tools, "get_conversation_summary", {"conversation_id": cid_a})
    assert "isError" not in result
    payload = json.loads(result["content"][0]["text"])
    assert payload["conversation_id"] == cid_a
    assert payload["state"] == State.INDEXED.value


def test_inspect_conversation_storage_redacts_title_for_private(tmp_vault, store, chatgpt_export_factory):
    cid_a, cid_b = _seed_indexed_and_private(tmp_vault, store, chatgpt_export_factory)
    tools = build_tools(tmp_vault, store)
    result = _call(tools, "inspect_conversation_storage", {"conversation_id": cid_b})
    payload = json.loads(result["content"][0]["text"])
    assert payload["mcp_visible"] is False
    assert payload["title"] == "[redacted]"
    # Counts are still shown so the user can audit storage even for private.
    assert payload["message_count"] >= 1


def test_jsonrpc_initialize_and_tools_list(tmp_vault, store, chatgpt_export_factory):
    cid_a, cid_b = _seed_indexed_and_private(tmp_vault, store, chatgpt_export_factory)
    tools = build_tools(tmp_vault, store)
    init = _handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, tools)
    assert init["result"]["serverInfo"]["name"] == "threadatlas"
    listed = _handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, tools)
    names = {t["name"] for t in listed["result"]["tools"]}
    assert "search_conversations" in names
    assert "get_project" in names
    # No mutating tools in v1.
    assert not any(n.startswith(("delete_", "set_", "approve_", "quarantine_")) for n in names)


def test_serve_respects_eof(tmp_vault, store, chatgpt_export_factory):
    """Smoke-test serve() with an in-memory stdin that closes immediately."""
    _seed_indexed_and_private(tmp_vault, store, chatgpt_export_factory)
    store.close()  # serve() will reopen from disk.
    stdin = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n')
    stdout = io.StringIO()
    rc = serve(tmp_vault.root, stdin=stdin, stdout=stdout)
    assert rc == 0
    out = stdout.getvalue().strip()
    assert out
    payload = json.loads(out.splitlines()[0])
    assert "result" in payload
    assert any(t["name"] == "search_conversations" for t in payload["result"]["tools"])
