"""MCP query tool tests.

Validates the JSON-RPC surface for the ``query`` tool: response envelope,
content structure, visibility enforcement, limit handling, and filter
pass-through. Uses the rich_corpus fixture for realistic data.
"""

from __future__ import annotations

import json

from threadatlas.mcp.server import build_tools, _handle

from corpus_fixtures import rich_corpus  # noqa: F401 — pytest fixture


def _call(tools, name: str, args: dict) -> dict:
    """Invoke an MCP tool and return the raw result dict."""
    return tools[name].fn(args)


def _payload(result: dict):
    """Extract the parsed JSON payload from an MCP _ok() response."""
    return json.loads(result["content"][0]["text"])


def _rpc(tools, name: str, args: dict, rid: int = 1) -> dict:
    """Invoke via the JSON-RPC _handle path and return the response."""
    msg = {
        "jsonrpc": "2.0",
        "id": rid,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }
    return _handle(msg, tools)


# ===================================================================
# Basics: tool registration & response shape
# ===================================================================


class TestQueryToolRegistration:
    """The query tool is registered and listed."""

    def test_query_tool_exists(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        assert "query" in tools

    def test_query_is_first_tool(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        first_name = next(iter(tools))
        assert first_name == "query"

    def test_query_listed_in_tools_list(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        resp = _handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            tools,
        )
        tool_names = [t["name"] for t in resp["result"]["tools"]]
        assert "query" in tool_names

    def test_query_tool_has_schema(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        resp = _handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            tools,
        )
        query_tool = next(t for t in resp["result"]["tools"] if t["name"] == "query")
        schema = query_tool["inputSchema"]
        assert "query" in schema["properties"]
        assert "query" in schema.get("required", [])


class TestQueryToolResponseShape:
    """Verify the MCP JSON envelope and payload structure."""

    def test_ok_envelope(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        result = _call(tools, "query", {"query": "Atlas"})
        assert "content" in result
        assert result["content"][0]["type"] == "text"
        assert "isError" not in result

    def test_payload_fields(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        result = _call(tools, "query", {"query": "Atlas"})
        payload = _payload(result)
        assert "raw_query" in payload
        assert "filters" in payload
        assert "hits" in payload
        assert "total_by_type" in payload
        assert "elapsed_ms" in payload

    def test_hit_fields(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        result = _call(tools, "query", {"query": "Atlas"})
        payload = _payload(result)
        assert len(payload["hits"]) > 0
        hit = payload["hits"][0]
        assert "hit_type" in hit
        assert "id" in hit
        assert "title" in hit
        assert "snippet" in hit
        assert "score" in hit
        assert "metadata" in hit

    def test_raw_query_echoed(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        result = _call(tools, "query", {"query": "source:chatgpt Atlas"})
        payload = _payload(result)
        assert payload["raw_query"] == "source:chatgpt Atlas"

    def test_filters_parsed(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        result = _call(tools, "query", {"query": "source:chatgpt Atlas"})
        payload = _payload(result)
        assert "sources" in payload["filters"]
        assert "chatgpt" in payload["filters"]["sources"]
        assert payload["filters"]["text"] == "Atlas"

    def test_elapsed_ms_positive(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        result = _call(tools, "query", {"query": "Atlas"})
        payload = _payload(result)
        assert payload["elapsed_ms"] >= 0


# ===================================================================
# Keyword search via MCP
# ===================================================================


class TestQueryToolKeywordSearch:
    """Keyword searches through the MCP tool."""

    def test_keyword_finds_conversations(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "Kubernetes"}))
        conv_hits = [h for h in payload["hits"] if h["hit_type"] == "conversation"]
        assert len(conv_hits) >= 1

    def test_keyword_finds_derived_objects(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "Atlas"}))
        derived = [h for h in payload["hits"] if h["hit_type"] == "derived_object"]
        assert len(derived) >= 1

    def test_no_results_returns_empty_hits(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "zzzznonexistent999"}))
        assert payload["hits"] == []
        assert payload["total_by_type"] == {}


# ===================================================================
# Visibility enforcement via MCP
# ===================================================================


class TestQueryToolVisibility:
    """MCP query tool must never expose private/quarantined data."""

    def test_private_excluded(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "anxious therapy"}))
        all_ids = {h["id"] for h in payload["hits"]}
        assert rich_corpus.conv_ids["therapy"] not in all_ids

    def test_quarantined_excluded(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "API keys secrets"}))
        all_ids = {h["id"] for h in payload["hits"]}
        assert rich_corpus.conv_ids["quarantined"] not in all_ids

    def test_empty_query_excludes_private(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": ""}))
        all_ids = {h["id"] for h in payload["hits"]}
        assert rich_corpus.conv_ids["therapy"] not in all_ids
        assert rich_corpus.conv_ids["quarantined"] not in all_ids


# ===================================================================
# Filter pass-through via MCP
# ===================================================================


class TestQueryToolFilters:
    """Verify filters work end-to-end through the MCP layer."""

    def test_source_filter(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "source:claude"}))
        conv_hits = [h for h in payload["hits"] if h["hit_type"] == "conversation"]
        for h in conv_hits:
            assert h["metadata"]["source"] == "claude"

    def test_tag_filter(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "tag:finance"}))
        conv_hits = [h for h in payload["hits"] if h["hit_type"] == "conversation"]
        assert any(h["id"] == rich_corpus.conv_ids["q4"] for h in conv_hits)

    def test_kind_filter(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "kind:decision"}))
        for h in payload["hits"]:
            assert h["hit_type"] == "derived_object"
            assert h["metadata"]["kind"] == "decision"

    def test_project_filter(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        pid = rich_corpus.project_id
        payload = _payload(_call(tools, "query", {"query": f"project:{pid}"}))
        conv_hits = [h for h in payload["hits"] if h["hit_type"] == "conversation"]
        conv_ids = {h["id"] for h in conv_hits}
        assert rich_corpus.conv_ids["atlas"] in conv_ids

    def test_has_chunks_filter(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "has:chunks"}))
        conv_hits = [h for h in payload["hits"] if h["hit_type"] == "conversation"]
        assert len(conv_hits) >= 4

    def test_combined_filters(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "source:chatgpt tag:architecture"}))
        conv_hits = [h for h in payload["hits"] if h["hit_type"] == "conversation"]
        for h in conv_hits:
            assert h["metadata"]["source"] == "chatgpt"


# ===================================================================
# Limit handling
# ===================================================================


class TestQueryToolLimit:
    """Limit parameter via MCP."""

    def test_custom_limit(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": "", "limit": 2}))
        assert len(payload["hits"]) <= 2

    def test_default_limit(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        payload = _payload(_call(tools, "query", {"query": ""}))
        assert len(payload["hits"]) <= 25

    def test_limit_zero_or_missing(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        # Limit 0 should fall back to default (25) per int(args.get("limit") or 25).
        payload = _payload(_call(tools, "query", {"query": "", "limit": 0}))
        assert len(payload["hits"]) <= 25


# ===================================================================
# JSON-RPC integration (via _handle)
# ===================================================================


class TestQueryToolJsonRpc:
    """Exercise the query tool through the JSON-RPC _handle path."""

    def test_rpc_success(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        resp = _rpc(tools, "query", {"query": "Atlas"})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "result" in resp
        assert "error" not in resp

    def test_rpc_payload_structure(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        resp = _rpc(tools, "query", {"query": "Atlas"})
        content = resp["result"]["content"]
        assert content[0]["type"] == "text"
        payload = json.loads(content[0]["text"])
        assert "hits" in payload
        assert "raw_query" in payload

    def test_rpc_unknown_tool_returns_error(self, rich_corpus):
        tools = build_tools(rich_corpus.vault, rich_corpus.store)
        resp = _rpc(tools, "nonexistent_tool", {"query": "Atlas"})
        assert "error" in resp
        assert resp["error"]["code"] == -32601
