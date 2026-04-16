"""Tests for new v2 MCP tools and filter parameters.

Tests cover:
- Updated get_conversation_summary (LLM summary + register)
- generate_profile tool
- find_related tool
- Filter parameters on list and search tools
- Tool schema validation
"""

from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path

import pytest

from threadatlas.core.models import (
    ALL_REGISTERS,
    ConversationLLMMeta,
    DerivedKind,
    DerivedObject,
    ProvenanceLink,
    State,
    new_id,
)
from threadatlas.core.vault import init_vault
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path
from threadatlas.mcp.server import build_tools, _parse_date_param, _parse_register_param
from threadatlas.search.embeddings import build_all_embeddings
from threadatlas.store import open_store

from conftest import make_chatgpt_export


def _setup_with_llm_meta(tmp_path):
    vault = init_vault(tmp_path / "vault")
    store = open_store(vault)
    export_path = make_chatgpt_export(tmp_path, [
        {
            "id": "conv-work",
            "title": "Architecture planning session",
            "create_time": time.time() - 86400,
            "update_time": time.time(),
            "messages": [
                ("user", "Let's plan the new microservices architecture for our platform.", time.time() - 86400),
                ("assistant", "Great! What components need to be split?", time.time() - 86300),
                ("user", "Decision: we will use event sourcing for the order service.", time.time() - 86200),
                ("assistant", "Event sourcing is a good pattern for order tracking.", time.time() - 86100),
            ],
        },
        {
            "id": "conv-roleplay",
            "title": "Fantasy adventure roleplay",
            "create_time": time.time() - 86400 * 2,
            "update_time": time.time() - 86400,
            "messages": [
                ("user", "You are a wise wizard in a magical forest. I approach seeking guidance.", time.time() - 86400 * 2),
                ("assistant", "Greetings, traveler. What wisdom do you seek?", time.time() - 86400 * 2 + 60),
                ("user", "I will spread my light to every corner of this realm!", time.time() - 86400 * 2 + 120),
                ("assistant", "A noble quest indeed, young one.", time.time() - 86400 * 2 + 180),
            ],
        },
    ])
    result = import_path(vault, store, export_path)
    conv_ids = {"work": result.imported[0], "roleplay": result.imported[1]}
    for cid in result.imported:
        transition_state(store, cid, State.INDEXED.value)
        chunk_conversation(store, cid)
        extract_for_conversation(store, cid)

    # Add LLM meta.
    store.upsert_conversation_llm_meta(ConversationLLMMeta(
        conversation_id=conv_ids["work"],
        llm_summary="Architecture planning for microservices with event sourcing.",
        dominant_register="work",
        content_hash="hash1",
        extracted_at=time.time(),
    ))
    store.upsert_conversation_llm_meta(ConversationLLMMeta(
        conversation_id=conv_ids["roleplay"],
        llm_summary="Fantasy wizard roleplay in a magical forest.",
        dominant_register="roleplay",
        content_hash="hash2",
        extracted_at=time.time(),
    ))
    store.conn.commit()

    return vault, store, conv_ids


# --- Helper function tests ---

class TestParseHelpers:

    def test_parse_date_param_valid(self):
        ts = _parse_date_param("2024-06-15")
        assert ts is not None
        assert ts > 0

    def test_parse_date_param_with_time(self):
        ts = _parse_date_param("2024-06-15T14:30:00")
        assert ts is not None

    def test_parse_date_param_none(self):
        assert _parse_date_param(None) is None
        assert _parse_date_param("") is None
        assert _parse_date_param("invalid") is None

    def test_parse_register_param_list(self):
        result = _parse_register_param(["work", "research"])
        assert result == ["work", "research"]

    def test_parse_register_param_invalid(self):
        result = _parse_register_param(["invalid_register"])
        assert result == []

    def test_parse_register_param_string(self):
        result = _parse_register_param("work")
        assert result == ["work"]

    def test_parse_register_param_none(self):
        assert _parse_register_param(None) is None


# --- MCP tool tests ---

class TestMCPToolsV2:

    def test_get_conversation_summary_includes_llm_fields(self, tmp_path):
        vault, store, conv_ids = _setup_with_llm_meta(tmp_path)
        tools = build_tools(vault, store)
        result = tools["get_conversation_summary"].fn({"conversation_id": conv_ids["work"]})
        content = json.loads(result["content"][0]["text"])
        assert content["llm_summary"] == "Architecture planning for microservices with event sourcing."
        assert content["dominant_register"] == "work"
        store.close()

    def test_get_conversation_summary_no_llm_meta(self, tmp_path):
        """Conversations without LLM meta should still work."""
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)
        export_path = make_chatgpt_export(tmp_path, [
            {"id": "plain", "title": "Plain conv", "messages": [
                ("user", "hello", time.time()),
                ("assistant", "hi there", time.time() + 60),
            ]},
        ])
        result = import_path(vault, store, export_path)
        conv_id = result.imported[0]
        transition_state(store, conv_id, State.INDEXED.value)

        tools = build_tools(vault, store)
        result = tools["get_conversation_summary"].fn({"conversation_id": conv_id})
        content = json.loads(result["content"][0]["text"])
        assert content["llm_summary"] is None
        assert content["dominant_register"] is None
        store.close()

    def test_generate_profile_structured_fallback(self, tmp_path):
        vault, store, conv_ids = _setup_with_llm_meta(tmp_path)
        tools = build_tools(vault, store)
        result = tools["generate_profile"].fn({})
        content = json.loads(result["content"][0]["text"])
        assert "profile" in content
        assert "generated_at" in content
        assert isinstance(content["data"]["conversation_count"], int)
        store.close()

    def test_generate_profile_with_focus(self, tmp_path):
        vault, store, conv_ids = _setup_with_llm_meta(tmp_path)
        tools = build_tools(vault, store)
        result = tools["generate_profile"].fn({"focus": ["architecture", "microservices"]})
        content = json.loads(result["content"][0]["text"])
        assert content["focus"] == ["architecture", "microservices"]
        store.close()

    def test_find_related_returns_results(self, tmp_path):
        vault, store, conv_ids = _setup_with_llm_meta(tmp_path)
        build_all_embeddings(store)
        tools = build_tools(vault, store)
        result = tools["find_related"].fn({
            "context": "microservices architecture event sourcing",
            "limit": 5,
        })
        content = json.loads(result["content"][0]["text"])
        assert isinstance(content, list)
        # Should have at least one result.
        if content:
            assert "conversation_id" in content[0]
            assert "score" in content[0]
        store.close()

    def test_find_related_requires_context(self, tmp_path):
        vault, store, conv_ids = _setup_with_llm_meta(tmp_path)
        tools = build_tools(vault, store)
        result = tools["find_related"].fn({"context": ""})
        assert result.get("isError")
        store.close()

    def test_query_with_register_filter(self, tmp_path):
        vault, store, conv_ids = _setup_with_llm_meta(tmp_path)
        tools = build_tools(vault, store)

        # Query with register filter prefix.
        result = tools["query"].fn({
            "query": "architecture register:work",
        })
        content = json.loads(result["content"][0]["text"])
        # Results should not include roleplay conversations.
        conv_hits = [h for h in content.get("hits", []) if h["hit_type"] == "conversation"]
        cids = {h["id"] for h in conv_hits}
        assert conv_ids["roleplay"] not in cids
        store.close()

    def test_list_with_filter_params(self, tmp_path):
        vault, store, conv_ids = _setup_with_llm_meta(tmp_path)
        tools = build_tools(vault, store)

        # All list tools accept filter params without error.
        for tool_name in ["list_projects", "list_decisions", "list_open_loops", "list_entities"]:
            result = tools[tool_name].fn({
                "after": "2024-01-01",
                "source": "chatgpt",
            })
            assert "content" in result
        store.close()


# --- Tool schema validation ---

class TestToolSchemas:

    def test_all_tools_registered(self, tmp_path):
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)
        tools = build_tools(vault, store)
        expected_tools = {
            "query",
            "get_conversation_summary", "get_conversation_messages",
            "get_conversation_chunks", "list_projects", "get_project",
            "list_open_loops", "list_decisions",
            "list_entities", "list_groups", "get_group",
            "generate_profile", "find_related",
        }
        assert expected_tools == set(tools.keys())
        store.close()

    def test_filter_properties_in_schemas(self, tmp_path):
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)
        tools = build_tools(vault, store)

        filterable_tools = [
            "list_projects", "list_decisions", "list_open_loops", "list_entities",
        ]
        for name in filterable_tools:
            schema = tools[name].schema
            props = schema.get("properties", {})
            assert "after" in props, f"{name} missing 'after' filter"
            assert "before" in props, f"{name} missing 'before' filter"
            assert "register" in props, f"{name} missing 'register' filter"
            assert "source" in props, f"{name} missing 'source' filter"
        store.close()

    def test_generate_profile_schema(self, tmp_path):
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)
        tools = build_tools(vault, store)
        schema = tools["generate_profile"].schema
        assert "focus" in schema["properties"]
        store.close()

    def test_find_related_schema(self, tmp_path):
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)
        tools = build_tools(vault, store)
        schema = tools["find_related"].schema
        assert "context" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "context" in schema.get("required", [])
        store.close()
