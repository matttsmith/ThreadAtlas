"""Parser unit tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from threadatlas.ingest import get_parser
from threadatlas.ingest.chatgpt import ChatGPTParser, _flatten_content, _linearize
from threadatlas.ingest.claude import ClaudeParser, _flatten_claude_content, _normalize_role


# ---------------------------------------------------------------------------
# ChatGPT parser
# ---------------------------------------------------------------------------

def test_chatgpt_can_handle(chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "Hi", "messages": [("user", "hi", 1.0), ("assistant", "hello", 2.0)]},
    ])
    assert ChatGPTParser().can_handle(path) is True


def test_chatgpt_iter_conversations(chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "Project CHS planning",
         "messages": [
             ("user", "What about CHS?", 1700000000.0),
             ("assistant", "CHS stands for Center for Health Security.", 1700000010.0),
         ]},
        {"title": "Personal stuff",
         "messages": [
             ("user", "I feel anxious", 1700001000.0),
             ("assistant", "I'm sorry to hear", 1700001010.0),
         ]},
    ])
    parser = ChatGPTParser()
    convs = list(parser.iter_conversations(path))
    assert len(convs) == 2
    assert convs[0].source == "chatgpt"
    assert convs[0].title == "Project CHS planning"
    assert convs[0].message_count == 2
    assert convs[0].messages[0].role == "user"
    assert convs[0].messages[0].content_text == "What about CHS?"
    assert convs[0].messages[1].role == "assistant"
    # Ordinals are sequential per conversation.
    assert [m.ordinal for m in convs[0].messages] == [0, 1]


def test_chatgpt_handles_zip(tmp_path, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "Z", "messages": [("user", "x", 1.0), ("assistant", "y", 2.0)]},
    ])
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(path / "conversations.json", "conversations.json")
    parser = ChatGPTParser()
    assert parser.can_handle(zip_path)
    convs = list(parser.iter_conversations(zip_path))
    assert convs[0].title == "Z"


def test_chatgpt_flatten_content_variants():
    assert _flatten_content({"content_type": "text", "parts": ["a", "b"]}) == "a\n\nb"
    assert _flatten_content({"content_type": "code", "language": "py", "text": "print(1)"}).startswith("```py\n")
    assert _flatten_content({"content_type": "multimodal_text",
                             "parts": [{"text": "hello"}, {"asset_pointer": "file:asdf"}]}) == "hello\n\n[asset:file:asdf]"
    assert _flatten_content(None) == ""
    assert _flatten_content("plain") == "plain"


def test_chatgpt_linearize_uses_current_node():
    mapping = {
        "r": {"id": "r", "parent": None, "children": ["a", "b"], "message": None},
        "a": {"id": "a", "parent": "r", "children": [], "message": {"id": "a", "author": {"role": "user"}, "content": {"content_type": "text", "parts": ["A"]}, "create_time": 1.0}},
        "b": {"id": "b", "parent": "r", "children": [], "message": {"id": "b", "author": {"role": "user"}, "content": {"content_type": "text", "parts": ["B"]}, "create_time": 2.0}},
    }
    nodes = _linearize(mapping, current_node="a")
    ids = [n["id"] for n in nodes]
    assert "a" in ids
    assert "b" not in ids  # follows current_node branch only


# ---------------------------------------------------------------------------
# Claude parser
# ---------------------------------------------------------------------------

def test_claude_can_handle(claude_export_factory):
    path = claude_export_factory([
        {"name": "Hi", "messages": [("human", "hi", "2026-04-15T12:00:00Z"),
                                    ("assistant", "hello", "2026-04-15T12:00:05Z")]},
    ])
    assert ClaudeParser().can_handle(path)


def test_claude_iter_conversations(claude_export_factory):
    path = claude_export_factory([
        {"name": "Project Sprint Planning",
         "messages": [
             ("human", "Project Sprint kickoff. Decisions?", "2026-04-15T12:00:00Z"),
             ("assistant", "We agreed to ship v1 next month.", "2026-04-15T12:00:05Z"),
         ]},
    ])
    parser = ClaudeParser()
    convs = list(parser.iter_conversations(path))
    assert len(convs) == 1
    c = convs[0]
    assert c.source == "claude"
    assert c.title == "Project Sprint Planning"
    assert c.messages[0].role == "user"
    assert c.messages[1].role == "assistant"


def test_claude_normalize_role():
    assert _normalize_role("human") == "user"
    assert _normalize_role("assistant") == "assistant"
    assert _normalize_role(None) == "other"
    assert _normalize_role("system") == "system"


def test_claude_flatten_structured_content():
    msg = {"content": [
        {"type": "text", "text": "Hello"},
        {"type": "tool_use", "name": "search"},
        {"type": "tool_result", "content": [{"type": "text", "text": "found"}]},
    ], "attachments": [{"file_name": "spec.pdf"}]}
    out = _flatten_claude_content(msg)
    assert "Hello" in out
    assert "[tool_use:search]" in out
    assert "[tool_result] found" in out
    assert "[attachment:spec.pdf]" in out


# ---------------------------------------------------------------------------
# Autodetect
# ---------------------------------------------------------------------------

def test_autodetect_picks_chatgpt(chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "X", "messages": [("user", "x", 1.0), ("assistant", "y", 2.0)]},
    ])
    parser = get_parser("auto")
    convs = list(parser.iter_conversations(path))
    assert convs[0].source == "chatgpt"


def test_autodetect_picks_claude(claude_export_factory):
    path = claude_export_factory([
        {"name": "X", "messages": [("human", "x", "2026-04-15T12:00:00Z"),
                                   ("assistant", "y", "2026-04-15T12:00:05Z")]},
    ])
    parser = get_parser("auto")
    convs = list(parser.iter_conversations(path))
    assert convs[0].source == "claude"
