"""Shared pytest fixtures.

We expose a fresh vault per test (so DB writes never leak across tests) and
helpers for synthesizing realistic ChatGPT / Claude export shapes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from threadatlas.core.vault import Vault, init_vault
from threadatlas.store import open_store



# ---------------------------------------------------------------------------
# Fresh per-test vault.
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_vault(tmp_path: Path) -> Vault:
    return init_vault(tmp_path / "vault")


@pytest.fixture
def store(tmp_vault):
    s = open_store(tmp_vault)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# ChatGPT export fixtures.
# ---------------------------------------------------------------------------

def _chatgpt_node(node_id, parent, children, role, text, ts):
    return {
        "id": node_id,
        "parent": parent,
        "children": list(children),
        "message": None if role is None else {
            "id": node_id,
            "author": {"role": role},
            "content": {"content_type": "text", "parts": [text]},
            "create_time": ts,
        },
    }


def make_chatgpt_export(tmp_path: Path, conversations: list[dict]) -> Path:
    """Write a synthetic conversations.json that looks like ChatGPT's export.

    Each conversation in ``conversations`` is a dict like
    ``{"title": "...", "messages": [(role, text, ts), ...], "id": "...", "create_time": float, "update_time": float}``.
    """
    out_dir = tmp_path / "chatgpt_export"
    out_dir.mkdir()
    payload = []
    for i, conv in enumerate(conversations):
        ts0 = conv.get("create_time", time.time() - (len(conversations) - i) * 60)
        msgs = conv["messages"]
        # Build a linear tree: root -> m1 -> m2 -> ...
        mapping = {}
        prev_id = None
        node_ids = []
        # System root for realism (some exports include one).
        root_id = f"root-{i}"
        mapping[root_id] = _chatgpt_node(root_id, None, [], None, None, None)
        prev_id = root_id
        for j, (role, text, ts) in enumerate(msgs):
            nid = f"node-{i}-{j}"
            node_ids.append(nid)
            mapping[nid] = _chatgpt_node(nid, prev_id, [], role, text, ts)
            mapping[prev_id]["children"].append(nid)
            prev_id = nid
        payload.append({
            "id": conv.get("id", f"chatgpt-{i}"),
            "conversation_id": conv.get("id", f"chatgpt-{i}"),
            "title": conv["title"],
            "create_time": ts0,
            "update_time": conv.get("update_time", ts0 + 3600),
            "current_node": node_ids[-1] if node_ids else None,
            "mapping": mapping,
        })
    (out_dir / "conversations.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return out_dir


# ---------------------------------------------------------------------------
# Claude export fixtures.
# ---------------------------------------------------------------------------

def make_claude_export(tmp_path: Path, conversations: list[dict]) -> Path:
    """Synthetic Claude conversations.json export."""
    out_dir = tmp_path / "claude_export"
    out_dir.mkdir()
    payload = []
    for i, conv in enumerate(conversations):
        ts0 = conv.get("created_at", "2026-04-15T12:00:00Z")
        msgs = []
        for j, (sender, text, ts) in enumerate(conv["messages"]):
            msgs.append({
                "uuid": f"claude-msg-{i}-{j}",
                "sender": sender,
                "text": text,
                "content": [{"type": "text", "text": text}],
                "created_at": ts,
                "updated_at": ts,
            })
        payload.append({
            "uuid": conv.get("uuid", f"claude-conv-{i}"),
            "name": conv["name"],
            "created_at": ts0,
            "updated_at": conv.get("updated_at", ts0),
            "account": {"uuid": "user-uuid"},
            "chat_messages": msgs,
        })
    (out_dir / "conversations.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return out_dir


@pytest.fixture
def chatgpt_export_factory(tmp_path):
    def _make(convs):
        return make_chatgpt_export(tmp_path, convs)
    return _make


@pytest.fixture
def claude_export_factory(tmp_path):
    def _make(convs):
        return make_claude_export(tmp_path, convs)
    return _make
