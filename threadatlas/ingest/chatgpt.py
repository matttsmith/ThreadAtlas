"""ChatGPT export parser.

Handles the standard ChatGPT data export, which provides a
``conversations.json`` file with conversations whose messages form a tree
(``mapping``). We linearize each tree by walking from the root via
``current_node`` if present, otherwise by depth-first walking children and
preferring the most recent branch.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator

from .base import ParsedConversation, ParsedMessage, Parser, registry
from ._common import parse_timestamp, read_json_input


CONVERSATIONS_FILE = "conversations.json"


def _looks_like_chatgpt(payload) -> bool:
    if not isinstance(payload, list) or not payload:
        return False
    sample = payload[0]
    if not isinstance(sample, dict):
        return False
    return "mapping" in sample and (
        "title" in sample or "create_time" in sample
    )


def _flatten_content(content) -> str:
    """Coerce ChatGPT's varied ``content`` shapes to plain text.

    Common shapes:
    * ``{"content_type": "text", "parts": ["...", "..."]}``
    * ``{"content_type": "code", "text": "..."}``
    * ``{"content_type": "multimodal_text", "parts": [{"text": "..."}, {"asset_pointer": ...}]}``
    * ``{"content_type": "tether_quote", "title": "...", "text": "..."}``
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        ct = content.get("content_type")
        if "parts" in content:
            parts_out: list[str] = []
            for part in content.get("parts") or []:
                if isinstance(part, str):
                    parts_out.append(part)
                elif isinstance(part, dict):
                    if "text" in part and isinstance(part["text"], str):
                        parts_out.append(part["text"])
                    elif part.get("asset_pointer"):
                        parts_out.append(f"[asset:{part['asset_pointer']}]")
            return "\n\n".join(s for s in parts_out if s)
        if ct == "code" and "language" in content:
            return f"```{content.get('language','')}\n{content.get('text','')}\n```"
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        # Fall through: return JSON-ish but readable
        return ""
    return ""


def _linearize(mapping: dict, current_node: str | None) -> list[dict]:
    """Walk the message tree into an ordered list of message nodes.

    Strategy:
      1. If ``current_node`` is set, traverse parents back to the root and
         then return the chain in chronological order. This is the path the
         user actually saw.
      2. Otherwise, DFS from a root, taking the *last* (newest) child at
         each branch. This is the most-recent-branch fallback.
    """
    if not mapping:
        return []

    if current_node and current_node in mapping:
        chain: list[str] = []
        node = current_node
        guard = 0
        while node and guard < 10000:
            chain.append(node)
            parent = mapping.get(node, {}).get("parent")
            node = parent
            guard += 1
        chain.reverse()
        return [mapping[n] for n in chain if n in mapping]

    # Find root(s): nodes whose parent is None or missing from mapping.
    roots = [nid for nid, n in mapping.items() if not n.get("parent")]
    if not roots:
        # Fallback: pick the first
        roots = [next(iter(mapping))]
    chain_ids: list[str] = []
    visited: set[str] = set()
    stack = [roots[0]]
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        if nid not in mapping:
            continue
        chain_ids.append(nid)
        children = mapping[nid].get("children") or []
        if children:
            # Push the last child so it is visited next (newest branch).
            stack.append(children[-1])
    return [mapping[n] for n in chain_ids]


class ChatGPTParser(Parser):
    name = "chatgpt"

    def can_handle(self, path: Path) -> bool:
        try:
            payload, _ = read_json_input(path, CONVERSATIONS_FILE)
        except (FileNotFoundError, zipfile.BadZipFile, ValueError):
            return False
        return _looks_like_chatgpt(payload)

    def iter_conversations(self, path: Path) -> Iterator[ParsedConversation]:
        payload, _ = read_json_input(path, CONVERSATIONS_FILE)
        if not isinstance(payload, list):
            raise ValueError("ChatGPT conversations.json should be a list at the top level.")
        for raw in payload:
            yield self._parse_one(raw)

    def _parse_one(self, raw: dict) -> ParsedConversation:
        title = (raw.get("title") or "Untitled").strip() or "Untitled"
        mapping = raw.get("mapping") or {}
        nodes = _linearize(mapping, raw.get("current_node"))

        messages: list[ParsedMessage] = []
        ordinal = 0
        for node in nodes:
            msg = node.get("message")
            if not msg:
                continue
            author = (msg.get("author") or {})
            role = (author.get("role") or "other").lower()
            if role not in {"user", "assistant", "system", "tool"}:
                role = "other"
            text = _flatten_content(msg.get("content"))
            # Skip empty system/tool noise unless they carry useful text.
            if not text.strip() and role in {"system", "tool"}:
                continue
            messages.append(
                ParsedMessage(
                    ordinal=ordinal,
                    role=role,
                    content_text=text,
                    timestamp=parse_timestamp(msg.get("create_time")),
                    source_message_id=msg.get("id"),
                )
            )
            ordinal += 1

        return ParsedConversation(
            source="chatgpt",
            source_conversation_id=str(raw.get("conversation_id") or raw.get("id") or ""),
            title=title,
            created_at=parse_timestamp(raw.get("create_time")),
            updated_at=parse_timestamp(raw.get("update_time")),
            messages=messages,
            extra={"current_node": raw.get("current_node")},
        )


registry.register(ChatGPTParser())
