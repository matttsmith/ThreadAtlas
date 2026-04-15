"""Claude (anthropic.com) export parser.

Handles the standard Claude data export, which contains a
``conversations.json`` file with a list of conversation objects. Each
conversation has ``chat_messages``, an ordered list with ``sender``,
``text``, ``content``, and timestamp fields.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator

from .base import ParsedConversation, ParsedMessage, Parser, registry
from ._common import parse_timestamp, read_json_input


CONVERSATIONS_FILE = "conversations.json"


def _looks_like_claude(payload) -> bool:
    if not isinstance(payload, list) or not payload:
        return False
    sample = payload[0]
    if not isinstance(sample, dict):
        return False
    # Distinct shape: conversations have ``chat_messages`` and ``uuid``.
    return "chat_messages" in sample and "uuid" in sample


def _flatten_claude_content(msg: dict) -> str:
    """Coerce Claude message content to plain text.

    Newer exports give a structured ``content`` array of blocks; older give
    a top-level ``text`` field. Prefer structured if present.
    """
    parts: list[str] = []
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif btype == "tool_use":
                name = block.get("name", "tool")
                parts.append(f"[tool_use:{name}]")
            elif btype == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    parts.append(f"[tool_result] {inner}")
                elif isinstance(inner, list):
                    for ib in inner:
                        if isinstance(ib, dict) and isinstance(ib.get("text"), str):
                            parts.append(f"[tool_result] {ib['text']}")
            elif "text" in block and isinstance(block["text"], str):
                parts.append(block["text"])
    if not parts and isinstance(msg.get("text"), str):
        parts.append(msg["text"])

    # Attachments and files are mentioned briefly so search/inspection can see them.
    for att in msg.get("attachments") or []:
        if isinstance(att, dict):
            name = att.get("file_name") or att.get("name") or "attachment"
            parts.append(f"[attachment:{name}]")
    for f in msg.get("files") or []:
        if isinstance(f, dict):
            name = f.get("file_name") or f.get("name") or "file"
            parts.append(f"[file:{name}]")
    return "\n\n".join(p for p in parts if p)


def _normalize_role(sender: str | None) -> str:
    if not sender:
        return "other"
    s = sender.lower()
    if s in {"human", "user"}:
        return "user"
    if s in {"assistant", "claude"}:
        return "assistant"
    if s == "system":
        return "system"
    return "other"


class ClaudeParser(Parser):
    name = "claude"

    def can_handle(self, path: Path) -> bool:
        try:
            payload, _ = read_json_input(path, CONVERSATIONS_FILE)
        except (FileNotFoundError, zipfile.BadZipFile, ValueError):
            return False
        return _looks_like_claude(payload)

    def iter_conversations(self, path: Path) -> Iterator[ParsedConversation]:
        payload, _ = read_json_input(path, CONVERSATIONS_FILE)
        if not isinstance(payload, list):
            raise ValueError("Claude conversations.json should be a list at the top level.")
        for raw in payload:
            yield self._parse_one(raw)

    def _parse_one(self, raw: dict) -> ParsedConversation:
        title = (raw.get("name") or "Untitled").strip() or "Untitled"
        chat_messages = raw.get("chat_messages") or []
        # Sort defensively by created_at if present, else preserve order.
        def _key(m):
            ts = parse_timestamp(m.get("created_at"))
            return ts if ts is not None else 0.0
        try:
            chat_messages = sorted(chat_messages, key=_key)
        except Exception:
            pass

        messages: list[ParsedMessage] = []
        for i, msg in enumerate(chat_messages):
            text = _flatten_claude_content(msg)
            if not text.strip():
                # Allow tool/system without text to still record a marker; otherwise skip.
                continue
            messages.append(
                ParsedMessage(
                    ordinal=i,
                    role=_normalize_role(msg.get("sender")),
                    content_text=text,
                    timestamp=parse_timestamp(msg.get("created_at")),
                    source_message_id=msg.get("uuid"),
                )
            )

        return ParsedConversation(
            source="claude",
            source_conversation_id=str(raw.get("uuid") or ""),
            title=title,
            created_at=parse_timestamp(raw.get("created_at")),
            updated_at=parse_timestamp(raw.get("updated_at")),
            messages=messages,
            extra={"account": raw.get("account")},
        )


registry.register(ClaudeParser())
