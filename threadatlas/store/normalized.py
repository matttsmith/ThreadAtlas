"""Read/write the canonical normalized JSON file for a conversation.

The normalized file is the human-inspectable, recoverable source of truth for
a single conversation. The DB indexes over these.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..core.models import Conversation, Message
from ..core.vault import Vault


def write_normalized(vault: Vault, conv: Conversation, messages: list[Message]) -> Path:
    path = vault.normalized_path_for(conv.conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": conv.schema_version,
        "parser_version": conv.parser_version,
        "conversation": asdict(conv),
        "messages": [
            {
                "message_id": m.message_id,
                "ordinal": m.ordinal,
                "role": m.role,
                "timestamp": m.timestamp,
                "content_text": m.content_text,
                "content_structured": m.content_structured,
                "source_message_id": m.source_message_id,
            }
            for m in messages
        ],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_normalized(vault: Vault, conversation_id: str) -> dict | None:
    path = vault.normalized_path_for(conversation_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_normalized(vault: Vault, conversation_id: str) -> bool:
    path = vault.normalized_path_for(conversation_id)
    if not path.exists():
        return False
    path.unlink()
    # Try to clean empty shard dirs.
    try:
        path.parent.rmdir()
    except OSError:
        pass
    return True
