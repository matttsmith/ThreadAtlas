"""Shared types and helpers used by all LLM backends.

This module is imported by both the subprocess and llama_server backends.
It must not import any networking modules.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from ..core.vault import Vault


@dataclass
class LLMResponse:
    """Result of a single LLM call."""
    raw: str
    prompt_chars: int
    response_chars: int
    duration_s: float
    success: bool
    error: str | None = None


def append_log(vault: Vault, entry: dict[str, Any]) -> None:
    """Append a metadata-only log line to <vault>/logs/llm_calls.jsonl."""
    try:
        vault.logs.mkdir(parents=True, exist_ok=True)
        with (vault.logs / "llm_calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - logging should never break callers
        pass


def truncate_prompt(prompt: str, max_chars: int) -> str:
    """Truncate a prompt to *max_chars* on a clean boundary when possible."""
    if len(prompt) <= max_chars:
        return prompt
    cut = prompt.rfind("\n", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return prompt[:cut] + "\n\n[TRUNCATED]"


# --- response parsing -------------------------------------------------------

_JSON_BLOB_RX = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(resp: LLMResponse) -> dict | None:
    """Extract and parse the first JSON object from an LLM response.

    Returns ``None`` if the response is unsuccessful, empty, or not
    parseable.  Callers should always handle ``None``.
    """
    if not resp.success or not resp.raw:
        return None
    match = _JSON_BLOB_RX.search(resp.raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
