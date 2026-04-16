"""Local LLM configuration.

Single JSON file at ``<vault>/local_llm.json``.

Subprocess example (original, backward-compatible):

.. code-block:: json

    {
      "command": ["/usr/local/bin/llama-cli",
                  "-m", "/path/to/model.gguf",
                  "--prompt-file", "{PROMPT_FILE}",
                  "--no-conversation",
                  "--temp", "0.1",
                  "--n-predict", "256"],
      "timeout_seconds": 120,
      "max_prompt_chars": 12000,
      "max_response_chars": 4000,
      "used_for": ["summaries", "group_naming", "chunk_gating"],
      "dry_run": false
    }

llama-server example (OpenAI-compatible local server):

.. code-block:: json

    {
      "provider": "llama_server",
      "base_url": "http://127.0.0.1:8080",
      "model": "qwen2.5-3b-instruct",
      "temperature": 0.1,
      "max_tokens": 256,
      "timeout_seconds": 120,
      "max_prompt_chars": 12000,
      "max_response_chars": 4000,
      "used_for": ["summaries", "group_naming", "chunk_gating"],
      "dry_run": false
    }

Substitutions (subprocess only):
  ``{PROMPT_FILE}`` - absolute path of a temp file containing the prompt.
  ``{PROMPT}``      - inline substitution of the prompt (careful: shells).

If neither substitution appears in ``command``, the prompt is fed via
stdin.  Stdin is the default and the most reliable option.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


CONFIG_BASENAME = "local_llm.json"

# Tasks the config may opt into.
ALLOWED_USED_FOR = frozenset({
    "summaries",
    "group_naming",
    "chunk_gating",
    # v2 pipeline tasks
    "turn_classification",
    "extraction",
    "profile",
})

# Supported provider backends.
PROVIDERS = frozenset({"subprocess", "llama_server"})


@dataclass
class LLMConfig:
    """Parsed local_llm.json configuration."""

    # --- common fields (both providers) ---
    provider: str = "subprocess"
    timeout_seconds: int = 120
    max_prompt_chars: int = 12_000
    max_response_chars: int = 4_000
    used_for: frozenset[str] = field(default_factory=frozenset)
    dry_run: bool = False

    # --- subprocess fields ---
    command: list[str] = field(default_factory=list)

    # --- llama_server fields ---
    base_url: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    healthcheck_on_start: bool = True
    allow_nonlocal_host: bool = False

    def is_enabled_for(self, task: str) -> bool:
        return task in self.used_for


def config_path(vault_root: Path) -> Path:
    return Path(vault_root) / CONFIG_BASENAME


def load_config(vault_root: Path) -> LLMConfig | None:
    """Load local_llm.json.  Return ``None`` if the file is absent."""
    p = config_path(vault_root)
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))

    # --- used_for validation ---
    used_for = frozenset(raw.get("used_for") or [])
    unknown = used_for - ALLOWED_USED_FOR
    if unknown:
        raise ValueError(
            f"local_llm.json: unknown 'used_for' entries: {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_USED_FOR)}"
        )

    # --- provider detection (backward-compatible) ---
    provider = raw.get("provider", "subprocess")
    if provider not in PROVIDERS:
        raise ValueError(
            f"local_llm.json: unknown provider {provider!r}. "
            f"Allowed: {sorted(PROVIDERS)}"
        )

    # --- provider-specific validation ---
    if provider == "subprocess":
        command = raw.get("command") or []
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError("local_llm.json: 'command' must be a non-empty list of strings")
    else:
        command = []

    base_url = raw.get("base_url")
    if provider == "llama_server" and not base_url:
        raise ValueError("local_llm.json: llama_server provider requires 'base_url'")

    return LLMConfig(
        provider=provider,
        command=command,
        timeout_seconds=int(raw.get("timeout_seconds", 120)),
        max_prompt_chars=int(raw.get("max_prompt_chars", 12_000)),
        max_response_chars=int(raw.get("max_response_chars", 4_000)),
        used_for=used_for,
        dry_run=bool(raw.get("dry_run", False)),
        base_url=base_url,
        model=raw.get("model"),
        temperature=raw.get("temperature"),
        max_tokens=raw.get("max_tokens"),
        healthcheck_on_start=bool(raw.get("healthcheck_on_start", True)),
        allow_nonlocal_host=bool(raw.get("allow_nonlocal_host", False)),
    )
