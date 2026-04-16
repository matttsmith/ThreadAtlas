"""Backend-dispatched LLM runner.

Preserves the original ``LLMRunner.run(task, prompt, ...)`` interface but
internally dispatches to either the subprocess backend or the llama_server
backend based on ``LLMConfig.provider``.

Hard safety invariants (inherited from v1):

* Per-call timeout enforced across all backends.
* Prompt size capped before invocation; responses truncated after.
* Strict JSON parsing available via :func:`parse_json_response` — a
  malformed response never crashes the caller.
* Every call logged to ``<vault>/logs/llm_calls.jsonl`` with
  non-content metadata (task, sizes, duration, success).

The runner is NOT aware of conversation state.  Callers (summarize,
label_groups, chunking) are responsible for filtering to eligible states
(indexed + private only).
"""

from __future__ import annotations

import time

from ..core.vault import Vault
from .cache import cache_get, cache_put
from .common import LLMResponse, append_log, truncate_prompt, parse_json_response
from .config import LLMConfig
from .errors import LLMError, LLMNotConfiguredError


class LLMRunner:
    """Invoke the configured local LLM.

    Construct with a vault + config.  Call :meth:`run` with the task name
    (must be in the config's ``used_for`` whitelist) and the prompt string.

    Responses are cached in ``~/.threadatlas/llm_cache.json`` keyed by
    SHA-256(prompt).  Cache survives vault rebuilds so re-indexing after
    code changes doesn't re-run identical LLM queries.

    Pass ``use_cache=False`` to disable caching (useful in tests where
    the same prompt needs to produce different backend responses).
    """

    def __init__(self, vault: Vault, config: LLMConfig, *, use_cache: bool = True):
        self.vault = vault
        self.config = config
        self.use_cache = use_cache

    # ---- guards --------------------------------------------------------

    def _ensure_enabled(self, task: str) -> None:
        if not self.config.is_enabled_for(task):
            raise LLMNotConfiguredError(
                f"LLM task {task!r} not enabled in local_llm.json 'used_for'. "
                f"Currently enabled: {sorted(self.config.used_for)}"
            )

    # ---- main entry point ----------------------------------------------

    def run(self, task: str, prompt: str, *, conversation_ids: list[str] | None = None) -> LLMResponse:
        """Run one LLM call.  Returns an :class:`LLMResponse`.

        Checks the persistent prompt cache first.  On a cache hit, returns
        immediately without touching the LLM backend.

        Never raises on expected failures (timeout, non-zero exit, malformed
        utf-8); returns ``success=False`` with ``error`` set.
        """
        self._ensure_enabled(task)
        prompt = truncate_prompt(prompt, self.config.max_prompt_chars)

        if self.config.dry_run:
            resp = LLMResponse(
                raw=prompt,
                prompt_chars=len(prompt),
                response_chars=len(prompt),
                duration_s=0.0,
                success=True,
            )
            append_log(self.vault, {
                "ts": time.time(), "task": task, "mode": "dry_run",
                "provider": self.config.provider,
                "prompt_chars": len(prompt), "response_chars": len(prompt),
                "duration_s": 0.0, "success": True,
                "conversation_ids": conversation_ids or [],
            })
            return resp

        # --- Cache check ---
        cached_raw = cache_get(prompt) if self.use_cache else None
        if cached_raw is not None:
            resp = LLMResponse(
                raw=cached_raw,
                prompt_chars=len(prompt),
                response_chars=len(cached_raw),
                duration_s=0.0,
                success=True,
            )
            append_log(self.vault, {
                "ts": time.time(), "task": task, "mode": "cache_hit",
                "provider": self.config.provider,
                "prompt_chars": len(prompt), "response_chars": len(cached_raw),
                "duration_s": 0.0, "success": True,
                "conversation_ids": conversation_ids or [],
            })
            return resp

        # --- Backend call ---
        if self.config.provider == "llama_server":
            from .llama_server_backend import run_llama_server
            resp = run_llama_server(
                self.vault, self.config, task, prompt,
                conversation_ids=conversation_ids,
            )
        else:
            from .subprocess_backend import run_subprocess
            resp = run_subprocess(
                self.vault, self.config, task, prompt,
                conversation_ids=conversation_ids,
            )

        # --- Cache store (only successful responses) ---
        if self.use_cache and resp.success and resp.raw:
            cache_put(prompt, resp.raw, task=task)

        return resp


# Re-export for backward compatibility with callers that import from runner.
__all__ = ["LLMRunner", "LLMResponse", "LLMError", "parse_json_response"]
