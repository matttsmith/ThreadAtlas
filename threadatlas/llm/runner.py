"""Subprocess runner for the user's local LLM.

Hard safety invariants:

* Never opens a network socket. Only ``subprocess.run`` is used.
* Per-call timeout enforced; a hanging model does not block indefinitely.
* Prompt size capped before invocation; responses truncated after.
* Strict JSON parsing available via :func:`parse_json_response` - a
  malformed response never crashes the caller.
* Every call logged to ``<vault>/logs/llm_calls.jsonl`` with
  non-content metadata (task, sizes, duration, success).

The runner is NOT aware of conversation state. Callers (summarize,
label_groups, chunking) are responsible for filtering to eligible states
(indexed + private only).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.vault import Vault
from .config import LLMConfig


class LLMError(RuntimeError):
    """Raised when the LLM subprocess itself fails (timeout, non-zero exit)."""


@dataclass
class LLMResponse:
    raw: str
    prompt_chars: int
    response_chars: int
    duration_s: float
    success: bool
    error: str | None = None


def _append_log(vault: Vault, entry: dict[str, Any]) -> None:
    try:
        vault.logs.mkdir(parents=True, exist_ok=True)
        with (vault.logs / "llm_calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - logging should never break callers
        pass


class LLMRunner:
    """Invoke the configured local LLM.

    Construct with a vault + config. Call :meth:`run` with the task name
    (must be in the config's ``used_for`` whitelist) and the prompt string.
    """

    def __init__(self, vault: Vault, config: LLMConfig):
        self.vault = vault
        self.config = config

    # ---- guards --------------------------------------------------------

    def _ensure_enabled(self, task: str) -> None:
        if not self.config.is_enabled_for(task):
            raise LLMError(
                f"LLM task {task!r} not enabled in local_llm.json 'used_for'. "
                f"Currently enabled: {sorted(self.config.used_for)}"
            )

    def _truncate_prompt(self, prompt: str) -> str:
        mx = self.config.max_prompt_chars
        if len(prompt) <= mx:
            return prompt
        # Truncate on a clean boundary when possible.
        cut = prompt.rfind("\n", 0, mx)
        if cut < mx // 2:
            cut = mx
        return prompt[:cut] + "\n\n[TRUNCATED]"

    # ---- main entry point ----------------------------------------------

    def run(self, task: str, prompt: str, *, conversation_ids: list[str] | None = None) -> LLMResponse:
        """Run one LLM call. Returns an :class:`LLMResponse`.

        Never raises on expected failures (timeout, non-zero exit, malformed
        utf-8); returns ``success=False`` with ``error`` set.
        """
        self._ensure_enabled(task)
        prompt = self._truncate_prompt(prompt)
        start = time.monotonic()

        if self.config.dry_run:
            # We still "invoke" the contract: truncate, log, return the prompt
            # as the response so callers can see what would have been sent.
            resp = LLMResponse(
                raw=prompt,
                prompt_chars=len(prompt),
                response_chars=len(prompt),
                duration_s=0.0,
                success=True,
            )
            _append_log(self.vault, {
                "ts": time.time(), "task": task, "mode": "dry_run",
                "prompt_chars": len(prompt), "response_chars": len(prompt),
                "duration_s": 0.0, "success": True,
                "conversation_ids": conversation_ids or [],
            })
            return resp

        # Build argv, substituting {PROMPT} / {PROMPT_FILE} if present.
        argv = list(self.config.command)
        use_stdin = not any("{PROMPT}" in a or "{PROMPT_FILE}" in a for a in argv)
        prompt_file: Path | None = None
        try:
            if not use_stdin:
                # Write prompt to a temp file the caller may reference.
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".prompt", delete=False, encoding="utf-8"
                )
                tmp.write(prompt)
                tmp.flush()
                tmp.close()
                prompt_file = Path(tmp.name)
                argv = [
                    a.replace("{PROMPT_FILE}", str(prompt_file))
                     .replace("{PROMPT}", prompt)
                    for a in argv
                ]
            try:
                proc = subprocess.run(
                    argv,
                    input=prompt if use_stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                duration = time.monotonic() - start
                resp = LLMResponse(
                    raw="", prompt_chars=len(prompt), response_chars=0,
                    duration_s=duration, success=False, error="timeout",
                )
                _append_log(self.vault, {
                    "ts": time.time(), "task": task, "mode": "subprocess",
                    "prompt_chars": len(prompt), "response_chars": 0,
                    "duration_s": duration, "success": False, "error": "timeout",
                    "conversation_ids": conversation_ids or [],
                })
                return resp
            except FileNotFoundError as e:
                duration = time.monotonic() - start
                resp = LLMResponse(
                    raw="", prompt_chars=len(prompt), response_chars=0,
                    duration_s=duration, success=False, error=f"command not found: {e}",
                )
                _append_log(self.vault, {
                    "ts": time.time(), "task": task, "mode": "subprocess",
                    "prompt_chars": len(prompt), "response_chars": 0,
                    "duration_s": duration, "success": False, "error": "not_found",
                    "conversation_ids": conversation_ids or [],
                })
                return resp

            raw = (proc.stdout or "")[: self.config.max_response_chars]
            duration = time.monotonic() - start
            ok = proc.returncode == 0
            resp = LLMResponse(
                raw=raw,
                prompt_chars=len(prompt),
                response_chars=len(raw),
                duration_s=duration,
                success=ok,
                error=None if ok else f"exit={proc.returncode}: {(proc.stderr or '')[:500]}",
            )
            _append_log(self.vault, {
                "ts": time.time(), "task": task, "mode": "subprocess",
                "prompt_chars": len(prompt), "response_chars": len(raw),
                "duration_s": duration, "success": ok,
                "returncode": proc.returncode,
                "conversation_ids": conversation_ids or [],
            })
            return resp
        finally:
            if prompt_file is not None:
                try:
                    prompt_file.unlink(missing_ok=True)
                except Exception:
                    pass


# --- response parsing ------------------------------------------------------

_JSON_BLOB_RX = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(resp: LLMResponse) -> dict | None:
    """Extract and parse the first JSON object from an LLM response.

    Returns ``None`` if the response is unsuccessful, empty, or not
    parseable. Callers should always handle ``None``.
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
