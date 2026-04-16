"""Subprocess backend for the local LLM runner.

Behavior is identical to the original monolithic runner.py: the configured
``command`` list is invoked via ``subprocess.run`` with the prompt passed
via stdin, ``{PROMPT}`` inline substitution, or ``{PROMPT_FILE}`` temp-file
substitution.

Hard invariants preserved:
* Never opens a network socket.
* Per-call timeout enforced.
* Prompt size capped before invocation; responses truncated after.
* Every call logged (metadata only) to ``<vault>/logs/llm_calls.jsonl``.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from ..core.vault import Vault
from .common import LLMResponse, append_log
from .config import LLMConfig


def run_subprocess(
    vault: Vault,
    config: LLMConfig,
    task: str,
    prompt: str,
    *,
    conversation_ids: list[str] | None = None,
) -> LLMResponse:
    """Run one LLM call via subprocess.

    Never raises on expected failures (timeout, non-zero exit, missing
    binary); returns ``success=False`` with ``error`` set.
    """
    start = time.monotonic()

    # Build argv, substituting {PROMPT} / {PROMPT_FILE} if present.
    argv = list(config.command)
    use_stdin = not any("{PROMPT}" in a or "{PROMPT_FILE}" in a for a in argv)
    prompt_file: Path | None = None
    try:
        if not use_stdin:
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
                timeout=config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            resp = LLMResponse(
                raw="", prompt_chars=len(prompt), response_chars=0,
                duration_s=duration, success=False, error="timeout",
            )
            append_log(vault, {
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
            append_log(vault, {
                "ts": time.time(), "task": task, "mode": "subprocess",
                "prompt_chars": len(prompt), "response_chars": 0,
                "duration_s": duration, "success": False, "error": "not_found",
                "conversation_ids": conversation_ids or [],
            })
            return resp

        raw = (proc.stdout or "")[: config.max_response_chars]
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
        append_log(vault, {
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
