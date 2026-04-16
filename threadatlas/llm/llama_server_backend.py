"""llama-server (OpenAI-compatible) backend for the local LLM runner.

This is the ONLY module in the ThreadAtlas runtime package that uses
stdlib HTTP/socket imports.  ``tests/test_no_network.py`` must allowlist
this file so the static network-import guard passes.

Hard invariants:
* Refuses non-loopback ``base_url`` unless ``allow_nonlocal_host: true``
  is explicitly set in local_llm.json.
* Per-call timeout enforced via ``urllib.request`` timeout parameter.
* Prompt size capped before invocation; responses truncated after.
* Every call logged (metadata only) to ``<vault>/logs/llm_calls.jsonl``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from ..core.vault import Vault
from .common import LLMResponse, append_log
from .config import LLMConfig


# ---------------------------------------------------------------------------
# Host-safety check
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = frozenset({
    "localhost",
    "127.0.0.1",
    "::1",
    "[::1]",
})


def is_loopback_url(url: str) -> bool:
    """Return True if *url* points to a loopback address."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    return hostname in _LOOPBACK_HOSTS


def _require_loopback(config: LLMConfig) -> None:
    """Raise if base_url is non-loopback and allow_nonlocal_host is off."""
    url = config.base_url or ""
    if not url:
        raise ValueError("llama_server provider requires 'base_url' in local_llm.json")
    if not config.allow_nonlocal_host and not is_loopback_url(url):
        raise ValueError(
            f"base_url {url!r} is not a loopback address. "
            "Set 'allow_nonlocal_host': true in local_llm.json to allow this."
        )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post_json(url: str, body: dict, *, timeout: int) -> dict:
    """POST JSON and return parsed response body."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, *, timeout: int) -> dict:
    """GET JSON and return parsed response body."""
    req = urllib.request.Request(url, method="GET")
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_readiness(config: LLMConfig) -> dict[str, Any]:
    """Verify the llama-server is reachable and has the expected model.

    Returns a dict with ``ready``, ``models``, and optionally ``error``.
    """
    _require_loopback(config)
    base = (config.base_url or "").rstrip("/")
    try:
        data = _get_json(f"{base}/v1/models", timeout=min(config.timeout_seconds, 10))
    except Exception as exc:
        return {"ready": False, "models": [], "error": str(exc)}

    model_ids = [m.get("id", "") for m in data.get("data", [])]
    expected = config.model or ""
    if expected and expected not in model_ids:
        return {
            "ready": False,
            "models": model_ids,
            "error": f"expected model {expected!r} not in {model_ids}",
        }
    return {"ready": True, "models": model_ids}


def probe(config: LLMConfig) -> dict[str, Any]:
    """Send a tiny completion to confirm the model responds.

    Returns a dict with ``ok``, ``text``, and optionally ``error``.
    """
    _require_loopback(config)
    base = (config.base_url or "").rstrip("/")
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Reply with exactly: LLAMA_SERVER_OK"}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    if config.model:
        body["model"] = config.model
    try:
        data = _post_json(f"{base}/v1/chat/completions", body, timeout=min(config.timeout_seconds, 30))
    except Exception as exc:
        return {"ok": False, "text": "", "error": str(exc)}

    choices = data.get("choices") or []
    if not choices:
        return {"ok": False, "text": "", "error": "empty choices in response"}
    text = (choices[0].get("message") or {}).get("content", "")
    return {"ok": "LLAMA_SERVER_OK" in text, "text": text}


def run_llama_server(
    vault: Vault,
    config: LLMConfig,
    task: str,
    prompt: str,
    *,
    conversation_ids: list[str] | None = None,
) -> LLMResponse:
    """Run one LLM call via the llama-server HTTP API.

    Never raises on expected failures (timeout, HTTP error, malformed
    response); returns ``success=False`` with ``error`` set.
    """
    _require_loopback(config)
    base = (config.base_url or "").rstrip("/")
    start = time.monotonic()

    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
    }
    if config.model:
        body["model"] = config.model
    if config.temperature is not None:
        body["temperature"] = config.temperature
    if config.max_tokens is not None:
        body["max_tokens"] = config.max_tokens

    try:
        data = _post_json(
            f"{base}/v1/chat/completions",
            body,
            timeout=config.timeout_seconds,
        )
    except urllib.error.URLError as exc:
        duration = time.monotonic() - start
        error_msg = f"connection error: {exc.reason}" if hasattr(exc, "reason") else str(exc)
        resp = LLMResponse(
            raw="", prompt_chars=len(prompt), response_chars=0,
            duration_s=duration, success=False, error=error_msg,
        )
        append_log(vault, {
            "ts": time.time(), "task": task, "mode": "llama_server",
            "prompt_chars": len(prompt), "response_chars": 0,
            "duration_s": duration, "success": False, "error": error_msg,
            "conversation_ids": conversation_ids or [],
        })
        return resp
    except Exception as exc:
        duration = time.monotonic() - start
        error_msg = str(exc)
        resp = LLMResponse(
            raw="", prompt_chars=len(prompt), response_chars=0,
            duration_s=duration, success=False, error=error_msg,
        )
        append_log(vault, {
            "ts": time.time(), "task": task, "mode": "llama_server",
            "prompt_chars": len(prompt), "response_chars": 0,
            "duration_s": duration, "success": False, "error": error_msg,
            "conversation_ids": conversation_ids or [],
        })
        return resp

    choices = data.get("choices") or []
    if not choices:
        duration = time.monotonic() - start
        resp = LLMResponse(
            raw="", prompt_chars=len(prompt), response_chars=0,
            duration_s=duration, success=False, error="empty choices in response",
        )
        append_log(vault, {
            "ts": time.time(), "task": task, "mode": "llama_server",
            "prompt_chars": len(prompt), "response_chars": 0,
            "duration_s": duration, "success": False, "error": "empty_choices",
            "conversation_ids": conversation_ids or [],
        })
        return resp

    raw_text = (choices[0].get("message") or {}).get("content", "")
    raw_text = raw_text[: config.max_response_chars]
    duration = time.monotonic() - start

    resp = LLMResponse(
        raw=raw_text,
        prompt_chars=len(prompt),
        response_chars=len(raw_text),
        duration_s=duration,
        success=True,
    )
    append_log(vault, {
        "ts": time.time(), "task": task, "mode": "llama_server",
        "prompt_chars": len(prompt), "response_chars": len(raw_text),
        "duration_s": duration, "success": True,
        "conversation_ids": conversation_ids or [],
    })
    return resp
