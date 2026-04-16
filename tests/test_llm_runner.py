"""LLMRunner subprocess behavior: timeout, failure, dry-run, log file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from threadatlas.llm import LLMRunner, load_config
from threadatlas.llm.config import LLMConfig
from threadatlas.llm.runner import parse_json_response


FAKE_CMD = [sys.executable, "-m", "tests.fake_llm"]


def _write_config(vault, *, mode: str, timeout: int = 10, used_for=("summaries", "group_naming", "chunk_gating"), dry_run=False):
    (vault.root / "local_llm.json").write_text(json.dumps({
        "command": FAKE_CMD + [mode],
        "timeout_seconds": timeout,
        "max_prompt_chars": 5000,
        "max_response_chars": 2000,
        "used_for": list(used_for),
        "dry_run": dry_run,
    }), encoding="utf-8")


def test_dry_run_does_not_invoke_subprocess(tmp_vault):
    _write_config(tmp_vault, mode="nonzero", dry_run=True)
    cfg = load_config(tmp_vault.root)
    runner = LLMRunner(tmp_vault, cfg, use_cache=False)
    resp = runner.run("summaries", "hello")
    # Dry-run treats the prompt as the response, never invokes the subprocess,
    # so even a "nonzero" mode would succeed. That's the invariant we want.
    assert resp.success
    assert "hello" in resp.raw


def test_success_path(tmp_vault):
    _write_config(tmp_vault, mode="summary_ok")
    cfg = load_config(tmp_vault.root)
    runner = LLMRunner(tmp_vault, cfg, use_cache=False)
    resp = runner.run("summaries", "make a summary")
    assert resp.success
    parsed = parse_json_response(resp)
    assert parsed == {"summary": "Short topical summary."}


def test_nonzero_exit_is_not_success(tmp_vault):
    _write_config(tmp_vault, mode="nonzero")
    cfg = load_config(tmp_vault.root)
    runner = LLMRunner(tmp_vault, cfg, use_cache=False)
    resp = runner.run("summaries", "any")
    assert resp.success is False
    assert resp.error and "exit=1" in resp.error


def test_timeout_caps_subprocess(tmp_vault):
    _write_config(tmp_vault, mode="hang", timeout=1)
    cfg = load_config(tmp_vault.root)
    runner = LLMRunner(tmp_vault, cfg, use_cache=False)
    resp = runner.run("summaries", "any")
    assert resp.success is False
    assert resp.error == "timeout"


def test_refuses_unauthorized_task(tmp_vault):
    _write_config(tmp_vault, mode="summary_ok", used_for=("summaries",))
    cfg = load_config(tmp_vault.root)
    runner = LLMRunner(tmp_vault, cfg, use_cache=False)
    with pytest.raises(Exception):  # LLMError
        runner.run("chunk_gating", "any")


def test_log_file_written(tmp_vault):
    _write_config(tmp_vault, mode="summary_ok")
    cfg = load_config(tmp_vault.root)
    runner = LLMRunner(tmp_vault, cfg, use_cache=False)
    runner.run("summaries", "any", conversation_ids=["conv_abc"])
    log_path = tmp_vault.logs / "llm_calls.jsonl"
    assert log_path.exists()
    entries = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln]
    assert len(entries) == 1
    e = entries[0]
    assert e["task"] == "summaries"
    assert e["success"] is True
    assert e["conversation_ids"] == ["conv_abc"]
    # Logs must NOT contain prompt or response content.
    assert "prompt" not in e
    assert "response" not in e


def test_parse_json_response_recovers_from_preamble():
    """Small models often emit 'Sure! Here is the JSON: {...}'. We should cope."""
    from threadatlas.llm.runner import LLMResponse
    resp = LLMResponse(
        raw='Sure! Here is the JSON you asked for:\n{"summary": "hello"}\nHope that helps!',
        prompt_chars=0, response_chars=0, duration_s=0.0, success=True,
    )
    assert parse_json_response(resp) == {"summary": "hello"}
