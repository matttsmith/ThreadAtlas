"""Tests for the llama_server LLM backend.

All tests mock HTTP calls — no real server required.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock
from http.client import HTTPResponse
from io import BytesIO

import pytest

from threadatlas.llm.config import LLMConfig
from threadatlas.llm.llama_server_backend import (
    check_readiness,
    is_loopback_url,
    probe,
    run_llama_server,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llama_config(**overrides) -> LLMConfig:
    defaults = dict(
        provider="llama_server",
        base_url="http://127.0.0.1:8080",
        model="test-model",
        temperature=0.1,
        max_tokens=256,
        timeout_seconds=10,
        max_prompt_chars=2000,
        max_response_chars=1000,
        used_for=frozenset({"summaries"}),
        dry_run=False,
        allow_nonlocal_host=False,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


def _mock_urlopen(response_body: dict, status=200):
    """Create a mock that makes urllib.request.urlopen return *response_body*."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_body).encode("utf-8")
    mock_resp.status = status
    return patch("threadatlas.llm.llama_server_backend.urllib.request.urlopen",
                 return_value=mock_resp)


# ---------------------------------------------------------------------------
# Loopback enforcement
# ---------------------------------------------------------------------------

def test_is_loopback_localhost():
    assert is_loopback_url("http://localhost:8080") is True


def test_is_loopback_127():
    assert is_loopback_url("http://127.0.0.1:8080") is True


def test_is_loopback_ipv6():
    assert is_loopback_url("http://[::1]:8080") is True


def test_is_not_loopback_remote():
    assert is_loopback_url("http://192.168.1.100:8080") is False


def test_is_not_loopback_public():
    assert is_loopback_url("https://api.example.com") is False


def test_run_rejects_nonlocal_host(tmp_vault):
    cfg = _llama_config(base_url="http://10.0.0.5:8080")
    with pytest.raises(ValueError, match="not a loopback"):
        run_llama_server(tmp_vault, cfg, "summaries", "hello")


def test_run_allows_nonlocal_when_explicitly_set(tmp_vault):
    cfg = _llama_config(base_url="http://10.0.0.5:8080", allow_nonlocal_host=True)
    body = {"choices": [{"message": {"content": "ok"}}]}
    with _mock_urlopen(body):
        resp = run_llama_server(tmp_vault, cfg, "summaries", "hello")
    assert resp.success


# ---------------------------------------------------------------------------
# check_readiness
# ---------------------------------------------------------------------------

def test_check_readiness_success():
    cfg = _llama_config()
    body = {"data": [{"id": "test-model"}]}
    with _mock_urlopen(body):
        result = check_readiness(cfg)
    assert result["ready"] is True
    assert "test-model" in result["models"]


def test_check_readiness_model_mismatch():
    cfg = _llama_config(model="expected-model")
    body = {"data": [{"id": "other-model"}]}
    with _mock_urlopen(body):
        result = check_readiness(cfg)
    assert result["ready"] is False
    assert "expected-model" in result.get("error", "")


def test_check_readiness_connection_error():
    cfg = _llama_config()
    import urllib.error
    with patch("threadatlas.llm.llama_server_backend.urllib.request.urlopen",
               side_effect=urllib.error.URLError("Connection refused")):
        result = check_readiness(cfg)
    assert result["ready"] is False


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def test_probe_success():
    cfg = _llama_config()
    body = {"choices": [{"message": {"content": "LLAMA_SERVER_OK"}}]}
    with _mock_urlopen(body):
        result = probe(cfg)
    assert result["ok"] is True


def test_probe_wrong_response():
    cfg = _llama_config()
    body = {"choices": [{"message": {"content": "something else"}}]}
    with _mock_urlopen(body):
        result = probe(cfg)
    assert result["ok"] is False


def test_probe_empty_choices():
    cfg = _llama_config()
    body = {"choices": []}
    with _mock_urlopen(body):
        result = probe(cfg)
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# run_llama_server
# ---------------------------------------------------------------------------

def test_run_success(tmp_vault):
    cfg = _llama_config()
    body = {"choices": [{"message": {"content": '{"summary": "test"}'}}]}
    with _mock_urlopen(body):
        resp = run_llama_server(tmp_vault, cfg, "summaries", "test prompt")
    assert resp.success is True
    assert '{"summary": "test"}' in resp.raw
    assert resp.prompt_chars == len("test prompt")


def test_run_connection_error(tmp_vault):
    cfg = _llama_config()
    import urllib.error
    with patch("threadatlas.llm.llama_server_backend.urllib.request.urlopen",
               side_effect=urllib.error.URLError("Connection refused")):
        resp = run_llama_server(tmp_vault, cfg, "summaries", "test")
    assert resp.success is False
    assert "connection error" in resp.error.lower() or "Connection refused" in resp.error


def test_run_empty_choices(tmp_vault):
    cfg = _llama_config()
    body = {"choices": []}
    with _mock_urlopen(body):
        resp = run_llama_server(tmp_vault, cfg, "summaries", "test")
    assert resp.success is False
    assert "empty choices" in resp.error


def test_run_truncates_response(tmp_vault):
    cfg = _llama_config(max_response_chars=20)
    long_content = "x" * 500
    body = {"choices": [{"message": {"content": long_content}}]}
    with _mock_urlopen(body):
        resp = run_llama_server(tmp_vault, cfg, "summaries", "test")
    assert resp.success is True
    assert len(resp.raw) <= 20


def test_run_logs_call(tmp_vault):
    cfg = _llama_config()
    body = {"choices": [{"message": {"content": "ok"}}]}
    with _mock_urlopen(body):
        run_llama_server(tmp_vault, cfg, "summaries", "test",
                         conversation_ids=["c1"])
    log_path = tmp_vault.logs / "llm_calls.jsonl"
    assert log_path.exists()
    entry = json.loads(log_path.read_text().strip().split("\n")[-1])
    assert entry["task"] == "summaries"
    assert entry["mode"] == "llama_server"
    assert entry["success"] is True


def test_run_dry_run_skips_http(tmp_vault):
    """dry_run should never make HTTP calls."""
    cfg = _llama_config(dry_run=True)
    from threadatlas.llm.runner import LLMRunner
    runner = LLMRunner(tmp_vault, cfg, use_cache=False)
    # If HTTP were called, this would fail since there's no mock.
    resp = runner.run("summaries", "test prompt")
    assert resp.success is True
    assert resp.raw == "test prompt"


# ---------------------------------------------------------------------------
# General exception handling
# ---------------------------------------------------------------------------

def test_run_generic_exception(tmp_vault):
    cfg = _llama_config()
    with patch("threadatlas.llm.llama_server_backend.urllib.request.urlopen",
               side_effect=RuntimeError("unexpected")):
        resp = run_llama_server(tmp_vault, cfg, "summaries", "test")
    assert resp.success is False
    assert "unexpected" in resp.error
