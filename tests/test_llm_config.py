"""LLM config loader tests — both subprocess and llama_server backends."""

from __future__ import annotations

import json

import pytest

from threadatlas.llm import load_config
from threadatlas.llm.config import LLMConfig


# --- subprocess (backward-compatible) -----------------------------------------

def test_missing_config_returns_none(tmp_vault):
    assert load_config(tmp_vault.root) is None


def test_valid_subprocess_config_parses(tmp_vault):
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "command": ["/bin/echo", "{PROMPT}"],
        "timeout_seconds": 30,
        "max_prompt_chars": 5000,
        "max_response_chars": 2000,
        "used_for": ["summaries"],
        "dry_run": False,
    }), encoding="utf-8")
    cfg = load_config(tmp_vault.root)
    assert isinstance(cfg, LLMConfig)
    assert cfg.provider == "subprocess"
    assert cfg.is_enabled_for("summaries")
    assert not cfg.is_enabled_for("group_naming")


def test_subprocess_config_backward_compatible_no_provider(tmp_vault):
    """Configs without 'provider' must default to subprocess."""
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "command": ["/bin/echo"],
        "used_for": ["summaries"],
    }), encoding="utf-8")
    cfg = load_config(tmp_vault.root)
    assert cfg.provider == "subprocess"
    assert cfg.command == ["/bin/echo"]


def test_unknown_used_for_value_rejected(tmp_vault):
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "command": ["/bin/echo"],
        "used_for": ["summaries", "something_weird"],
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_vault.root)


def test_empty_command_rejected(tmp_vault):
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "command": [],
        "used_for": ["summaries"],
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_vault.root)


# --- llama_server backend ----------------------------------------------------

def test_valid_llama_server_config_parses(tmp_vault):
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "provider": "llama_server",
        "base_url": "http://127.0.0.1:8080",
        "model": "qwen2.5-3b-instruct",
        "temperature": 0.1,
        "max_tokens": 256,
        "timeout_seconds": 60,
        "used_for": ["summaries", "group_naming"],
    }), encoding="utf-8")
    cfg = load_config(tmp_vault.root)
    assert cfg.provider == "llama_server"
    assert cfg.base_url == "http://127.0.0.1:8080"
    assert cfg.model == "qwen2.5-3b-instruct"
    assert cfg.temperature == 0.1
    assert cfg.max_tokens == 256
    assert cfg.is_enabled_for("summaries")
    assert cfg.is_enabled_for("group_naming")
    assert not cfg.is_enabled_for("chunk_gating")


def test_llama_server_requires_base_url(tmp_vault):
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "provider": "llama_server",
        "used_for": ["summaries"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="base_url"):
        load_config(tmp_vault.root)


def test_unknown_provider_rejected(tmp_vault):
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "provider": "openai_cloud",
        "command": ["/bin/echo"],
        "used_for": ["summaries"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown provider"):
        load_config(tmp_vault.root)


def test_llama_server_allow_nonlocal_host_default_false(tmp_vault):
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "provider": "llama_server",
        "base_url": "http://127.0.0.1:8080",
        "used_for": ["summaries"],
    }), encoding="utf-8")
    cfg = load_config(tmp_vault.root)
    assert cfg.allow_nonlocal_host is False


def test_llama_server_allow_nonlocal_host_explicit(tmp_vault):
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "provider": "llama_server",
        "base_url": "http://192.168.1.100:8080",
        "allow_nonlocal_host": True,
        "used_for": ["summaries"],
    }), encoding="utf-8")
    cfg = load_config(tmp_vault.root)
    assert cfg.allow_nonlocal_host is True
