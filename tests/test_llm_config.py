"""LLM config loader tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from threadatlas.llm import load_config
from threadatlas.llm.config import LLMConfig


def test_missing_config_returns_none(tmp_vault):
    assert load_config(tmp_vault.root) is None


def test_valid_config_parses(tmp_vault):
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
    assert cfg.is_enabled_for("summaries")
    assert not cfg.is_enabled_for("group_naming")


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
