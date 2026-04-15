"""Vault layout invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from threadatlas.core.vault import init_vault, open_vault, VAULT_SUBDIRS


def test_init_creates_subdirs(tmp_path: Path):
    v = init_vault(tmp_path / "vault")
    for sub in VAULT_SUBDIRS:
        assert (v.root / sub).is_dir()
    assert v.marker_path.exists()


def test_init_is_idempotent(tmp_path: Path):
    v1 = init_vault(tmp_path / "vault")
    v2 = init_vault(tmp_path / "vault")
    assert v1.root == v2.root
    assert v1.marker_path.read_text() == v2.marker_path.read_text()


def test_open_vault_requires_marker(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(FileNotFoundError):
        open_vault(raw)


def test_normalized_path_is_sharded(tmp_path: Path):
    v = init_vault(tmp_path / "vault")
    p = v.normalized_path_for("conv_abcdef0123456789")
    assert p.parent.name == "ab"
    assert p.name == "conv_abcdef0123456789.json"
