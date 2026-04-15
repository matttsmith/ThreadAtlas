"""End-to-end CLI smoke tests."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from threadatlas.cli.main import main


def _run(argv) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_init_then_review_empty(tmp_path: Path):
    vault = tmp_path / "vault"
    rc, out = _run(["init", str(vault)])
    assert rc == 0
    rc, out = _run(["review", str(vault)])
    assert rc == 0
    assert "No conversations" in out


def test_full_lifecycle(tmp_path: Path, chatgpt_export_factory):
    vault = tmp_path / "vault"
    _run(["init", str(vault)])
    export = chatgpt_export_factory([
        {"title": "Project Atlas planning",
         "messages": [
             ("user", "Project Atlas decisions: I will pick option B.", 1.0),
             ("assistant", "We agreed to ship in April.", 2.0),
             ("user", "TODO: revisit staffing.", 3.0),
             ("assistant", "Noted.", 4.0),
         ]},
    ])
    rc, out = _run(["import", str(vault), str(export), "--source", "chatgpt"])
    assert rc == 0
    assert "Imported: 1" in out

    # Discover the conversation id from review output.
    rc, out = _run(["review", str(vault)])
    assert rc == 0
    assert "Project Atlas" in out
    cid = next(line.split()[0] for line in out.splitlines() if line.startswith("conv_"))

    rc, _ = _run(["approve", str(vault), cid])
    assert rc == 0

    rc, _ = _run(["chunk", str(vault)])
    assert rc == 0
    rc, _ = _run(["extract", str(vault)])
    assert rc == 0

    rc, out = _run(["search", str(vault), "Atlas"])
    assert rc == 0
    assert cid[:18] in out

    out_path = tmp_path / "wb.xlsx"
    rc, _ = _run(["export", str(vault), "--profile", "review_workbook", "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()

    rc, out = _run(["inspect", str(vault), cid])
    assert rc == 0
    payload = json.loads(out)
    assert payload["conversation_id"] == cid
    assert payload["mcp_visible"] is True


def test_delete_requires_confirmation(tmp_path: Path, chatgpt_export_factory, monkeypatch):
    vault = tmp_path / "vault"
    _run(["init", str(vault)])
    export = chatgpt_export_factory([
        {"title": "X", "messages": [("user", "x", 1.0), ("assistant", "y", 2.0)]},
    ])
    _run(["import", str(vault), str(export)])
    rc, out = _run(["review", str(vault)])
    cid = next(line.split()[0] for line in out.splitlines() if line.startswith("conv_"))
    # Decline the prompt:
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    rc, out = _run(["delete", str(vault), cid])
    assert rc == 1
    assert "Aborted" in out
