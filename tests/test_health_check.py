"""Vault health check tests."""

from __future__ import annotations

from threadatlas import health
from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.ingest import import_path


def test_healthy_vault_returns_no_warnings(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "X", "messages": [("user", "x", 1.0), ("assistant", "y", 2.0)]},
    ])
    import_path(tmp_vault, store, path)
    assert health.quick_check(tmp_vault, store) == []


def test_missing_normalized_file_is_flagged(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "X", "messages": [("user", "x", 1.0), ("assistant", "y", 2.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    # Remove the normalized file to create a drift.
    norm = tmp_vault.normalized_path_for(cid)
    norm.unlink()
    warnings = health.quick_check(tmp_vault, store)
    assert any("missing normalized JSON" in w for w in warnings)


def test_visibility_leak_is_flagged(tmp_vault, store, chatgpt_export_factory):
    """Smuggle an fts_messages row in for a quarantined conversation; the
    health check must surface it."""
    path = chatgpt_export_factory([
        {"title": "leaky", "messages": [("user", "x", 1.0), ("assistant", "y", 2.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value)
    # Now go back to quarantined.
    transition_state(store, cid, State.QUARANTINED.value)
    # Manually insert an FTS row to simulate a leak.
    msg_row = store.conn.execute(
        "SELECT rowid FROM messages WHERE conversation_id = ? LIMIT 1", (cid,)
    ).fetchone()
    store.conn.execute(
        "INSERT INTO fts_messages(rowid, body, role) VALUES (?, ?, ?)",
        (msg_row["rowid"], "leaked content", "user"),
    )
    store.conn.commit()
    warnings = health.quick_check(tmp_vault, store)
    assert any("VISIBILITY LEAK" in w for w in warnings)
