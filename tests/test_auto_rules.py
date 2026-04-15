"""Auto-rules: import-time down-classification, auto-approve, rescan."""

from __future__ import annotations

import json

import pytest

from threadatlas.core.models import State
from threadatlas.ingest import import_path
from threadatlas.rescan import rescan
from threadatlas.rules import evaluate, load_rules


# ---------------------------------------------------------------------------
# Rules engine unit tests
# ---------------------------------------------------------------------------

def _write_rules(tmp_vault, rules: dict) -> None:
    (tmp_vault.root / "auto_rules.json").write_text(
        json.dumps(rules), encoding="utf-8"
    )


def test_no_rules_file_is_empty_ruleset(tmp_vault):
    rs = load_rules(tmp_vault.root)
    assert rs.empty


def test_keyword_rule_matches_case_insensitive(tmp_vault):
    _write_rules(tmp_vault, {
        "auto_private": [{"patterns": ["THERAPY"], "fields": ["title"]}],
    })
    rs = load_rules(tmp_vault.root)
    target, matches = evaluate(rs, title="weekly therapy check-in")
    assert target == State.PRIVATE.value
    assert matches and matches[0].pattern == "THERAPY"


def test_regex_rule_matches_ssn_shape(tmp_vault):
    _write_rules(tmp_vault, {
        "auto_private": [{
            "patterns": [r"\b\d{3}-\d{2}-\d{4}\b"],
            "mode": "regex",
            "fields": ["messages"],
        }],
    })
    rs = load_rules(tmp_vault.root)
    target, _ = evaluate(
        rs, title="benign title",
        messages=["user said 123-45-6789 here"],
    )
    assert target == State.PRIVATE.value


def test_quarantine_wins_over_private_when_both_match(tmp_vault):
    _write_rules(tmp_vault, {
        "auto_private":    [{"patterns": ["anxious"], "fields": ["messages"]}],
        "auto_quarantine": [{"patterns": ["[no-index]"], "fields": ["title"]}],
    })
    rs = load_rules(tmp_vault.root)
    target, _ = evaluate(
        rs, title="[no-index] personal log",
        messages=["I feel anxious today"],
    )
    assert target == State.QUARANTINED.value


def test_invalid_regex_rejected_loudly(tmp_vault):
    _write_rules(tmp_vault, {
        "auto_private": [{"patterns": ["(unclosed"], "mode": "regex", "fields": ["title"]}],
    })
    with pytest.raises(ValueError):
        load_rules(tmp_vault.root)


def test_unknown_field_rejected_loudly(tmp_vault):
    _write_rules(tmp_vault, {
        "auto_private": [{"patterns": ["x"], "fields": ["messages", "bogus"]}],
    })
    with pytest.raises(ValueError):
        load_rules(tmp_vault.root)


# ---------------------------------------------------------------------------
# Import-time behavior
# ---------------------------------------------------------------------------

def test_import_routes_match_to_private_not_indexed(tmp_vault, store, chatgpt_export_factory):
    _write_rules(tmp_vault, {
        "auto_private": [{"patterns": ["therapy"], "fields": ["messages"]}],
    })
    path = chatgpt_export_factory([
        {"title": "weekly check", "messages": [
            ("user", "therapy discussion details", 1.0),
            ("assistant", "ok", 2.0),
        ]},
        {"title": "project standup", "messages": [
            ("user", "Project planning for Q2 delivery", 1.0),
            ("assistant", "ok", 2.0),
        ]},
    ])
    result = import_path(tmp_vault, store, path)
    assert result.auto_rule_matches == 1
    rows = store.conn.execute(
        "SELECT title, state, notes_local FROM conversations ORDER BY title"
    ).fetchall()
    by_title = {r["title"]: r for r in rows}
    # Therapy thread should be private; project thread pending_review.
    assert by_title["weekly check"]["state"] == State.PRIVATE.value
    assert "auto-rule match" in by_title["weekly check"]["notes_local"]
    assert by_title["project standup"]["state"] == State.PENDING_REVIEW.value


def test_auto_approve_lifts_non_matching_to_indexed(tmp_vault, store, chatgpt_export_factory):
    _write_rules(tmp_vault, {
        "auto_private": [{"patterns": ["therapy"], "fields": ["messages"]}],
    })
    path = chatgpt_export_factory([
        {"title": "therapy check", "messages": [
            ("user", "therapy notes", 1.0), ("assistant", "ok", 2.0),
        ]},
        {"title": "harmless project", "messages": [
            ("user", "Project Q2 plan", 1.0), ("assistant", "ok", 2.0),
        ]},
    ])
    result = import_path(tmp_vault, store, path, auto_approve=True)
    assert result.by_initial_state.get(State.PRIVATE.value) == 1
    assert result.by_initial_state.get(State.INDEXED.value) == 1
    assert result.by_initial_state.get(State.PENDING_REVIEW.value, 0) == 0
    # Rule match must override --auto-approve.
    rows = store.conn.execute(
        "SELECT title, state FROM conversations ORDER BY title"
    ).fetchall()
    by_title = {r["title"]: r["state"] for r in rows}
    assert by_title["therapy check"] == State.PRIVATE.value
    assert by_title["harmless project"] == State.INDEXED.value


def test_auto_approve_without_rules_still_works(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "project x", "messages": [
            ("user", "anything", 1.0), ("assistant", "ok", 2.0),
        ]},
    ])
    result = import_path(tmp_vault, store, path, auto_approve=True)
    assert result.by_initial_state == {State.INDEXED.value: 1}
    c = store.list_conversations()[0]
    assert c.state == State.INDEXED.value


# ---------------------------------------------------------------------------
# Rescan-rules: down-classify only
# ---------------------------------------------------------------------------

def test_rescan_down_classifies_existing_corpus(tmp_vault, store, chatgpt_export_factory):
    # Import WITHOUT rules, approve everything.
    path = chatgpt_export_factory([
        {"title": "therapy check", "messages": [
            ("user", "therapy session notes", 1.0),
            ("assistant", "ok", 2.0),
        ]},
        {"title": "project planning", "messages": [
            ("user", "Q2 roadmap review", 1.0),
            ("assistant", "ok", 2.0),
        ]},
    ])
    result = import_path(tmp_vault, store, path, auto_approve=True)
    assert result.by_initial_state.get(State.INDEXED.value) == 2

    # Add rules AFTER import and rescan.
    _write_rules(tmp_vault, {
        "auto_private": [{"patterns": ["therapy"], "fields": ["messages"]}],
    })
    rscan = rescan(tmp_vault, store)
    assert rscan.scanned == 2
    assert rscan.down_classified == 1
    # Therapy thread should now be private.
    rows = store.conn.execute(
        "SELECT title, state FROM conversations ORDER BY title"
    ).fetchall()
    by_title = {r["title"]: r["state"] for r in rows}
    assert by_title["therapy check"] == State.PRIVATE.value
    assert by_title["project planning"] == State.INDEXED.value


def test_rescan_never_up_classifies(tmp_vault, store, chatgpt_export_factory):
    """Having a rule should never pull a private conversation BACK to indexed.

    Rescan only moves toward more restrictive states.
    """
    path = chatgpt_export_factory([
        {"title": "kept private manually", "messages": [
            ("user", "planet saturn cat bicycle", 1.0),
            ("assistant", "ok", 2.0),
        ]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    from threadatlas.core.workflow import transition_state
    transition_state(store, cid, State.PRIVATE.value, vault=tmp_vault)

    # No rule matches at all.
    _write_rules(tmp_vault, {
        "auto_private": [{"patterns": ["completely unrelated"], "fields": ["title"]}],
    })
    rscan = rescan(tmp_vault, store)
    assert rscan.down_classified == 0
    c = store.get_conversation(cid)
    assert c.state == State.PRIVATE.value


def test_rescan_private_to_quarantined_allowed(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "[no-index] sensitive note", "messages": [
            ("user", "something about taxes", 1.0),
            ("assistant", "ok", 2.0),
        ]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    from threadatlas.core.workflow import transition_state
    transition_state(store, cid, State.PRIVATE.value, vault=tmp_vault)

    _write_rules(tmp_vault, {
        "auto_quarantine": [{"patterns": ["[no-index]"], "fields": ["title"]}],
    })
    rscan = rescan(tmp_vault, store)
    assert rscan.down_classified == 1
    assert store.get_conversation(cid).state == State.QUARANTINED.value
