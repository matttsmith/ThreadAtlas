"""Integration tests for LLM summarize / group naming / chunk gating.

These use the fake LLM from ``tests/fake_llm.py``. They exercise the full
store + runner path but never invoke a real model, so they pass
offline/deterministically.
"""

from __future__ import annotations

import json
import sys

from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation
from threadatlas.ingest import import_path
from threadatlas.llm import LLMRunner, load_config
from threadatlas.llm.chunking import llm_chunk_conversation
from threadatlas.llm.label_groups import label_all_groups
from threadatlas.llm.summarize import summarize_conversation


FAKE_CMD = [sys.executable, "-m", "tests.fake_llm"]


def _config(vault, mode, *, used_for):
    (vault.root / "local_llm.json").write_text(json.dumps({
        "command": FAKE_CMD + [mode],
        "timeout_seconds": 10,
        "max_prompt_chars": 20000,
        "max_response_chars": 4000,
        "used_for": list(used_for),
    }), encoding="utf-8")
    return LLMRunner(vault, load_config(vault.root), use_cache=False)


def _one_conversation(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([{
        "title": "Sample thread",
        "messages": [
            ("user", "Hey can you help me plan Project Delta Q2?", 1.0),
            ("assistant", "Sure, what constraints do you have?", 2.0),
            ("user", "Staffing and budget are the blockers.", 3.0),
            ("assistant", "Let's start with headcount.", 4.0),
        ],
    }])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value)
    chunk_conversation(store, cid)
    return cid


# --- summarize ------------------------------------------------------------

def test_summarize_updates_summary_and_source(tmp_vault, store, chatgpt_export_factory):
    cid = _one_conversation(tmp_vault, store, chatgpt_export_factory)
    runner = _config(tmp_vault, "summary_ok", used_for=("summaries",))
    outcome = summarize_conversation(tmp_vault, store, runner, cid)
    assert outcome.success
    row = store.conn.execute(
        "SELECT summary_short, summary_source FROM conversations WHERE conversation_id = ?",
        (cid,),
    ).fetchone()
    assert row["summary_short"] == "Short topical summary."
    assert row["summary_source"] == "llm"


def test_summarize_leaves_summary_on_malformed_response(tmp_vault, store, chatgpt_export_factory):
    cid = _one_conversation(tmp_vault, store, chatgpt_export_factory)
    # Take a snapshot before.
    before = store.conn.execute(
        "SELECT summary_short, summary_source FROM conversations WHERE conversation_id = ?",
        (cid,),
    ).fetchone()
    runner = _config(tmp_vault, "summary_malformed", used_for=("summaries",))
    outcome = summarize_conversation(tmp_vault, store, runner, cid)
    assert outcome.success is False
    after = store.conn.execute(
        "SELECT summary_short, summary_source FROM conversations WHERE conversation_id = ?",
        (cid,),
    ).fetchone()
    assert after["summary_short"] == before["summary_short"]
    assert after["summary_source"] == before["summary_source"]


def test_summarize_refuses_non_extractable_state(tmp_vault, store, chatgpt_export_factory):
    cid = _one_conversation(tmp_vault, store, chatgpt_export_factory)
    transition_state(store, cid, State.QUARANTINED.value)
    runner = _config(tmp_vault, "summary_ok", used_for=("summaries",))
    outcome = summarize_conversation(tmp_vault, store, runner, cid)
    assert outcome.success is False
    assert outcome.error and outcome.error.startswith("ineligible_state:")


# --- group naming ---------------------------------------------------------

def test_group_naming_sets_llm_label(tmp_vault, store, chatgpt_export_factory):
    # Seed enough conversations that clustering + naming actually runs.
    from threadatlas.cluster import regroup_all
    convs = []
    for i in range(6):
        convs.append({"title": f"Project Delta thread {i}", "messages": [
            ("user", f"Project Delta Q2 planning item {i}.", 1.0),
            ("assistant", "Noted.", 2.0),
        ]})
    path = chatgpt_export_factory(convs)
    res = import_path(tmp_vault, store, path)
    for cid in res.imported:
        transition_state(store, cid, State.INDEXED.value)
    regroup_all(store, broad_k=2, fine_k=4, seed=42)
    runner = _config(tmp_vault, "group_ok", used_for=("group_naming",))
    outcomes = label_all_groups(tmp_vault, store, runner)
    # At least one group has >= 3 members and should get labeled.
    labeled = [o for o in outcomes if o.success]
    assert labeled, f"no groups labeled; outcomes={[o.__dict__ for o in outcomes]}"
    for o in labeled:
        assert o.label == "chs staffing planning"


def test_group_naming_rejects_generic(tmp_vault, store, chatgpt_export_factory):
    from threadatlas.cluster import regroup_all
    convs = []
    for i in range(6):
        convs.append({"title": f"Project Delta thread {i}", "messages": [
            ("user", f"Delta staffing {i}.", 1.0),
            ("assistant", "Noted.", 2.0),
        ]})
    path = chatgpt_export_factory(convs)
    res = import_path(tmp_vault, store, path)
    for cid in res.imported:
        transition_state(store, cid, State.INDEXED.value)
    regroup_all(store, broad_k=2, fine_k=4, seed=42)
    runner = _config(tmp_vault, "group_generic", used_for=("group_naming",))
    outcomes = label_all_groups(tmp_vault, store, runner)
    # All named groups should have errored out on the generic rejection.
    assert all(not o.success or o.error == "generic_or_invalid" for o in outcomes)


# --- LLM-gated chunking ---------------------------------------------------

def _multi_topic_conversation(tmp_vault, store, chatgpt_export_factory):
    # Need enough messages per topic that the heuristic chunker won't merge
    # small trailing segments back into the previous one. We give each topic
    # four user+assistant pairs of substantial length.
    def topic_messages(topic_word: str, start_ordinal: int):
        msgs = []
        for i in range(4):
            user_text = f"{topic_word} discussion point {i} about {topic_word} specifics"
            asst_text = f"Regarding {topic_word}: " + f"{topic_word} analysis " * 60
            msgs.append(("user", user_text, float(start_ordinal + 2 * i)))
            msgs.append(("assistant", asst_text, float(start_ordinal + 2 * i + 1)))
        return msgs

    messages = topic_messages("cascade", 1) + topic_messages("risotto", 100) + topic_messages("budget", 200)
    path = chatgpt_export_factory([{"title": "Mixed thread", "messages": messages}])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value)
    return cid


def test_llm_chunking_gate_never_adds_splits(tmp_vault, store, chatgpt_export_factory):
    cid = _multi_topic_conversation(tmp_vault, store, chatgpt_export_factory)
    runner = _config(tmp_vault, "gate_split_true", used_for=("chunk_gating",))
    outcome = llm_chunk_conversation(tmp_vault, store, runner, cid)
    # When LLM always says "split=true", we keep all deterministic boundaries.
    assert outcome.after_chunks <= outcome.before_chunks


def test_llm_chunking_gate_can_merge_everything(tmp_vault, store, chatgpt_export_factory):
    cid = _multi_topic_conversation(tmp_vault, store, chatgpt_export_factory)
    runner = _config(tmp_vault, "gate_split_false", used_for=("chunk_gating",))
    outcome = llm_chunk_conversation(tmp_vault, store, runner, cid)
    # When LLM always says "split=false", all chunks should collapse to one.
    assert outcome.after_chunks == 1
    assert outcome.merges >= 1


def test_llm_chunking_failure_keeps_deterministic_boundaries(tmp_vault, store, chatgpt_export_factory):
    cid = _multi_topic_conversation(tmp_vault, store, chatgpt_export_factory)
    runner = _config(tmp_vault, "summary_malformed", used_for=("chunk_gating",))
    outcome = llm_chunk_conversation(tmp_vault, store, runner, cid)
    # Malformed responses -> preserve deterministic boundaries.
    assert outcome.after_chunks == outcome.before_chunks
    assert outcome.llm_failures > 0
