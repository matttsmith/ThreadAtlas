"""Tag/untag, summarize resumability, report generation."""

from __future__ import annotations

import json
import sys

from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.ingest import import_path
from threadatlas.llm import LLMRunner, load_config
from threadatlas.llm.summarize import summarize_all_eligible
from threadatlas.report import generate_report


FAKE_CMD = [sys.executable, "-m", "tests.fake_llm"]


def _seed(tmp_vault, store, factory, n=3):
    convs = [{"title": f"Thread {i}", "messages": [
        ("user", f"topic {i} discussion", 1.0),
        ("assistant", "ok", 2.0),
    ]} for i in range(n)]
    path = factory(convs)
    res = import_path(tmp_vault, store, path)
    for cid in res.imported:
        transition_state(store, cid, State.INDEXED.value)
    return res.imported


# ---- tagging -------------------------------------------------------------

def test_tag_adds_and_dedupes(tmp_vault, store, chatgpt_export_factory):
    ids = _seed(tmp_vault, store, chatgpt_export_factory, n=1)
    cid = ids[0]
    store.add_manual_tags(cid, ["chs", "urgent"])
    store.add_manual_tags(cid, ["urgent", "q2"])  # dedupe 'urgent'
    store.conn.commit()
    c = store.get_conversation(cid)
    assert c.manual_tags == ["chs", "urgent", "q2"]


def test_untag_removes(tmp_vault, store, chatgpt_export_factory):
    ids = _seed(tmp_vault, store, chatgpt_export_factory, n=1)
    cid = ids[0]
    store.add_manual_tags(cid, ["chs", "urgent", "q2"])
    store.remove_manual_tags(cid, ["urgent"])
    store.conn.commit()
    c = store.get_conversation(cid)
    assert c.manual_tags == ["chs", "q2"]


def test_tagged_conversation_is_fts_searchable_by_tag(tmp_vault, store, chatgpt_export_factory):
    ids = _seed(tmp_vault, store, chatgpt_export_factory, n=1)
    cid = ids[0]
    store.add_manual_tags(cid, ["lambda_project"])
    store.conn.commit()
    from threadatlas.search import search_conversations
    hits = search_conversations(store, "lambda_project", visible_states=(State.INDEXED.value,))
    assert any(h.conversation_id == cid for h in hits)


# ---- summarize resumability ---------------------------------------------

def _runner(tmp_vault, mode, used_for=("summaries",)):
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "command": FAKE_CMD + [mode],
        "timeout_seconds": 10,
        "max_prompt_chars": 5000,
        "max_response_chars": 2000,
        "used_for": list(used_for),
    }), encoding="utf-8")
    return LLMRunner(tmp_vault, load_config(tmp_vault.root), use_cache=False)


def test_summarize_resumable_skips_already_summarized(tmp_vault, store, chatgpt_export_factory):
    ids = _seed(tmp_vault, store, chatgpt_export_factory, n=3)
    runner = _runner(tmp_vault, "summary_ok")
    outcomes = summarize_all_eligible(tmp_vault, store, runner)
    assert all(o.success for o in outcomes)
    # Second run with force=False should skip everything.
    outcomes2 = summarize_all_eligible(tmp_vault, store, runner)
    assert len(outcomes2) == 0
    # But force=True should redo them.
    outcomes3 = summarize_all_eligible(tmp_vault, store, runner, force=True)
    assert len(outcomes3) == 3


# ---- HTML report --------------------------------------------------------

def test_report_generates_html_file(tmp_vault, store, chatgpt_export_factory):
    ids = _seed(tmp_vault, store, chatgpt_export_factory, n=4)
    out = generate_report(tmp_vault, store)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<html" in html and "ThreadAtlas report" in html
    # Report must not contain any forbidden external references.
    for bad in ("http://", "https://", "<script", "<iframe"):
        assert bad not in html, f"report contains forbidden content: {bad}"


def test_report_shows_state_counts(tmp_vault, store, chatgpt_export_factory):
    ids = _seed(tmp_vault, store, chatgpt_export_factory, n=3)
    # Move one to private.
    transition_state(store, ids[0], State.PRIVATE.value)
    out = generate_report(tmp_vault, store)
    html = out.read_text(encoding="utf-8")
    assert "indexed" in html
    assert "private" in html
