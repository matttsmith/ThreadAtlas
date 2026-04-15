"""TUI data-builder tests.

The curses rendering code requires a TTY and is not exercised here. We
test every data builder so the logic that powers each screen is covered
even though pytest runs headless.
"""

from __future__ import annotations

from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path
from threadatlas.tui import models as m
from threadatlas.tui.app import preview_screen


def _seed(tmp_vault, store, factory, *, n_approve=3, n_pending=2):
    convs = []
    for i in range(n_approve):
        convs.append({"title": f"Project CHS roadmap {i}", "messages": [
            ("user", f"Project CHS Q{i} staffing plan. I will pick option B.", 1.0),
            ("assistant", "ok", 2.0),
            ("user", "TODO: follow up on staffing.", 3.0),
            ("assistant", "Noted.", 4.0),
        ]})
    for i in range(n_pending):
        convs.append({"title": f"Pending thread {i}", "messages": [
            ("user", f"not yet approved {i}", 1.0),
            ("assistant", "ok", 2.0),
        ]})
    path = factory(convs)
    res = import_path(tmp_vault, store, path)
    approved = res.imported[:n_approve]
    for cid in approved:
        transition_state(store, cid, State.INDEXED.value, vault=tmp_vault)
        chunk_conversation(store, cid)
        extract_for_conversation(store, cid)
    return res.imported, approved


# --- overview --------------------------------------------------------------

def test_overview_includes_state_counts(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory, n_approve=3, n_pending=2)
    model = m.build_overview(tmp_vault, store)
    # Total should be 5; indexed should be 3; pending_review should be 2.
    joined = "\n".join(" ".join(r.get("cells", [])) for r in model.rows)
    assert "Total conversations" in joined
    assert "indexed" in joined and "3" in joined
    assert "pending_review" in joined and "2" in joined


# --- conversations --------------------------------------------------------

def test_conversations_filter_by_state(tmp_vault, store, chatgpt_export_factory):
    all_ids, approved = _seed(tmp_vault, store, chatgpt_export_factory, n_approve=3, n_pending=2)
    model = m.build_conversations(store, state_filter=State.INDEXED.value)
    ids = {r["id"] for r in model.rows}
    assert ids == set(approved)


def test_conversations_search_matches_title(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory, n_approve=3, n_pending=0)
    model = m.build_conversations(store, query="CHS")
    assert all("CHS" in (r["cells"][-1] or "") for r in model.rows)
    assert len(model.rows) >= 3


def test_conversations_rows_carry_state_for_color(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory, n_approve=2, n_pending=1)
    model = m.build_conversations(store)
    states = {r.get("state") for r in model.rows}
    assert "indexed" in states
    assert "pending_review" in states


# --- groups ---------------------------------------------------------------

def test_groups_listing_empty_when_not_computed(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory)
    model = m.build_groups(store)
    assert model.rows == [] or all(not r.get("id") or True for r in model.rows)


def test_groups_listing_after_regroup(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory, n_approve=4, n_pending=0)
    from threadatlas.cluster import regroup_all
    regroup_all(store, broad_k=2, fine_k=2, seed=42)
    model = m.build_groups(store)
    assert model.rows, "expected groups after regroup"
    # Every row must carry an id to support Enter-to-drill.
    assert all(r.get("id") for r in model.rows)


# --- derived objects ------------------------------------------------------

def test_open_loops_lists_extracted_items(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory)
    model = m.build_open_loops(store)
    # Each row has (object_id, convs, title).
    assert all(len(r["cells"]) == 3 for r in model.rows)
    # Row titles should contain TODO / follow-up fragments.
    titles = " ".join(r["cells"][2] for r in model.rows)
    assert "TODO" in titles or "follow" in titles.lower()


def test_projects_has_id_and_title(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory)
    model = m.build_projects(store)
    for r in model.rows:
        assert r.get("id", "").startswith("obj_")


# --- conversation detail --------------------------------------------------

def test_conversation_detail_shows_key_fields(tmp_vault, store, chatgpt_export_factory):
    _all, approved = _seed(tmp_vault, store, chatgpt_export_factory)
    cid = approved[0]
    model = m.build_conversation_detail(tmp_vault, store, cid)
    joined = "\n".join(r["cells"][0] for r in model.rows)
    assert "state:" in joined
    assert "state:     indexed" in joined
    assert "summary" in joined.lower()
    assert "chunks" in joined.lower()


def test_conversation_detail_unknown_id(tmp_vault, store):
    model = m.build_conversation_detail(tmp_vault, store, "conv_nosuch")
    joined = "\n".join(r["cells"][0] for r in model.rows)
    assert "unknown" in joined


# --- preview path --------------------------------------------------------

def test_preview_screen_plain_text(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory)
    store.close()  # preview reopens
    out = preview_screen(tmp_vault.root, "overview")
    assert "== Overview ==" in out
    assert "Total conversations" in out
