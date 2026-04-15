"""Canonicalization: merge / rename / suppress derived objects."""

from __future__ import annotations

import time

import pytest

from threadatlas.core.models import DerivedKind, DerivedObject, ProvenanceLink, State, new_id
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path


def _seed_two_projects(tmp_vault, store, factory):
    """Seed two conversations that produce two distinct project objects."""
    path = factory([
        {"title": "Project Alpha planning", "messages": [
            ("user", "Project Alpha kickoff. I will pick option A.", 1.0),
            ("assistant", "ok", 2.0),
            ("user", "TODO: spec Alpha.", 3.0),
            ("assistant", "ok", 4.0),
        ]},
        {"title": "Project Bravo roadmap", "messages": [
            ("user", "Project Bravo scope. We agreed to ship Q4.", 1.0),
            ("assistant", "ok", 2.0),
            ("user", "TODO: review Bravo budget.", 3.0),
            ("assistant", "ok", 4.0),
        ]},
    ])
    res = import_path(tmp_vault, store, path)
    for cid in res.imported:
        transition_state(store, cid, State.INDEXED.value, vault=tmp_vault)
        chunk_conversation(store, cid)
        extract_for_conversation(store, cid)
    projects = {
        r["title"]: r["object_id"]
        for r in store.conn.execute(
            "SELECT object_id, title FROM derived_objects WHERE kind='project'"
        ).fetchall()
    }
    return res.imported, projects


def test_merge_collapses_two_projects_into_one(tmp_vault, store, chatgpt_export_factory):
    convs, projects = _seed_two_projects(tmp_vault, store, chatgpt_export_factory)
    alpha_id = next(v for k, v in projects.items() if "Alpha" in k)
    bravo_id = next(v for k, v in projects.items() if "Bravo" in k)

    prov_before = store.conn.execute(
        "SELECT COUNT(*) AS c FROM provenance_links WHERE object_id IN (?, ?)",
        (alpha_id, bravo_id),
    ).fetchone()["c"]
    assert prov_before >= 2

    report = store.merge_derived_objects(alpha_id, [bravo_id])
    store.conn.commit()
    assert report["losers_removed"] == 1
    assert report["provenance_links_moved"] >= 1

    # Loser is gone; winner has all provenance.
    assert store.get_derived_object(bravo_id) is None
    assert store.get_derived_object(alpha_id) is not None
    prov_after = store.conn.execute(
        "SELECT COUNT(*) AS c FROM provenance_links WHERE object_id = ?",
        (alpha_id,),
    ).fetchone()["c"]
    assert prov_after == prov_before


def test_merge_refuses_cross_kind(tmp_vault, store, chatgpt_export_factory):
    convs, projects = _seed_two_projects(tmp_vault, store, chatgpt_export_factory)
    alpha = next(v for k, v in projects.items() if "Alpha" in k)
    # Pick any decision to merge alpha into - should be rejected.
    dec_row = store.conn.execute(
        "SELECT object_id FROM derived_objects WHERE kind='decision' LIMIT 1"
    ).fetchone()
    assert dec_row is not None, "test precondition: need at least one decision"
    with pytest.raises(ValueError):
        store.merge_derived_objects(dec_row["object_id"], [alpha])


def test_merge_unknown_winner_raises(tmp_vault, store):
    with pytest.raises(KeyError):
        store.merge_derived_objects("obj_nosuch", ["obj_alsounknown"])


def test_rename(tmp_vault, store, chatgpt_export_factory):
    _convs, projects = _seed_two_projects(tmp_vault, store, chatgpt_export_factory)
    alpha = next(v for k, v in projects.items() if "Alpha" in k)
    store.rename_derived_object(alpha, "Project Alpha (renamed)")
    store.conn.commit()
    obj = store.get_derived_object(alpha)
    assert obj.title == "Project Alpha (renamed)"


def test_suppress_hides_from_listings(tmp_vault, store, chatgpt_export_factory):
    _convs, projects = _seed_two_projects(tmp_vault, store, chatgpt_export_factory)
    alpha = next(v for k, v in projects.items() if "Alpha" in k)
    store.suppress_derived_object(alpha)
    store.conn.commit()
    actives = [o.object_id for o in store.list_derived_objects(kind="project")]
    assert alpha not in actives
    # Unsuppress brings it back.
    store.unsuppress_derived_object(alpha)
    store.conn.commit()
    actives = [o.object_id for o in store.list_derived_objects(kind="project")]
    assert alpha in actives


def test_suppress_preserves_provenance_for_audit(tmp_vault, store, chatgpt_export_factory):
    _convs, projects = _seed_two_projects(tmp_vault, store, chatgpt_export_factory)
    alpha = next(v for k, v in projects.items() if "Alpha" in k)
    before = len(store.list_provenance_for_object(alpha))
    store.suppress_derived_object(alpha)
    store.conn.commit()
    after = len(store.list_provenance_for_object(alpha))
    assert before == after, "suppress must not touch provenance links"
