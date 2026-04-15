"""MCP safe-label leak test.

Groups cluster over indexed + private. Labels exposed via MCP must be
derived from only the indexed members. This test builds a mixed group
where private content would heavily influence a mixed-pool label, and
confirms MCP receives a label that's derived from indexed-only text.
"""

from __future__ import annotations

import json

from threadatlas.cluster import regroup_all
from threadatlas.cluster.safe_labels import compute_safe_keyword_label
from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.ingest import import_path
from threadatlas.mcp.server import build_tools


def test_mcp_label_derives_only_from_indexed_members(tmp_vault, store, chatgpt_export_factory):
    convs = []
    # 3 indexed: all about widgets
    for i in range(3):
        convs.append({"title": f"widget engineering thread {i}", "messages": [
            ("user", f"widget engineering notes {i}", 1.0),
            ("assistant", "ok", 2.0),
        ]})
    # 5 private: all about therapy
    for i in range(5):
        convs.append({"title": f"therapy session notes {i}", "messages": [
            ("user", f"therapy anxiety processing {i}", 1.0),
            ("assistant", "ok", 2.0),
        ]})
    path = chatgpt_export_factory(convs)
    res = import_path(tmp_vault, store, path)
    for cid in res.imported[:3]:
        transition_state(store, cid, State.INDEXED.value)
    for cid in res.imported[3:]:
        transition_state(store, cid, State.PRIVATE.value)
    # Force a single broad group so all 8 cluster together.
    regroup_all(store, broad_k=1, fine_k=1, seed=42)

    group = store.list_groups(level="broad")[0]
    mixed_label = group["keyword_label"]
    # Mixed label will include 'therapy' / 'anxiety' from the private members.
    assert any(t in mixed_label for t in ("therapy", "anxiety", "processing")), \
        f"test precondition failed: mixed label doesn't include private terms: {mixed_label}"

    safe = compute_safe_keyword_label(store, group["group_id"])
    assert safe is not None
    # Safe label must NOT contain any terms derived from private content.
    for token in ("therapy", "anxiety", "processing"):
        assert token not in safe, f"private-derived token leaked into safe label: {safe}"
    # Safe label SHOULD contain indexed-derived terms.
    assert any(t in safe for t in ("widget", "engineering")), f"missing indexed terms: {safe}"


def test_mcp_list_groups_exposes_safe_label_only(tmp_vault, store, chatgpt_export_factory):
    convs = []
    for i in range(3):
        convs.append({"title": f"widget engineering thread {i}", "messages": [
            ("user", f"widget engineering notes {i}", 1.0),
            ("assistant", "ok", 2.0),
        ]})
    for i in range(5):
        convs.append({"title": f"therapy session notes {i}", "messages": [
            ("user", f"therapy anxiety processing {i}", 1.0),
            ("assistant", "ok", 2.0),
        ]})
    path = chatgpt_export_factory(convs)
    res = import_path(tmp_vault, store, path)
    for cid in res.imported[:3]:
        transition_state(store, cid, State.INDEXED.value)
    for cid in res.imported[3:]:
        transition_state(store, cid, State.PRIVATE.value)
    regroup_all(store, broad_k=1, fine_k=1, seed=42)

    tools = build_tools(tmp_vault, store)
    result = tools["list_groups"].fn({})
    payload = json.loads(result["content"][0]["text"])
    assert payload, "expected at least one visible group"
    for g in payload:
        label = g.get("label") or ""
        for token in ("therapy", "anxiety", "processing"):
            assert token not in label, f"MCP leaked private-derived token in list_groups: {g}"
        # MCP exposes keyword_label? It should NOT.
        assert "keyword_label" not in g, "MCP must not expose mixed-pool keyword_label"
        assert "llm_label" not in g, "MCP must not expose mixed-pool llm_label"


def test_mcp_list_groups_hides_groups_with_no_indexed_members(tmp_vault, store, chatgpt_export_factory):
    """If a group is 100% private, it has zero indexed visible_member_count,
    and must not appear in MCP list_groups at all."""
    convs = [{"title": f"private thread {i}", "messages": [
        ("user", f"private content {i}", 1.0),
        ("assistant", "ok", 2.0),
    ]} for i in range(6)]
    path = chatgpt_export_factory(convs)
    res = import_path(tmp_vault, store, path)
    for cid in res.imported:
        transition_state(store, cid, State.PRIVATE.value)
    regroup_all(store, broad_k=2, fine_k=2, seed=42)

    tools = build_tools(tmp_vault, store)
    result = tools["list_groups"].fn({})
    payload = json.loads(result["content"][0]["text"])
    assert payload == [], f"MCP exposed private-only groups: {payload}"
