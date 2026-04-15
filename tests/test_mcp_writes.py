"""MCP write tools: opt-in gating, audit log, refusal paths."""

from __future__ import annotations

import json

from threadatlas.cluster import regroup_all
from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path
from threadatlas.mcp.server import build_tools
from threadatlas.mcp.writes import MUTATION_LOG_BASENAME


def _seed(tmp_vault, store, factory, *, with_private=False):
    convs = [
        {"title": "Project CHS planning", "messages": [
            ("user", "Project CHS kickoff. I will pick option B.", 1.0),
            ("assistant", "ok", 2.0),
            ("user", "TODO: followups.", 3.0),
            ("assistant", "ok", 4.0),
        ]},
    ]
    if with_private:
        convs.append({"title": "therapy notes", "messages": [
            ("user", "private session", 1.0),
            ("assistant", "ok", 2.0),
        ]})
    path = factory(convs)
    res = import_path(tmp_vault, store, path)
    transition_state(store, res.imported[0], State.INDEXED.value, vault=tmp_vault)
    if with_private:
        transition_state(store, res.imported[1], State.PRIVATE.value, vault=tmp_vault)
    for cid in res.imported:
        chunk_conversation(store, cid)
        extract_for_conversation(store, cid)
    return res.imported


def _enable_writes(tmp_vault):
    (tmp_vault.root / "mcp_config.json").write_text(
        json.dumps({"allow_writes": True}), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Opt-in gating
# ---------------------------------------------------------------------------

def test_writes_not_registered_by_default(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory)
    tools = build_tools(tmp_vault, store)
    for name in ("set_group_label", "add_tag", "remove_tag", "rename_derived_object"):
        assert name not in tools, f"{name} should not be registered without opt-in"


def test_writes_registered_when_enabled(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory)
    _enable_writes(tmp_vault)
    tools = build_tools(tmp_vault, store)
    for name in ("set_group_label", "add_tag", "remove_tag", "rename_derived_object"):
        assert name in tools


# ---------------------------------------------------------------------------
# set_group_label
# ---------------------------------------------------------------------------

def test_set_group_label_updates_and_logs(tmp_vault, store, chatgpt_export_factory):
    # Need enough conversations for grouping to produce a group.
    path = chatgpt_export_factory([
        {"title": f"Project CHS topic {i}", "messages": [
            ("user", f"CHS planning item {i}.", 1.0), ("assistant", "ok", 2.0),
        ]} for i in range(5)
    ])
    res = import_path(tmp_vault, store, path)
    for cid in res.imported:
        transition_state(store, cid, State.INDEXED.value, vault=tmp_vault)
    regroup_all(store, broad_k=1, fine_k=1, seed=42)
    group = store.list_groups(level="broad")[0]
    _enable_writes(tmp_vault)
    tools = build_tools(tmp_vault, store)
    result = tools["set_group_label"].fn({
        "group_id": group["group_id"], "label": "CHS program management",
    })
    assert "isError" not in result
    refreshed = store.get_group(group["group_id"])
    assert refreshed["llm_label"] == "CHS program management"
    log = (tmp_vault.logs / MUTATION_LOG_BASENAME).read_text().splitlines()
    assert log
    entry = json.loads(log[-1])
    assert entry["tool"] == "set_group_label"
    assert entry["new_label"] == "CHS program management"


# ---------------------------------------------------------------------------
# add_tag / remove_tag
# ---------------------------------------------------------------------------

def test_add_tag_on_indexed_conversation_ok(tmp_vault, store, chatgpt_export_factory):
    cid = _seed(tmp_vault, store, chatgpt_export_factory)[0]
    _enable_writes(tmp_vault)
    tools = build_tools(tmp_vault, store)
    result = tools["add_tag"].fn({"conversation_id": cid, "tags": ["urgent", "CHS"]})
    assert "isError" not in result
    c = store.get_conversation(cid)
    assert set(c.manual_tags) == {"urgent", "CHS"}


def test_add_tag_refuses_non_indexed(tmp_vault, store, chatgpt_export_factory):
    ids = _seed(tmp_vault, store, chatgpt_export_factory, with_private=True)
    private_id = ids[1]
    _enable_writes(tmp_vault)
    tools = build_tools(tmp_vault, store)
    result = tools["add_tag"].fn({"conversation_id": private_id, "tags": ["leak"]})
    assert result.get("isError") is True
    # Verify no tag landed on the private conversation.
    c = store.get_conversation(private_id)
    assert c.manual_tags == []


def test_add_tag_caps_count_and_length(tmp_vault, store, chatgpt_export_factory):
    cid = _seed(tmp_vault, store, chatgpt_export_factory)[0]
    _enable_writes(tmp_vault)
    tools = build_tools(tmp_vault, store)
    too_many = [f"t{i}" for i in range(30)]
    huge_label = "x" * 500
    result = tools["add_tag"].fn({"conversation_id": cid, "tags": too_many + [huge_label]})
    assert "isError" not in result
    c = store.get_conversation(cid)
    # Cap at MAX_TAGS_PER_CALL=10.
    assert len(c.manual_tags) <= 10
    # Individual tags are truncated.
    assert all(len(t) <= 60 for t in c.manual_tags)


def test_remove_tag_ok(tmp_vault, store, chatgpt_export_factory):
    cid = _seed(tmp_vault, store, chatgpt_export_factory)[0]
    _enable_writes(tmp_vault)
    tools = build_tools(tmp_vault, store)
    tools["add_tag"].fn({"conversation_id": cid, "tags": ["a", "b", "c"]})
    tools["remove_tag"].fn({"conversation_id": cid, "tags": ["b"]})
    c = store.get_conversation(cid)
    assert set(c.manual_tags) == {"a", "c"}


# ---------------------------------------------------------------------------
# rename_derived_object
# ---------------------------------------------------------------------------

def test_rename_derived_object_updates_title(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory)
    obj = next(o for o in store.list_derived_objects(kind="project"))
    _enable_writes(tmp_vault)
    tools = build_tools(tmp_vault, store)
    result = tools["rename_derived_object"].fn({
        "object_id": obj.object_id, "title": "Center for Health Security program",
    })
    assert "isError" not in result
    refreshed = store.get_derived_object(obj.object_id)
    assert refreshed.title == "Center for Health Security program"
    # Audit log entry written.
    log_lines = (tmp_vault.logs / MUTATION_LOG_BASENAME).read_text().splitlines()
    assert any(json.loads(ln)["tool"] == "rename_derived_object" for ln in log_lines)


def test_rename_refuses_object_with_no_indexed_provenance(tmp_vault, store, chatgpt_export_factory):
    """An object whose provenance is entirely from non-indexed conversations
    should not be renamable through MCP."""
    path = chatgpt_export_factory([
        {"title": "private thread with Project Ghost", "messages": [
            ("user", "Project Ghost is a private workstream we discussed.", 1.0),
            ("assistant", "ok", 2.0),
            ("user", "more Project Ghost detail.", 3.0),
            ("assistant", "ok", 4.0),
        ]},
    ])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.PRIVATE.value, vault=tmp_vault)
    chunk_conversation(store, cid)
    extract_for_conversation(store, cid)
    ghost = next(
        (o for o in store.list_derived_objects(kind="project")
         if "ghost" in o.title.lower()),
        None,
    )
    assert ghost is not None, "test precondition: expected Project Ghost to be extracted"
    _enable_writes(tmp_vault)
    tools = build_tools(tmp_vault, store)
    result = tools["rename_derived_object"].fn({
        "object_id": ghost.object_id, "title": "Renamed via MCP",
    })
    assert result.get("isError") is True
    # Title must not have changed.
    assert store.get_derived_object(ghost.object_id).title == ghost.title


def test_rename_unknown_object_errors_cleanly(tmp_vault, store, chatgpt_export_factory):
    _seed(tmp_vault, store, chatgpt_export_factory)
    _enable_writes(tmp_vault)
    tools = build_tools(tmp_vault, store)
    result = tools["rename_derived_object"].fn({
        "object_id": "obj_nosuch", "title": "anything",
    })
    assert result.get("isError") is True


# ---------------------------------------------------------------------------
# Audit log does not contain content
# ---------------------------------------------------------------------------

def test_mutation_log_does_not_contain_message_bodies(tmp_vault, store, chatgpt_export_factory):
    cid = _seed(tmp_vault, store, chatgpt_export_factory)[0]
    _enable_writes(tmp_vault)
    tools = build_tools(tmp_vault, store)
    tools["add_tag"].fn({"conversation_id": cid, "tags": ["marker"]})
    log = (tmp_vault.logs / MUTATION_LOG_BASENAME).read_text()
    assert "Project CHS kickoff" not in log
    assert "TODO" not in log
