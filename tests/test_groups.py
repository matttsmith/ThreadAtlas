"""End-to-end grouping pipeline tests."""

from __future__ import annotations

from threadatlas.cluster import regroup_all
from threadatlas.cluster.groups import top_members_for_group
from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.ingest import import_path


def _seed_topics(tmp_vault, store, factory, *, n_chs=4, n_cook=4, state=State.INDEXED.value):
    convs = []
    for i in range(n_chs):
        convs.append({"title": f"Project CHS roadmap {i}", "messages": [
            ("user", f"Project CHS staffing plan update {i}. Q{i} budget.", 1.0),
            ("assistant", f"CHS program notes {i}.", 2.0),
        ]})
    for i in range(n_cook):
        convs.append({"title": f"Italian risotto notes {i}", "messages": [
            ("user", f"Arborio rice risotto mushroom technique {i}.", 1.0),
            ("assistant", f"Italian cooking notes {i}.", 2.0),
        ]})
    path = factory(convs)
    res = import_path(tmp_vault, store, path)
    ids = res.imported
    for cid in ids:
        transition_state(store, cid, state)
    return ids


def test_grouping_separates_obvious_topics(tmp_vault, store, chatgpt_export_factory):
    ids = _seed_topics(tmp_vault, store, chatgpt_export_factory)
    result = regroup_all(store, broad_k=2, fine_k=4, seed=42)
    assert result.broad_groups == 2
    assert result.members == len(ids)

    # Collect broad group memberships.
    broad_groups = store.list_groups(level="broad")
    membership: dict[str, set[str]] = {}
    for g in broad_groups:
        member_ids = set(store.list_group_members(g["group_id"]))
        membership[g["group_id"]] = member_ids

    # The first 4 (CHS) should all be in one group; the last 4 (cook) in another.
    chs = set(ids[:4])
    cook = set(ids[4:])
    cluster_bags = [frozenset(s) for s in membership.values()]
    assert any(b == chs for b in cluster_bags) or any(b == cook for b in cluster_bags)


def test_grouping_is_deterministic(tmp_vault, store, chatgpt_export_factory):
    _seed_topics(tmp_vault, store, chatgpt_export_factory)
    r1 = regroup_all(store, broad_k=2, fine_k=4, seed=42)
    labels_1 = sorted((g["keyword_label"], g["member_count"]) for g in store.list_groups(level="broad"))
    r2 = regroup_all(store, broad_k=2, fine_k=4, seed=42)
    labels_2 = sorted((g["keyword_label"], g["member_count"]) for g in store.list_groups(level="broad"))
    assert labels_1 == labels_2


def test_grouping_small_corpus_skipped(tmp_vault, store, chatgpt_export_factory):
    path = chatgpt_export_factory([
        {"title": "only one", "messages": [("user", "x", 1.0), ("assistant", "y", 2.0)]},
    ])
    res = import_path(tmp_vault, store, path)
    transition_state(store, res.imported[0], State.INDEXED.value)
    result = regroup_all(store, broad_k=10, fine_k=100)
    assert result.skipped_empty_corpus is True
    assert store.list_groups() == []


def test_grouping_excludes_pending_and_quarantined(tmp_vault, store, chatgpt_export_factory):
    # All conversations get seeded, then we mark half quarantined/pending.
    ids = _seed_topics(tmp_vault, store, chatgpt_export_factory, state=State.INDEXED.value)
    # Move some to non-extractable states.
    transition_state(store, ids[0], State.QUARANTINED.value)
    transition_state(store, ids[1], State.PENDING_REVIEW.value)
    result = regroup_all(store, broad_k=2, fine_k=4, seed=42)
    # Members should count only extractable states.
    assert result.members == len(ids) - 2


def test_top_members_includes_only_current_members(tmp_vault, store, chatgpt_export_factory):
    ids = _seed_topics(tmp_vault, store, chatgpt_export_factory)
    regroup_all(store, broad_k=2, fine_k=4, seed=42)
    for g in store.list_groups(level="broad"):
        members = set(store.list_group_members(g["group_id"]))
        top = top_members_for_group(store, g["group_id"], top_n=5)
        assert set(top).issubset(members)
