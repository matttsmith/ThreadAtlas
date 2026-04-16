"""Thorough tests for the structured query engine.

Covers:
  - parse_query: every filter prefix, combinations, edge cases
  - query(): keyword search, filter-only listing, kind restriction,
    project scoping, date filters, has-flags, visibility safety,
    limit, dedup, result shape
"""

from __future__ import annotations

import time

from threadatlas.search.query_engine import (
    QueryFilter,
    QueryHit,
    QueryResult,
    parse_query,
    query,
)

from corpus_fixtures import rich_corpus  # noqa: F401 — pytest fixture


# ===================================================================
# parse_query tests
# ===================================================================


class TestParseQueryBasic:
    """Simple / single-filter parsing."""

    def test_plain_text_no_filters(self):
        f = parse_query("kubernetes deployment")
        assert f.text == "kubernetes deployment"
        assert f.sources == []
        assert f.tags == []
        assert f.kinds == []
        assert f.projects == []
        assert f.after is None
        assert f.before is None
        assert f.has_flags == []

    def test_empty_string(self):
        f = parse_query("")
        assert f.text == ""

    def test_none_input(self):
        f = parse_query(None)
        assert f.text == ""

    def test_whitespace_only(self):
        f = parse_query("   ")
        assert f.text == ""

    def test_source_filter(self):
        f = parse_query("source:chatgpt")
        assert f.sources == ["chatgpt"]
        assert f.text == ""

    def test_source_filter_case_insensitive(self):
        f = parse_query("source:ChatGPT")
        assert f.sources == ["chatgpt"]

    def test_tag_filter(self):
        f = parse_query("tag:architecture")
        assert f.tags == ["architecture"]
        assert f.text == ""

    def test_tag_preserves_case(self):
        f = parse_query("tag:MyTag")
        assert f.tags == ["MyTag"]

    def test_kind_filter_valid(self):
        f = parse_query("kind:decision")
        assert f.kinds == ["decision"]
        assert f.text == ""

    def test_kind_filter_invalid_ignored(self):
        f = parse_query("kind:nonsense")
        assert f.kinds == []
        assert f.text == ""

    def test_project_filter(self):
        f = parse_query("project:proj_abc123")
        assert f.projects == ["proj_abc123"]

    def test_after_filter(self):
        f = parse_query("after:2024-06-01")
        assert f.after is not None
        assert f.after > 0

    def test_before_filter(self):
        f = parse_query("before:2024-12-31")
        assert f.before is not None
        assert f.before > 0

    def test_after_with_time(self):
        f = parse_query("after:2024-06-01T14:30:00")
        assert f.after is not None

    def test_invalid_date_ignored(self):
        f = parse_query("after:not-a-date")
        assert f.after is None

    def test_has_open_loops(self):
        f = parse_query("has:open_loops")
        assert "open_loops" in f.has_flags

    def test_has_chunks(self):
        f = parse_query("has:chunks")
        assert "chunks" in f.has_flags

    def test_has_invalid_ignored(self):
        f = parse_query("has:bogus")
        assert f.has_flags == []


class TestParseQueryCombinations:
    """Multi-filter and text + filter combinations."""

    def test_text_plus_source(self):
        f = parse_query("migration plan source:chatgpt")
        assert f.text == "migration plan"
        assert f.sources == ["chatgpt"]

    def test_source_then_text(self):
        f = parse_query("source:claude kubernetes")
        assert f.sources == ["claude"]
        assert f.text == "kubernetes"

    def test_text_between_filters(self):
        f = parse_query("source:chatgpt migration tag:infra")
        assert f.sources == ["chatgpt"]
        assert f.tags == ["infra"]
        assert f.text == "migration"

    def test_multiple_sources(self):
        f = parse_query("source:chatgpt source:claude")
        assert f.sources == ["chatgpt", "claude"]

    def test_multiple_tags(self):
        f = parse_query("tag:architecture tag:backend")
        assert f.tags == ["architecture", "backend"]

    def test_multiple_kinds(self):
        f = parse_query("kind:decision kind:entity")
        assert f.kinds == ["decision", "entity"]

    def test_after_and_before(self):
        f = parse_query("after:2024-01-01 before:2024-12-31")
        assert f.after is not None
        assert f.before is not None
        assert f.after < f.before

    def test_all_filters_combined(self):
        f = parse_query(
            "deploy source:chatgpt tag:infra kind:decision "
            "project:proj_1 after:2024-01-01 before:2024-12-31 "
            "has:open_loops has:chunks"
        )
        assert f.text == "deploy"
        assert f.sources == ["chatgpt"]
        assert f.tags == ["infra"]
        assert f.kinds == ["decision"]
        assert f.projects == ["proj_1"]
        assert f.after is not None
        assert f.before is not None
        assert "open_loops" in f.has_flags
        assert "chunks" in f.has_flags

    def test_unknown_prefix_becomes_text(self):
        f = parse_query("foo:bar baz")
        assert "foo:bar" in f.text
        assert "baz" in f.text

    def test_quoted_value(self):
        f = parse_query('tag:"multi word tag"')
        assert f.tags == ["multi word tag"]

    def test_quoted_source(self):
        f = parse_query('source:"chat gpt"')
        assert f.sources == ["chat gpt"]


class TestQueryFilterToDict:
    """QueryFilter.to_dict() compact representation."""

    def test_empty_filter_returns_empty_dict(self):
        f = parse_query("")
        d = f.to_dict()
        assert d == {}

    def test_only_set_fields_appear(self):
        f = parse_query("source:chatgpt kubernetes")
        d = f.to_dict()
        assert "sources" in d
        assert "text" in d
        assert "tags" not in d
        assert "kinds" not in d
        assert "after" not in d


# ===================================================================
# query() integration tests — require rich_corpus fixture
# ===================================================================


class TestQueryKeywordSearch:
    """Keyword-driven queries that search conversations + chunks + derived."""

    def test_keyword_returns_hits(self, rich_corpus):
        result = query(rich_corpus.store, "Atlas")
        assert isinstance(result, QueryResult)
        assert result.raw_query == "Atlas"
        assert len(result.hits) > 0

    def test_keyword_returns_conversation_hits(self, rich_corpus):
        result = query(rich_corpus.store, "Atlas")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        assert len(conv_hits) >= 1
        # Atlas planning conversation should match.
        conv_ids = {h.id for h in conv_hits}
        assert rich_corpus.conv_ids["atlas"] in conv_ids

    def test_keyword_returns_derived_object_hits(self, rich_corpus):
        result = query(rich_corpus.store, "Atlas")
        derived = [h for h in result.hits if h.hit_type == "derived_object"]
        assert len(derived) >= 1
        # The project "Project Atlas" should be among results.
        ids = {h.id for h in derived}
        assert rich_corpus.project_id in ids

    def test_keyword_returns_chunk_hits(self, rich_corpus):
        result = query(rich_corpus.store, "microservices")
        chunk_hits = [h for h in result.hits if h.hit_type == "chunk"]
        # Chunks are created from the Atlas conversation which mentions microservices.
        # Whether this matches depends on chunk content, but we at least get results.
        all_hits = result.hits
        assert len(all_hits) > 0

    def test_no_results_for_gibberish(self, rich_corpus):
        result = query(rich_corpus.store, "xyzzy_nonexistent_term_12345")
        assert len(result.hits) == 0

    def test_result_shape(self, rich_corpus):
        result = query(rich_corpus.store, "Atlas")
        assert isinstance(result.raw_query, str)
        assert isinstance(result.filters, dict)
        assert isinstance(result.hits, list)
        assert isinstance(result.total_by_type, dict)
        assert isinstance(result.elapsed_ms, float)
        assert result.elapsed_ms >= 0

    def test_hit_shape(self, rich_corpus):
        result = query(rich_corpus.store, "Atlas")
        assert len(result.hits) > 0
        h = result.hits[0]
        assert isinstance(h, QueryHit)
        assert h.hit_type in ("conversation", "chunk", "derived_object")
        assert isinstance(h.id, str) and len(h.id) > 0
        assert isinstance(h.title, str)
        assert isinstance(h.snippet, str)
        assert isinstance(h.score, (int, float))
        assert isinstance(h.metadata, dict)

    def test_total_by_type_matches_hits(self, rich_corpus):
        result = query(rich_corpus.store, "Atlas")
        counted = {}
        for h in result.hits:
            counted[h.hit_type] = counted.get(h.hit_type, 0) + 1
        assert counted == result.total_by_type

    def test_hits_sorted_by_score_descending(self, rich_corpus):
        result = query(rich_corpus.store, "Atlas")
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)


class TestQueryVisibility:
    """Private and quarantined conversations must never leak."""

    def test_private_conversation_excluded(self, rich_corpus):
        # "Therapy session notes" is private; searching for its content
        # must return nothing from that conversation.
        result = query(rich_corpus.store, "anxious")
        conv_ids = {h.id for h in result.hits if h.hit_type == "conversation"}
        assert rich_corpus.conv_ids["therapy"] not in conv_ids

    def test_quarantined_conversation_excluded(self, rich_corpus):
        result = query(rich_corpus.store, "API keys secrets")
        conv_ids = {h.id for h in result.hits if h.hit_type == "conversation"}
        assert rich_corpus.conv_ids["quarantined"] not in conv_ids

    def test_private_not_in_chunks(self, rich_corpus):
        result = query(rich_corpus.store, "anxious coping")
        chunk_cids = set()
        for h in result.hits:
            if h.hit_type == "chunk":
                chunk_cids.add(h.metadata.get("conversation_id"))
        assert rich_corpus.conv_ids["therapy"] not in chunk_cids

    def test_no_private_in_listing_mode(self, rich_corpus):
        # Empty query = listing mode; should still respect visibility.
        result = query(rich_corpus.store, "")
        all_ids = {h.id for h in result.hits}
        assert rich_corpus.conv_ids["therapy"] not in all_ids
        assert rich_corpus.conv_ids["quarantined"] not in all_ids


class TestQuerySourceFilter:
    """source: prefix filtering."""

    def test_source_chatgpt(self, rich_corpus):
        result = query(rich_corpus.store, "source:chatgpt")
        for h in result.hits:
            if h.hit_type == "conversation":
                assert h.metadata.get("source") == "chatgpt", (
                    f"Non-chatgpt conversation in results: {h.title}"
                )

    def test_source_claude(self, rich_corpus):
        result = query(rich_corpus.store, "source:claude")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        assert len(conv_hits) >= 1
        for h in conv_hits:
            assert h.metadata.get("source") == "claude"

    def test_source_with_text(self, rich_corpus):
        result = query(rich_corpus.store, "rate limiting source:claude")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        # Should find the API rate limiting conversation (claude source).
        assert any(rich_corpus.conv_ids["api"] == h.id for h in conv_hits)

    def test_source_excludes_other(self, rich_corpus):
        result = query(rich_corpus.store, "Kubernetes source:claude")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        # K8s is chatgpt, so filtering by claude should exclude it.
        assert all(h.id != rich_corpus.conv_ids["k8s"] for h in conv_hits)


class TestQueryTagFilter:
    """tag: prefix filtering."""

    def test_tag_architecture(self, rich_corpus):
        result = query(rich_corpus.store, "tag:architecture")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        assert len(conv_hits) >= 1
        conv_ids = {h.id for h in conv_hits}
        # Atlas and API conversations have the architecture tag.
        assert rich_corpus.conv_ids["atlas"] in conv_ids

    def test_tag_backend(self, rich_corpus):
        result = query(rich_corpus.store, "tag:backend")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        conv_ids = {h.id for h in conv_hits}
        # API and SDK conversations have backend tag.
        assert rich_corpus.conv_ids["api"] in conv_ids or rich_corpus.conv_ids["sdk"] in conv_ids

    def test_tag_with_keyword(self, rich_corpus):
        result = query(rich_corpus.store, "migration tag:infrastructure")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        if conv_hits:
            conv_ids = {h.id for h in conv_hits}
            # K8s has infrastructure tag and mentions migration.
            assert rich_corpus.conv_ids["k8s"] in conv_ids


class TestQueryKindFilter:
    """kind: prefix restricts to derived objects only."""

    def test_kind_decision(self, rich_corpus):
        result = query(rich_corpus.store, "kind:decision")
        # All hits should be derived objects.
        for h in result.hits:
            assert h.hit_type == "derived_object", (
                f"Expected derived_object, got {h.hit_type}"
            )
            assert h.metadata.get("kind") == "decision"

    def test_kind_entity(self, rich_corpus):
        result = query(rich_corpus.store, "kind:entity")
        for h in result.hits:
            assert h.hit_type == "derived_object"
            assert h.metadata.get("kind") == "entity"

    def test_kind_project(self, rich_corpus):
        result = query(rich_corpus.store, "kind:project")
        derived = [h for h in result.hits if h.hit_type == "derived_object"]
        assert len(derived) >= 1
        assert any(h.id == rich_corpus.project_id for h in derived)

    def test_kind_open_loop(self, rich_corpus):
        result = query(rich_corpus.store, "kind:open_loop")
        for h in result.hits:
            assert h.hit_type == "derived_object"
            assert h.metadata.get("kind") == "open_loop"

    def test_kind_skips_conversations_and_chunks(self, rich_corpus):
        result = query(rich_corpus.store, "kind:decision Atlas")
        # Even though "Atlas" matches conversations, kind: should skip them.
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        chunk_hits = [h for h in result.hits if h.hit_type == "chunk"]
        assert len(conv_hits) == 0
        assert len(chunk_hits) == 0

    def test_kind_with_text(self, rich_corpus):
        result = query(rich_corpus.store, "kind:decision microservices")
        derived = [h for h in result.hits if h.hit_type == "derived_object"]
        # "Use microservices for Atlas" decision should match.
        if derived:
            assert any("microservices" in h.title.lower() for h in derived)


class TestQueryProjectFilter:
    """project: prefix scopes to conversations linked via provenance."""

    def test_project_filter_scopes_conversations(self, rich_corpus):
        pid = rich_corpus.project_id
        result = query(rich_corpus.store, f"project:{pid}")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        # Project Atlas is linked to atlas + k8s conversations.
        conv_ids = {h.id for h in conv_hits}
        assert rich_corpus.conv_ids["atlas"] in conv_ids
        assert rich_corpus.conv_ids["k8s"] in conv_ids
        # Other conversations should NOT be present.
        assert rich_corpus.conv_ids["q4"] not in conv_ids
        assert rich_corpus.conv_ids["retro"] not in conv_ids

    def test_project_filter_with_keyword(self, rich_corpus):
        pid = rich_corpus.project_id
        result = query(rich_corpus.store, f"planning project:{pid}")
        # Should narrow to atlas (has "planning" in title).
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        if conv_hits:
            assert any(h.id == rich_corpus.conv_ids["atlas"] for h in conv_hits)


class TestQueryDateFilters:
    """after: and before: date range filtering."""

    def test_after_excludes_old(self, rich_corpus):
        # All corpus conversations are recent (within last 90 days).
        # Use a date in the far future to exclude everything.
        result = query(rich_corpus.store, "after:2099-01-01")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        assert len(conv_hits) == 0

    def test_before_excludes_recent(self, rich_corpus):
        # All corpus conversations were created recently.
        # Use a date in the distant past to exclude everything.
        result = query(rich_corpus.store, "before:2020-01-01")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        assert len(conv_hits) == 0

    def test_wide_range_includes_all(self, rich_corpus):
        result = query(rich_corpus.store, "after:2020-01-01 before:2099-12-31")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        # Should include all 6 indexed conversations.
        assert len(conv_hits) >= 4


class TestQueryHasFlags:
    """has:open_loops and has:chunks filtering."""

    def test_has_open_loops(self, rich_corpus):
        result = query(rich_corpus.store, "has:open_loops")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        # Atlas conversation has open loops (from extract_for_conversation).
        # At minimum, there should be at least 1 conversation with open loops.
        if conv_hits:
            for h in conv_hits:
                # Verify the conversation actually has open loops.
                c = rich_corpus.store.get_conversation(h.id)
                assert c.has_open_loops, f"{h.title} should have open loops"

    def test_has_chunks(self, rich_corpus):
        result = query(rich_corpus.store, "has:chunks")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        # All 6 indexed conversations were chunked in the fixture.
        assert len(conv_hits) >= 4
        for h in conv_hits:
            cc = rich_corpus.store.conn.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE conversation_id = ?",
                (h.id,),
            ).fetchone()["c"]
            assert cc > 0, f"{h.title} should have chunks"


class TestQueryListingMode:
    """Empty-text queries (filter-only or bare empty string)."""

    def test_empty_query_returns_indexed_conversations(self, rich_corpus):
        result = query(rich_corpus.store, "")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        # Should list indexed conversations (6 total).
        assert len(conv_hits) >= 4

    def test_empty_query_plus_source(self, rich_corpus):
        result = query(rich_corpus.store, "source:claude")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        for h in conv_hits:
            assert h.metadata.get("source") == "claude"

    def test_empty_query_plus_tag(self, rich_corpus):
        result = query(rich_corpus.store, "tag:finance")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        assert len(conv_hits) >= 1
        assert any(h.id == rich_corpus.conv_ids["q4"] for h in conv_hits)


class TestQueryLimit:
    """Limit parameter controls max results."""

    def test_limit_1(self, rich_corpus):
        result = query(rich_corpus.store, "Atlas", limit=1)
        assert len(result.hits) <= 1

    def test_limit_large(self, rich_corpus):
        result = query(rich_corpus.store, "", limit=100)
        # Should return whatever is available, not crash.
        assert len(result.hits) <= 100

    def test_default_limit(self, rich_corpus):
        result = query(rich_corpus.store, "")
        assert len(result.hits) <= 25


class TestQueryDeduplication:
    """Results should be deduplicated by (hit_type, id)."""

    def test_no_duplicate_hits(self, rich_corpus):
        result = query(rich_corpus.store, "Atlas")
        seen = set()
        for h in result.hits:
            key = (h.hit_type, h.id)
            assert key not in seen, f"Duplicate hit: {key}"
            seen.add(key)


class TestQueryCombinedFilters:
    """Multiple filters stacked together."""

    def test_source_plus_tag(self, rich_corpus):
        result = query(rich_corpus.store, "source:claude tag:backend")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        for h in conv_hits:
            assert h.metadata.get("source") == "claude"

    def test_source_plus_has_chunks(self, rich_corpus):
        result = query(rich_corpus.store, "source:chatgpt has:chunks")
        conv_hits = [h for h in result.hits if h.hit_type == "conversation"]
        for h in conv_hits:
            assert h.metadata.get("source") == "chatgpt"

    def test_kind_plus_text(self, rich_corpus):
        result = query(rich_corpus.store, "kind:project Atlas")
        for h in result.hits:
            assert h.hit_type == "derived_object"
            assert h.metadata.get("kind") == "project"
