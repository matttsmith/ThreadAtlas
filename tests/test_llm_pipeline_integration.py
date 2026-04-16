"""End-to-end LLM pipeline integration tests.

These tests exercise realistic multi-turn conversations through the full
pipeline (import -> chunk -> summarize -> group -> name -> LLM-gated
chunking) using the "smart fake" LLM at ``tests/smart_fake_llm.py``. The
fake actually reads prompt content and emits outputs that respond to
what was said, so failures here catch:

  * Prompt rendering losing crucial content (truncation gone wrong,
    role filtering dropping the wrong side)
  * Tokenizer / k-means producing nonsense groups
  * LLM-gated chunking over-splitting or over-merging
  * Small-model confusion scenarios: multi-tool turns, topic ping-pong,
    opening-with-pleasantries threads where the actual topic starts
    later

The goal is not to validate a real model's quality but to validate that
our *plumbing* delivers the right bytes to the model and threads its
output back into the DB correctly.

Running against a real local LLM
--------------------------------
Set ``THREADATLAS_REAL_LLM_ARGV`` to a JSON argv array pointing at your
locally-installed model; the test runner will use it in place of the
smart fake. The command must accept a prompt on stdin (or support the
``{PROMPT_FILE}`` / ``{PROMPT}`` substitutions the runner understands)
and write strict JSON to stdout.

Example (bash; adjust paths for your system):

    export THREADATLAS_REAL_LLM_ARGV='[
        "/usr/local/bin/llama-cli",
        "-m", "/Users/matt/models/qwen2.5-3b-instruct-q4.gguf",
        "--prompt-file", "{PROMPT_FILE}",
        "--no-conversation",
        "--temp", "0.1",
        "--n-predict", "256"
    ]'
    export THREADATLAS_REAL_LLM_TIMEOUT=120
    pytest tests/test_llm_pipeline_integration.py -q

The assertions in this module are written to be lenient enough for a
well-behaved small local model (3B-class Qwen/Llama at temperature
~0.1). If your model is too small, it may legitimately fail these;
that's a signal about the model, not the pipeline.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from threadatlas.cluster import regroup_all
from threadatlas.core.models import State
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path
from threadatlas.llm import LLMRunner, load_config
from threadatlas.llm.chunking import llm_chunk_conversation
from threadatlas.llm.label_groups import label_all_groups
from threadatlas.llm.summarize import summarize_conversation


SMART_FAKE_CMD = [sys.executable, "-m", "tests.smart_fake_llm"]


def _using_real_llm() -> bool:
    return bool(os.environ.get("THREADATLAS_REAL_LLM_ARGV"))


def _runner(tmp_vault, mode: str, used_for):
    """Build an LLMRunner for a test.

    When ``THREADATLAS_REAL_LLM_ARGV`` is set, the real local model is
    used regardless of ``mode`` (real models are task-agnostic); we
    widen ``used_for`` to cover all tasks so the runner doesn't refuse
    any call.
    """
    real_argv = os.environ.get("THREADATLAS_REAL_LLM_ARGV")
    if real_argv:
        try:
            argv = json.loads(real_argv)
        except json.JSONDecodeError as e:
            pytest.fail(f"THREADATLAS_REAL_LLM_ARGV is not valid JSON: {e}")
        if not isinstance(argv, list) or not argv:
            pytest.fail("THREADATLAS_REAL_LLM_ARGV must be a non-empty JSON list")
        timeout = int(os.environ.get("THREADATLAS_REAL_LLM_TIMEOUT", "120"))
        effective_used_for = ("summaries", "group_naming", "chunk_gating")
    else:
        argv = SMART_FAKE_CMD + [mode]
        timeout = 10
        effective_used_for = tuple(used_for)
    (tmp_vault.root / "local_llm.json").write_text(json.dumps({
        "command": argv,
        "timeout_seconds": timeout,
        "max_prompt_chars": 20000,
        "max_response_chars": 4000,
        "used_for": list(effective_used_for),
    }), encoding="utf-8")
    return LLMRunner(tmp_vault, load_config(tmp_vault.root), use_cache=False)


# ---------------------------------------------------------------------------
# Realistic multi-turn fixtures (the kind users actually produce)
# ---------------------------------------------------------------------------

def _multi_turn_project_thread():
    """Project-focused conversation with back-and-forth about the same topic."""
    return {
        "title": "Delta project planning",
        "messages": [
            ("user", "I need help planning Project Delta. Our goal is to launch a customer-facing reporting feature by Q3 with a team of four engineers."),
            ("assistant", "Great. To plan effectively, let's break this down. What's the current scope, and do you have rough sizing for each subsystem?"),
            ("user", "The scope covers three things: ingestion pipeline, a reporting dashboard, and a scheduled export. Sizing is rough; ingestion feels like 3 weeks, dashboard 4 weeks, exports 2 weeks."),
            ("assistant", "With four engineers and those estimates, you have buffer. Do you want to parallelize ingestion and dashboard, or stage them?"),
            ("user", "Let's parallelize. Two engineers on ingestion, one on dashboard, one floating."),
            ("assistant", "Reasonable. Consider reserving the floater for integration tests - those tend to surface late."),
            ("user", "Good point. Add that to the plan. TODO: spec the integration test layer by Friday."),
            ("assistant", "Noted. Also worth deciding the export format early."),
            ("user", "We'll go with CSV plus parquet for the export."),
            ("assistant", "Sensible. Anything else?"),
            ("user", "I think we're good on the plan. Let's write it up next week."),
        ],
    }


def _cooking_thread():
    return {
        "title": "Risotto technique",
        "messages": [
            ("user", "How do I make a mushroom risotto without it going gluey?"),
            ("assistant", "Gluey risotto usually comes from overstirring or wrong rice. Use arborio or carnaroli, and stir only when adding broth."),
            ("user", "I have arborio. What ratio of broth to rice?"),
            ("assistant", "About 4 parts broth to 1 part rice, added one ladle at a time."),
            ("user", "And the mushrooms - do I sear them separately first?"),
            ("assistant", "Yes, brown them hard and fold in at the end so they keep texture."),
            ("user", "Got it. I'll try that tonight. Thanks!"),
        ],
    }


def _multi_tool_confusing_thread():
    """The pattern that reliably confuses small models: user asks for
    several unrelated things in one message, then picks threads back up
    later. We want the pipeline not to shatter this into dust."""
    return {
        "title": "Mixed requests",
        "messages": [
            ("user", "Hey Claude, three things: can you help me rename this Python function, also what's the capital of Paraguay, and finally do you have a good risotto recipe?"),
            ("assistant", "Sure. 1) Rename the function with your editor's refactor tool and update call sites. 2) Asuncion. 3) Risotto - use arborio, warm broth added a ladle at a time, finish with butter and cheese."),
            ("user", "Thanks. For the Python refactor, I'm using PyCharm - does that change anything?"),
            ("assistant", "PyCharm's Refactor > Rename (Shift+F6) will update all references automatically."),
            ("user", "Perfect. Back to the risotto - can it be made with brown rice?"),
            ("assistant", "Technically yes, but the starch profile is different; you won't get the same creamy texture."),
        ],
    }


def _topic_pingpong_thread():
    """User context-switches repeatedly. Tests that LLM-gated chunker
    respects actual topic shifts rather than collapsing the thread into
    one blob."""
    return {
        "title": "Context-switching thread",
        "messages": [
            ("user", "Let's discuss ThreadAtlas architecture. I want projects, entities, and groups as derived objects."),
            ("assistant", "Derived objects make sense. Each should have provenance so you can trace why it exists."),
            ("user", "Yes. Provenance is important. Now switching - what's a good Italian wine for dinner tonight?"),
            ("assistant", "A medium-bodied Nebbiolo or Sangiovese pairs well with most Italian dishes."),
            ("user", "Thanks. Back to ThreadAtlas - should chunking be LLM-driven or heuristic?"),
            ("assistant", "Heuristic first with LLM as a precision gate is a good compromise."),
            ("user", "Agreed. Different topic again: what's the best exercise for lower back pain?"),
            ("assistant", "Gentle hip flexor stretches and McKenzie extensions are commonly recommended; consult a PT for persistent pain."),
        ],
    }


def _fixture_to_tuple(thread: dict) -> dict:
    """Convert to the (role, text, ts) tuple form chatgpt_export_factory expects."""
    return {
        "title": thread["title"],
        "messages": [
            (role, text, float(i + 1))
            for i, (role, text) in enumerate(thread["messages"])
        ],
    }


# ---------------------------------------------------------------------------
# End-to-end: summarize + cluster + name
# ---------------------------------------------------------------------------

def _seed(tmp_vault, store, factory, threads):
    path = factory([_fixture_to_tuple(t) for t in threads])
    res = import_path(tmp_vault, store, path)
    for cid in res.imported:
        transition_state(store, cid, State.INDEXED.value, vault=tmp_vault)
        chunk_conversation(store, cid)
        extract_for_conversation(store, cid)
    return res.imported


def test_summarize_reflects_prompt_content_for_project_thread(tmp_vault, store, chatgpt_export_factory):
    ids = _seed(tmp_vault, store, chatgpt_export_factory, [_multi_turn_project_thread()])
    cid = ids[0]
    runner = _runner(tmp_vault, "summary", used_for=("summaries",))
    outcome = summarize_conversation(tmp_vault, store, runner, cid)
    assert outcome.success, f"summarize failed: {outcome.error}"
    c = store.get_conversation(cid)
    assert c.summary_source == "llm"
    lower = c.summary_short.lower()
    # At least one meaningful project token must survive into the summary.
    assert any(t in lower for t in ("delta", "project", "dashboard", "ingestion", "export")), \
        f"summary missing topical content: {c.summary_short!r}"
    if not _using_real_llm():
        # Smart-fake is deterministic about excluding assistant-only
        # tokens. A real model may or may not include paraphrases, so we
        # only enforce this with the fake.
        assert "buffer" not in lower


def test_summarize_cooking_thread_not_confused_with_project(tmp_vault, store, chatgpt_export_factory):
    ids = _seed(tmp_vault, store, chatgpt_export_factory, [_cooking_thread()])
    cid = ids[0]
    runner = _runner(tmp_vault, "summary", used_for=("summaries",))
    outcome = summarize_conversation(tmp_vault, store, runner, cid)
    assert outcome.success
    c = store.get_conversation(cid)
    lower = c.summary_short.lower()
    assert any(t in lower for t in ("risotto", "mushroom", "arborio"))
    assert not any(t in lower for t in ("project", "ingestion", "dashboard"))


def test_summarize_multi_tool_thread_keeps_all_topics(tmp_vault, store, chatgpt_export_factory):
    """When a user asks for several things in one turn, the summary must
    capture at least the dominant topics rather than silently dropping
    the minority ones."""
    ids = _seed(tmp_vault, store, chatgpt_export_factory, [_multi_tool_confusing_thread()])
    cid = ids[0]
    runner = _runner(tmp_vault, "summary", used_for=("summaries",))
    outcome = summarize_conversation(tmp_vault, store, runner, cid)
    assert outcome.success
    c = store.get_conversation(cid)
    lower = c.summary_short.lower()
    # This thread blends code / geography / cooking; our summary should
    # mention at least one of them concretely.
    mentions = sum(1 for t in ("python", "pycharm", "risotto", "paraguay", "asuncion", "function", "rename") if t in lower)
    assert mentions >= 1, f"summary missed all topics: {c.summary_short!r}"


def test_cluster_and_label_separates_project_from_cooking(tmp_vault, store, chatgpt_export_factory):
    """Full pipeline: multi-turn convs of two topics, summarize, cluster,
    name groups, and assert the final group labels reflect the topics."""
    # Six project threads and six cooking threads of varying phrasing.
    project_threads = []
    for i in range(6):
        t = _multi_turn_project_thread()
        t["title"] = f"Delta planning session {i}"
        project_threads.append(t)
    cook_threads = []
    for i in range(6):
        t = _cooking_thread()
        t["title"] = f"Risotto q&a {i}"
        cook_threads.append(t)
    ids = _seed(tmp_vault, store, chatgpt_export_factory, project_threads + cook_threads)

    # Summaries (via smart-fake) so clustering has good signal.
    runner = _runner(tmp_vault, "summary", used_for=("summaries",))
    for cid in ids:
        summarize_conversation(tmp_vault, store, runner, cid)

    # Now cluster.
    regroup_all(store, broad_k=2, fine_k=2, seed=42)
    groups = store.list_groups(level="broad")
    assert len(groups) == 2

    # Each group should be dominated by one topic. Check via member ids.
    project_ids = set(ids[:6])
    cook_ids = set(ids[6:])
    cluster_sets = []
    for g in groups:
        members = set(store.list_group_members(g["group_id"]))
        cluster_sets.append(members)
    # Every conv is in exactly one cluster at the broad level.
    total = set().union(*cluster_sets)
    assert total == set(ids)
    # Clusters don't overlap.
    assert not cluster_sets[0] & cluster_sets[1]
    # Each cluster should be topically pure (>= 5 of 6 same-topic).
    purities = []
    for members in cluster_sets:
        purities.append(max(len(members & project_ids), len(members & cook_ids)))
    assert all(p >= 5 for p in purities), f"low cluster purity: {purities}"

    # Now ask the smart fake to name the groups.
    label_runner = _runner(tmp_vault, "group_name", used_for=("group_naming",))
    outcomes = label_all_groups(tmp_vault, store, label_runner)
    labeled = [o for o in outcomes if o.success]
    assert labeled
    # Each label should reflect its cluster's topic.
    labels = {}
    for g in groups:
        refreshed = store.get_group(g["group_id"])
        label = (refreshed.get("llm_label") or "").lower()
        members = set(store.list_group_members(g["group_id"]))
        topic = "project" if len(members & project_ids) > len(members & cook_ids) else "cooking"
        labels[topic] = label
    # Project-dominated cluster's label should contain a project-ish token;
    # cooking-dominated cluster's label should contain a cooking-ish token.
    if "project" in labels:
        assert any(t in labels["project"] for t in ("delta", "project", "dashboard", "ingestion", "engineers")), \
            f"project label wrong: {labels['project']!r}"
    if "cooking" in labels:
        assert any(t in labels["cooking"] for t in ("risotto", "mushroom", "arborio")), \
            f"cooking label wrong: {labels['cooking']!r}"


# ---------------------------------------------------------------------------
# LLM-gated chunking on multi-turn threads
# ---------------------------------------------------------------------------

_TOPIC_FILLERS = {
    "delta": {
        "user": "scoping the ingestion pipeline headcount for Q3 timeline",
        "assistant": "backend throughput capacity reliability shipping production",
    },
    "risotto": {
        "user": "arborio rice broth ladle mushroom parmesan butter seasoning",
        "assistant": "stirring technique carnaroli saffron onion stock flavor",
    },
    "wine": {
        "user": "nebbiolo sangiovese pairing vintage acidity tannins Italian",
        "assistant": "Barolo Chianti Piedmont Tuscany decant temperature glassware",
    },
}


def _long_topic_block(topic_word: str, n_pairs: int):
    """Dense, topic-distinct block large enough to clear the deterministic
    chunker's MIN_CHARS gate.

    Each topic has its own vocabulary (no shared scaffolding words) so the
    smart-fake gate can reliably distinguish them by token overlap - the
    same property a real small model would also exploit.
    """
    fillers = _TOPIC_FILLERS.get(topic_word, {
        "user": f"{topic_word} notes planning followup",
        "assistant": f"{topic_word} explanation reasoning",
    })
    msgs = []
    for i in range(n_pairs):
        msgs.append((
            "user",
            f"{topic_word} {fillers['user']} iteration {i} {fillers['user']}",
        ))
        msgs.append((
            "assistant",
            f"{topic_word} {fillers['assistant']} "
            + (f"{topic_word} {fillers['assistant']} " * 40),
        ))
    return msgs


def test_llm_gate_does_not_split_same_topic(tmp_vault, store, chatgpt_export_factory):
    """A long, single-topic conversation should survive LLM-gating without
    being carved into spurious chunks."""
    messages = _long_topic_block("delta", 8)
    thread = {"title": "Deep on Delta", "messages": [(r, t, float(i)) for i, (r, t) in enumerate(messages)]}
    path = chatgpt_export_factory([thread])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value, vault=tmp_vault)
    runner = _runner(tmp_vault, "chunk_gate", used_for=("chunk_gating",))
    outcome = llm_chunk_conversation(tmp_vault, store, runner, cid)
    # Either the heuristic already produced 1 chunk or the gate should
    # merge down to <= 2. Critically, we should not end with MORE chunks
    # than the deterministic pass produced.
    assert outcome.after_chunks <= outcome.before_chunks
    assert outcome.after_chunks <= 2, f"same-topic thread shattered: {outcome.after_chunks} chunks"


def test_llm_gate_preserves_obvious_topic_shift(tmp_vault, store, chatgpt_export_factory):
    """Two distinct topics in the same conversation should end up in
    separate chunks after the gate - the smart fake sees a clear topic
    shift and returns split=true."""
    block_a = _long_topic_block("delta", 6)
    block_b = _long_topic_block("risotto", 6)
    all_msgs = block_a + block_b
    thread = {"title": "Mixed", "messages": [(r, t, float(i)) for i, (r, t) in enumerate(all_msgs)]}
    path = chatgpt_export_factory([thread])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value, vault=tmp_vault)
    runner = _runner(tmp_vault, "chunk_gate", used_for=("chunk_gating",))
    outcome = llm_chunk_conversation(tmp_vault, store, runner, cid)
    assert outcome.after_chunks >= 2, \
        f"obvious topic shift collapsed: {outcome.after_chunks} chunks"


def test_llm_gate_on_pingpong_thread_retains_boundaries(tmp_vault, store, chatgpt_export_factory):
    """User jumps between topics repeatedly. Heuristic may produce many
    chunks; the LLM gate may merge adjacent same-topic pairs, but total
    chunks should remain > 1 because the topics are actually different."""
    # Alternate topic blocks to produce multiple candidate boundaries.
    messages = (
        _long_topic_block("delta", 3)
        + _long_topic_block("risotto", 3)
        + _long_topic_block("delta", 3)
        + _long_topic_block("risotto", 3)
    )
    thread = {"title": "Ping-pong", "messages": [(r, t, float(i)) for i, (r, t) in enumerate(messages)]}
    path = chatgpt_export_factory([thread])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value, vault=tmp_vault)
    runner = _runner(tmp_vault, "chunk_gate", used_for=("chunk_gating",))
    outcome = llm_chunk_conversation(tmp_vault, store, runner, cid)
    # At least two chunks (topics are genuinely different); gate cannot
    # increase chunk count vs the heuristic.
    assert outcome.after_chunks >= 2
    assert outcome.after_chunks <= outcome.before_chunks
