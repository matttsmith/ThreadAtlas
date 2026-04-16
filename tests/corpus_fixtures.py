"""Rich test corpus for query engine tests.

Provides a ``rich_corpus`` fixture that seeds a vault with 8 conversations
spanning two sources (ChatGPT + Claude), multiple states, tags, chunks,
and derived objects. The corpus is designed to exercise every filter prefix
the query engine supports.

Corpus layout
-------------
1. "Project Atlas planning"      — indexed, chatgpt, tag:architecture, has open loops
2. "Kubernetes cluster migration" — indexed, chatgpt, tag:infrastructure
3. "API rate limiting discussion" — indexed, claude, tag:architecture tag:backend
4. "Therapy session notes"        — private, chatgpt (should never appear in MCP)
5. "Q4 budget review"             — indexed, chatgpt, tag:finance
6. "Claude SDK integration"       — indexed, claude, tag:backend
7. "Team retrospective Jan"       — indexed, chatgpt, tag:process, after 2025-01-15
8. "Quarantined sensitive data"   — quarantined, chatgpt (should never appear)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from threadatlas.core.models import (
    DerivedKind,
    DerivedObject,
    ProvenanceLink,
    State,
    new_id,
)
from threadatlas.core.vault import Vault, init_vault
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path
from threadatlas.store import open_store, Store

import sys as _sys
from pathlib import Path as _Path
# conftest isn't a regular module; import helpers directly.
_tests_dir = _Path(__file__).parent
if str(_tests_dir) not in _sys.path:
    _sys.path.insert(0, str(_tests_dir))
from conftest import make_chatgpt_export, make_claude_export


@dataclass
class CorpusInfo:
    """Metadata about the seeded corpus, for assertions."""
    vault: Vault
    store: Store
    # conversation_ids by title keyword for easy lookup
    conv_ids: dict[str, str]
    # derived object ids
    project_id: str
    entity_id: str
    decision_id: str
    open_loop_id: str


def _ts(days_ago: float) -> float:
    return time.time() - days_ago * 86400


def _seed_corpus(tmp_path: Path) -> CorpusInfo:
    vault = init_vault(tmp_path / "vault")
    store = open_store(vault)

    # --- ChatGPT conversations ---
    chatgpt_path = make_chatgpt_export(tmp_path, [
        {
            "id": "cg-atlas",
            "title": "Project Atlas planning",
            "create_time": _ts(30),
            "update_time": _ts(5),
            "messages": [
                ("user", "Let's plan Project Atlas. This is an architecture initiative.", _ts(30)),
                ("assistant", "Project Atlas sounds exciting. What's the scope?", _ts(30) + 60),
                ("user", "Decision: we will use microservices for Atlas.", _ts(29)),
                ("assistant", "Good choice. TODO: finalize the deployment strategy.", _ts(29) + 60),
                ("user", "I need to follow up on the Atlas staffing. Open loop.", _ts(28)),
                ("assistant", "Noted. I'll remind you about Project Atlas staffing.", _ts(28) + 60),
            ],
        },
        {
            "id": "cg-k8s",
            "title": "Kubernetes cluster migration",
            "create_time": _ts(20),
            "update_time": _ts(10),
            "messages": [
                ("user", "We need to migrate our Kubernetes cluster to the new region.", _ts(20)),
                ("assistant", "I'll outline the migration steps for the cluster.", _ts(20) + 60),
                ("user", "The migration involves moving 50 pods across three namespaces.", _ts(19)),
                ("assistant", "That's a significant infrastructure change.", _ts(19) + 60),
            ],
        },
        {
            "id": "cg-therapy",
            "title": "Therapy session notes",
            "create_time": _ts(15),
            "update_time": _ts(15),
            "messages": [
                ("user", "I feel anxious about the upcoming deadline.", _ts(15)),
                ("assistant", "It's normal to feel anxious. Let's talk about coping strategies.", _ts(15) + 60),
            ],
        },
        {
            "id": "cg-q4",
            "title": "Q4 budget review",
            "create_time": _ts(60),
            "update_time": _ts(55),
            "messages": [
                ("user", "Let's review the Q4 budget allocations.", _ts(60)),
                ("assistant", "The Q4 budget shows a 15% increase in infrastructure spending.", _ts(60) + 60),
                ("user", "Decision: increase the cloud budget by 20%.", _ts(59)),
                ("assistant", "I'll update the budget spreadsheet.", _ts(59) + 60),
            ],
        },
        {
            "id": "cg-retro",
            "title": "Team retrospective Jan",
            "create_time": _ts(90),
            "update_time": _ts(85),
            "messages": [
                ("user", "Let's do our January team retrospective.", _ts(90)),
                ("assistant", "What went well this sprint?", _ts(90) + 60),
                ("user", "The deployment process improved significantly.", _ts(89)),
                ("assistant", "Great to hear about the deployment improvements.", _ts(89) + 60),
            ],
        },
        {
            "id": "cg-quarantined",
            "title": "Quarantined sensitive data",
            "create_time": _ts(10),
            "update_time": _ts(10),
            "messages": [
                ("user", "Here are the API keys and secrets for the production system.", _ts(10)),
                ("assistant", "I've noted the credentials.", _ts(10) + 60),
            ],
        },
    ])

    # --- Claude conversations ---
    claude_path = make_claude_export(tmp_path, [
        {
            "uuid": "cl-api",
            "name": "API rate limiting discussion",
            "created_at": "2025-03-15T12:00:00Z",
            "updated_at": "2025-03-20T12:00:00Z",
            "messages": [
                ("human", "How should we implement API rate limiting for our backend?", "2025-03-15T12:00:00Z"),
                ("assistant", "For API rate limiting, I recommend a token bucket algorithm.", "2025-03-15T12:01:00Z"),
                ("human", "What about the architecture for distributed rate limiting?", "2025-03-16T12:00:00Z"),
                ("assistant", "Use Redis as a centralized rate limit store for the architecture.", "2025-03-16T12:01:00Z"),
            ],
        },
        {
            "uuid": "cl-sdk",
            "name": "Claude SDK integration",
            "created_at": "2025-04-01T12:00:00Z",
            "updated_at": "2025-04-05T12:00:00Z",
            "messages": [
                ("human", "I want to integrate the Claude SDK into our backend service.", "2025-04-01T12:00:00Z"),
                ("assistant", "The Claude SDK provides a simple Python interface.", "2025-04-01T12:01:00Z"),
                ("human", "How do we handle SDK authentication and error handling?", "2025-04-02T12:00:00Z"),
                ("assistant", "Use API key authentication with exponential backoff for errors.", "2025-04-02T12:01:00Z"),
            ],
        },
    ])

    # Import both sources.
    cg_res = import_path(vault, store, chatgpt_path)
    cl_res = import_path(vault, store, claude_path)

    # Map IDs for lookup. import_path returns IDs in order of the input list.
    cg_ids = cg_res.imported  # atlas, k8s, therapy, q4, retro, quarantined
    cl_ids = cl_res.imported  # api, sdk

    conv_ids = {
        "atlas": cg_ids[0],
        "k8s": cg_ids[1],
        "therapy": cg_ids[2],
        "q4": cg_ids[3],
        "retro": cg_ids[4],
        "quarantined": cg_ids[5],
        "api": cl_ids[0],
        "sdk": cl_ids[1],
    }

    # --- Transition states ---
    for key in ("atlas", "k8s", "q4", "retro"):
        transition_state(store, conv_ids[key], State.INDEXED.value)
    for key in ("api", "sdk"):
        transition_state(store, conv_ids[key], State.INDEXED.value)
    transition_state(store, conv_ids["therapy"], State.PRIVATE.value)
    transition_state(store, conv_ids["quarantined"], State.QUARANTINED.value)

    # --- Chunk + extract indexed conversations ---
    for key in ("atlas", "k8s", "q4", "retro", "api", "sdk"):
        chunk_conversation(store, conv_ids[key])
        extract_for_conversation(store, conv_ids[key])

    # --- Add manual tags ---
    store.add_manual_tags(conv_ids["atlas"], ["architecture"])
    store.add_manual_tags(conv_ids["k8s"], ["infrastructure"])
    store.add_manual_tags(conv_ids["api"], ["architecture", "backend"])
    store.add_manual_tags(conv_ids["q4"], ["finance"])
    store.add_manual_tags(conv_ids["sdk"], ["backend"])
    store.add_manual_tags(conv_ids["retro"], ["process"])

    # --- Seed explicit derived objects with provenance ---
    now = time.time()

    project_id = new_id("proj")
    store.upsert_derived_object(DerivedObject(
        object_id=project_id,
        kind=DerivedKind.PROJECT.value,
        title="Project Atlas",
        description="A multi-quarter architecture initiative to modernize the platform.",
        canonical_key="project atlas",
        created_at=now,
        updated_at=now,
    ))
    store.insert_provenance(ProvenanceLink(
        link_id=new_id("prov"),
        object_id=project_id,
        conversation_id=conv_ids["atlas"],
        chunk_id=None,
        excerpt="Project Atlas planning",
        created_at=now,
    ))

    entity_id = new_id("ent")
    store.upsert_derived_object(DerivedObject(
        object_id=entity_id,
        kind=DerivedKind.ENTITY.value,
        title="Redis",
        description="In-memory data store used for caching and rate limiting.",
        canonical_key="redis",
        created_at=now,
        updated_at=now,
    ))
    store.insert_provenance(ProvenanceLink(
        link_id=new_id("prov"),
        object_id=entity_id,
        conversation_id=conv_ids["api"],
        chunk_id=None,
        excerpt="Redis as a centralized rate limit store",
        created_at=now,
    ))

    decision_id = new_id("dec")
    store.upsert_derived_object(DerivedObject(
        object_id=decision_id,
        kind=DerivedKind.DECISION.value,
        title="Use microservices for Atlas",
        description="Decision to adopt microservices architecture for Project Atlas.",
        canonical_key="use microservices for atlas",
        created_at=now,
        updated_at=now,
    ))
    store.insert_provenance(ProvenanceLink(
        link_id=new_id("prov"),
        object_id=decision_id,
        conversation_id=conv_ids["atlas"],
        chunk_id=None,
        excerpt="Decision: we will use microservices for Atlas.",
        created_at=now,
    ))

    open_loop_id = new_id("loop")
    store.upsert_derived_object(DerivedObject(
        object_id=open_loop_id,
        kind=DerivedKind.OPEN_LOOP.value,
        title="Finalize Atlas staffing",
        description="Need to follow up on staffing for Project Atlas.",
        canonical_key="finalize atlas staffing",
        created_at=now,
        updated_at=now,
    ))
    store.insert_provenance(ProvenanceLink(
        link_id=new_id("prov"),
        object_id=open_loop_id,
        conversation_id=conv_ids["atlas"],
        chunk_id=None,
        excerpt="I need to follow up on the Atlas staffing.",
        created_at=now,
    ))

    # Link project to k8s conversation too (cross-conversation linkage).
    store.insert_provenance(ProvenanceLink(
        link_id=new_id("prov"),
        object_id=project_id,
        conversation_id=conv_ids["k8s"],
        chunk_id=None,
        excerpt="Infrastructure work related to Project Atlas",
        created_at=now,
    ))

    store.conn.commit()

    return CorpusInfo(
        vault=vault,
        store=store,
        conv_ids=conv_ids,
        project_id=project_id,
        entity_id=entity_id,
        decision_id=decision_id,
        open_loop_id=open_loop_id,
    )


@pytest.fixture
def rich_corpus(tmp_path) -> CorpusInfo:
    """Seed a realistic multi-source corpus for query engine testing."""
    info = _seed_corpus(tmp_path)
    yield info
    info.store.close()
