"""Tests for prompt loading, model enums, and schema changes.

Tests cover:
- Prompt file loading and version extraction
- Prompt template rendering
- New model enums (Register, RealityMode, EntityType)
- New dataclass fields
- Schema migration (new tables and columns exist)
- Content hash for incremental indexing
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from threadatlas.core.models import (
    ALL_REGISTERS,
    DEFAULT_REGISTER_EXCLUDES,
    ConversationLLMMeta,
    DerivedObject,
    EntityType,
    MessageClassification,
    RealityMode,
    Register,
)
from threadatlas.core.vault import init_vault
from threadatlas.llm.config import ALLOWED_USED_FOR
from threadatlas.llm.prompt_loader import (
    EXTRACTION_PROMPT,
    PROFILE_PROMPT,
    TURN_CLASSIFIER_PROMPT,
    get_prompt_version,
    render_prompt,
)
from threadatlas.store import open_store


# --- Prompt loading ---

class TestPromptLoader:

    def test_load_turn_classifier(self):
        version = get_prompt_version(TURN_CLASSIFIER_PROMPT)
        assert version == "turn_classifier_v1"

    def test_load_extraction(self):
        version = get_prompt_version(EXTRACTION_PROMPT)
        assert version == "extraction_v1"

    def test_load_profile(self):
        version = get_prompt_version(PROFILE_PROMPT)
        assert version == "profile_v1"

    def test_render_turn_classifier(self):
        result = render_prompt(
            TURN_CLASSIFIER_PROMPT,
            TITLE="Test Conversation",
            MESSAGES='{"role": "user", "text": "hello"}',
        )
        assert "Test Conversation" in result
        assert "hello" in result
        assert "<<TITLE>>" not in result
        assert "<<MESSAGES>>" not in result

    def test_render_extraction(self):
        result = render_prompt(
            EXTRACTION_PROMPT,
            TITLE="Test",
            CREATED="2024-01-01",
            UPDATED="2024-06-01",
            MESSAGES_WITH_TAGS="[user | register=work] hello",
        )
        assert "2024-01-01" in result
        assert "register=work" in result

    def test_render_profile(self):
        result = render_prompt(
            PROFILE_PROMPT,
            FOCUS_INSTRUCTION="Focus on architecture.",
            PROJECTS="- Project Atlas",
            DECISIONS="- Use Kubernetes",
            OPEN_LOOPS="- CI/CD pending",
            ENTITIES="- Kubernetes (concept)",
            SUMMARIES="- Architecture planning",
            REGISTER_DIST='{"work": 5}',
        )
        assert "Focus on architecture" in result
        assert "Project Atlas" in result

    def test_missing_prompt_file(self):
        with pytest.raises(FileNotFoundError):
            render_prompt("nonexistent_prompt.txt")


# --- Model enums ---

class TestModelEnums:

    def test_register_values(self):
        assert Register.WORK.value == "work"
        assert Register.ROLEPLAY.value == "roleplay"
        assert Register.JAILBREAK_EXPERIMENT.value == "jailbreak_experiment"
        assert len(Register) == 9

    def test_reality_mode_values(self):
        assert RealityMode.LITERAL.value == "literal"
        assert RealityMode.FICTIONAL.value == "fictional"
        assert RealityMode.HYPOTHETICAL.value == "hypothetical"
        assert len(RealityMode) == 3

    def test_entity_type_values(self):
        assert EntityType.PERSON.value == "person"
        assert EntityType.GLITCH_TOKEN.value == "glitch_token"
        assert EntityType.FICTIONAL_CHARACTER.value == "fictional_character"
        assert len(EntityType) == 8

    def test_default_register_excludes(self):
        assert "roleplay" in DEFAULT_REGISTER_EXCLUDES
        assert "jailbreak_experiment" in DEFAULT_REGISTER_EXCLUDES
        assert "work" not in DEFAULT_REGISTER_EXCLUDES

    def test_all_registers_complete(self):
        assert len(ALL_REGISTERS) == len(Register)
        for r in Register:
            assert r.value in ALL_REGISTERS


# --- Dataclass fields ---

class TestDataclasses:

    def test_derived_object_new_fields(self):
        obj = DerivedObject(
            object_id="test-1",
            kind="entity",
            title="Kubernetes",
            entity_type="concept",
            source_register="work",
            source_reality_mode="literal",
            paraphrase="Container orchestration platform",
            first_seen=time.time() - 86400,
            last_seen=time.time(),
            status="active",
        )
        assert obj.entity_type == "concept"
        assert obj.source_register == "work"
        assert obj.paraphrase == "Container orchestration platform"

    def test_derived_object_new_fields_default_none(self):
        obj = DerivedObject(object_id="test-2", kind="project", title="Test")
        assert obj.entity_type is None
        assert obj.source_register is None
        assert obj.paraphrase is None

    def test_message_classification(self):
        cls = MessageClassification(
            message_id="msg-1",
            register="work",
            reality_mode="literal",
            prompt_version="turn_classifier_v1",
            classified_at=time.time(),
        )
        assert cls.register == "work"
        assert cls.reality_mode == "literal"

    def test_conversation_llm_meta(self):
        meta = ConversationLLMMeta(
            conversation_id="conv-1",
            llm_summary="Test summary",
            dominant_register="work",
            content_hash="abc123",
        )
        assert meta.llm_summary == "Test summary"
        assert meta.dominant_register == "work"


# --- LLM config ---

class TestLLMConfig:

    def test_new_task_types_allowed(self):
        assert "turn_classification" in ALLOWED_USED_FOR
        assert "extraction" in ALLOWED_USED_FOR
        assert "profile" in ALLOWED_USED_FOR
        # Old tasks still allowed.
        assert "summaries" in ALLOWED_USED_FOR
        assert "group_naming" in ALLOWED_USED_FOR


# --- Schema migration ---

class TestSchemaMigration:

    def test_new_tables_exist(self, tmp_path):
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)

        # message_classifications table.
        store.conn.execute("SELECT * FROM message_classifications LIMIT 0")
        # conversation_llm_meta table.
        store.conn.execute("SELECT * FROM conversation_llm_meta LIMIT 0")
        # chunk_embeddings table.
        store.conn.execute("SELECT * FROM chunk_embeddings LIMIT 0")
        store.close()

    def test_derived_objects_new_columns(self, tmp_path):
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)

        cols = store.conn.execute("PRAGMA table_info(derived_objects)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "entity_type" in col_names
        assert "source_register" in col_names
        assert "source_reality_mode" in col_names
        assert "paraphrase" in col_names
        assert "first_seen" in col_names
        assert "last_seen" in col_names
        assert "status" in col_names
        store.close()

    def test_message_classification_crud(self, tmp_path):
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)

        # Need a conversation + message first.
        from threadatlas.core.models import Conversation, Message, new_id
        conv_id = new_id("conv")
        now = time.time()
        store.insert_conversation(Conversation(
            conversation_id=conv_id, source="chatgpt", title="Test",
            created_at=now, updated_at=now, imported_at=now,
            state="indexed", message_count=1,
        ))
        msg_id = new_id("msg")
        store.insert_messages([Message(
            message_id=msg_id, conversation_id=conv_id, ordinal=0,
            role="user", content_text="hello", visibility_state_inherited="indexed",
        )])
        store.conn.commit()

        cls = MessageClassification(
            message_id=msg_id, register="work", reality_mode="literal",
            prompt_version="v1", classified_at=now,
        )
        store.upsert_message_classification(cls)
        store.conn.commit()

        results = store.get_message_classifications(conv_id)
        assert len(results) == 1
        assert results[0].register == "work"
        assert results[0].reality_mode == "literal"
        store.close()

    def test_conversation_llm_meta_crud(self, tmp_path):
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)

        from threadatlas.core.models import Conversation, new_id
        conv_id = new_id("conv")
        now = time.time()
        store.insert_conversation(Conversation(
            conversation_id=conv_id, source="chatgpt", title="Test",
            created_at=now, updated_at=now, imported_at=now,
        ))
        store.conn.commit()

        meta = ConversationLLMMeta(
            conversation_id=conv_id,
            llm_summary="Test summary",
            dominant_register="work",
            content_hash="abc123",
            extracted_at=now,
        )
        store.upsert_conversation_llm_meta(meta)
        store.conn.commit()

        result = store.get_conversation_llm_meta(conv_id)
        assert result is not None
        assert result.llm_summary == "Test summary"
        assert result.content_hash == "abc123"
        assert result.dominant_register == "work"
        store.close()

    def test_chunk_embedding_crud(self, tmp_path):
        vault = init_vault(tmp_path / "vault")
        store = open_store(vault)

        from threadatlas.core.models import Chunk, Conversation, new_id
        conv_id = new_id("conv")
        chunk_id = new_id("chk")
        now = time.time()
        store.insert_conversation(Conversation(
            conversation_id=conv_id, source="chatgpt", title="Test",
            created_at=now, updated_at=now, imported_at=now,
            state="indexed",
        ))
        store.replace_chunks(conv_id, [Chunk(
            chunk_id=chunk_id, conversation_id=conv_id, chunk_index=0,
            start_message_ordinal=0, end_message_ordinal=1,
        )])
        store.conn.commit()

        import struct
        embedding = struct.pack("3f", 0.1, 0.2, 0.3)
        store.upsert_chunk_embedding(chunk_id, embedding, "test-model", now)
        store.conn.commit()

        results = store.get_chunk_embeddings_for_conversation(conv_id)
        assert len(results) == 1
        assert results[0][0] == chunk_id
        assert results[0][1] == embedding

        all_embs = store.get_all_chunk_embeddings(visible_states=("indexed",))
        assert len(all_embs) == 1
        store.close()
