-- ThreadAtlas SQLite schema (v1).
--
-- Design notes:
-- * The vault's normalized JSON files are the source of truth for raw
--   message text. The DB exists for indexing, search, derived objects, and
--   provenance.
-- * Visibility state is stored on the conversation row. Messages and chunks
--   inherit. We keep a denormalized 'visibility_state_inherited' on messages
--   so FTS can filter without a join.
-- * FTS5 contentless tables would force manual delete tracking; we use
--   "external content" tables so deletes propagate via triggers.

PRAGMA foreign_keys = ON;
PRAGMA secure_delete = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_conversation_id TEXT,
    source_export_fingerprint TEXT,
    title TEXT NOT NULL,
    created_at REAL,
    updated_at REAL,
    imported_at REAL NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending_review',
    message_count INTEGER NOT NULL DEFAULT 0,
    summary_short TEXT NOT NULL DEFAULT '',
    summary_long TEXT,
    manual_tags TEXT NOT NULL DEFAULT '[]',
    auto_tags TEXT NOT NULL DEFAULT '[]',
    primary_project_id TEXT,
    importance_score REAL NOT NULL DEFAULT 0.0,
    resurfacing_score REAL NOT NULL DEFAULT 0.0,
    has_open_loops INTEGER NOT NULL DEFAULT 0,
    schema_version INTEGER NOT NULL DEFAULT 1,
    parser_version INTEGER NOT NULL DEFAULT 1,
    notes_local TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_conv_state ON conversations(state);
CREATE INDEX IF NOT EXISTS idx_conv_source ON conversations(source);
CREATE INDEX IF NOT EXISTS idx_conv_project ON conversations(primary_project_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_fingerprint
    ON conversations(source_export_fingerprint)
    WHERE source_export_fingerprint IS NOT NULL;

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL,
    timestamp REAL,
    content_text TEXT NOT NULL,
    source_message_id TEXT,
    visibility_state_inherited TEXT NOT NULL DEFAULT 'pending_review'
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_msg_state ON messages(visibility_state_inherited);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    start_message_ordinal INTEGER NOT NULL,
    end_message_ordinal INTEGER NOT NULL,
    chunk_title TEXT NOT NULL DEFAULT '',
    summary_short TEXT NOT NULL DEFAULT '',
    project_id TEXT,
    importance_score REAL NOT NULL DEFAULT 0.0,
    has_open_loops INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunk_conv ON chunks(conversation_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunk_project ON chunks(project_id);

-- Derived objects: projects, entities, decisions, open_loops, artifacts, preferences.
CREATE TABLE IF NOT EXISTS derived_objects (
    object_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    project_id TEXT,
    state TEXT NOT NULL DEFAULT 'active', -- active | suppressed
    canonical_key TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_obj_kind ON derived_objects(kind, state);
CREATE INDEX IF NOT EXISTS idx_obj_project ON derived_objects(project_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_obj_canonical
    ON derived_objects(kind, canonical_key)
    WHERE canonical_key != '';

CREATE TABLE IF NOT EXISTS provenance_links (
    link_id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL REFERENCES derived_objects(object_id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    chunk_id TEXT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    excerpt TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prov_obj ON provenance_links(object_id);
CREATE INDEX IF NOT EXISTS idx_prov_conv ON provenance_links(conversation_id);

CREATE TABLE IF NOT EXISTS exports_log (
    export_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at REAL NOT NULL,
    row_counts TEXT NOT NULL DEFAULT '{}'
);

-- FTS5 indexes ----------------------------------------------------------------
--
-- We use *contentful* FTS5 tables (no ``content=''``) so DELETE works
-- naturally during reindex and hard delete. The duplication cost is
-- acceptable for a single-user vault and trades a small amount of disk for a
-- huge amount of correctness when wiping rows.

CREATE VIRTUAL TABLE IF NOT EXISTS fts_conversations USING fts5(
    title,
    summary_short,
    summary_long,
    tags,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    chunk_title,
    summary_short,
    body,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_messages USING fts5(
    body,
    role,
    tokenize='unicode61 remove_diacritics 2'
);
