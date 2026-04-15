"""SQLite store wrapper.

Encapsulates schema bootstrap, transactions, and the small set of CRUD
operations that the rest of the codebase needs.

Design notes
------------
* We rebuild FTS rows from authoritative tables rather than maintaining
  triggers, because FTS5 contentless tables make it easy to get out of sync
  during deletes. Reindex is cheap on a single-user corpus.
* All filesystem-aware operations live in higher layers; this module talks
  only to ``sqlite3``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from ..core.models import (
    Chunk,
    Conversation,
    DerivedObject,
    FTS_INDEXED_STATES,
    Message,
    ProvenanceLink,
    State,
    new_id,
)
from ..core.vault import Vault


SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass
class Store:
    conn: sqlite3.Connection

    # --- bootstrap ----------------------------------------------------------

    def bootstrap(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(sql)
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            ("schema_version", "1"),
        )
        self.conn.commit()

    def close(self) -> None:
        # Idempotent: callers may close more than once during teardown.
        try:
            self.conn.commit()
        except sqlite3.ProgrammingError:
            return
        try:
            self.conn.close()
        except sqlite3.ProgrammingError:
            pass

    # --- conversations ------------------------------------------------------

    def insert_conversation(self, c: Conversation) -> None:
        self.conn.execute(
            """
            INSERT INTO conversations (
                conversation_id, source, source_conversation_id,
                source_export_fingerprint, title, created_at, updated_at,
                imported_at, state, message_count, summary_short, summary_long,
                manual_tags, auto_tags, primary_project_id,
                importance_score, resurfacing_score, has_open_loops,
                schema_version, parser_version, notes_local
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                c.conversation_id, c.source, c.source_conversation_id,
                c.source_export_fingerprint, c.title, c.created_at, c.updated_at,
                c.imported_at, c.state, c.message_count, c.summary_short,
                c.summary_long, json.dumps(c.manual_tags), json.dumps(c.auto_tags),
                c.primary_project_id, c.importance_score, c.resurfacing_score,
                1 if c.has_open_loops else 0, c.schema_version, c.parser_version,
                c.notes_local,
            ),
        )

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return _row_to_conversation(row) if row else None

    def find_conversation_by_fingerprint(self, fingerprint: str) -> Conversation | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE source_export_fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return _row_to_conversation(row) if row else None

    def list_conversations(
        self,
        state: str | None = None,
        source: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Conversation]:
        sql = "SELECT * FROM conversations WHERE 1=1"
        params: list = []
        if state is not None:
            sql += " AND state = ?"
            params.append(state)
        if source is not None:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY COALESCE(updated_at, created_at, imported_at) DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_conversation(r) for r in rows]

    def set_conversation_state(self, conversation_id: str, state: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET state = ? WHERE conversation_id = ?",
            (state, conversation_id),
        )
        self.conn.execute(
            "UPDATE messages SET visibility_state_inherited = ? WHERE conversation_id = ?",
            (state, conversation_id),
        )

    def update_conversation_meta(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        notes_local: str | None = None,
        manual_tags: list[str] | None = None,
        auto_tags: list[str] | None = None,
        summary_short: str | None = None,
        summary_long: str | None = None,
        primary_project_id: str | None = None,
        importance_score: float | None = None,
        resurfacing_score: float | None = None,
        has_open_loops: bool | None = None,
    ) -> None:
        sets: list[str] = []
        params: list = []
        if title is not None:
            sets.append("title = ?"); params.append(title)
        if notes_local is not None:
            sets.append("notes_local = ?"); params.append(notes_local)
        if manual_tags is not None:
            sets.append("manual_tags = ?"); params.append(json.dumps(manual_tags))
        if auto_tags is not None:
            sets.append("auto_tags = ?"); params.append(json.dumps(auto_tags))
        if summary_short is not None:
            sets.append("summary_short = ?"); params.append(summary_short)
        if summary_long is not None:
            sets.append("summary_long = ?"); params.append(summary_long)
        if primary_project_id is not None:
            sets.append("primary_project_id = ?"); params.append(primary_project_id)
        if importance_score is not None:
            sets.append("importance_score = ?"); params.append(importance_score)
        if resurfacing_score is not None:
            sets.append("resurfacing_score = ?"); params.append(resurfacing_score)
        if has_open_loops is not None:
            sets.append("has_open_loops = ?"); params.append(1 if has_open_loops else 0)
        if not sets:
            return
        params.append(conversation_id)
        self.conn.execute(
            f"UPDATE conversations SET {', '.join(sets)} WHERE conversation_id = ?",
            params,
        )

    def delete_conversation(self, conversation_id: str) -> None:
        # FK cascades remove messages, chunks, provenance_links.
        # Suppressed-derived objects remain unless caller explicitly
        # recomputes; see deletion module for full cascade.
        self.conn.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )

    # --- messages -----------------------------------------------------------

    def insert_messages(self, messages: Iterable[Message]) -> None:
        rows = [
            (
                m.message_id, m.conversation_id, m.ordinal, m.role,
                m.timestamp, m.content_text, m.source_message_id,
                m.visibility_state_inherited,
            )
            for m in messages
        ]
        self.conn.executemany(
            """
            INSERT INTO messages (
                message_id, conversation_id, ordinal, role, timestamp,
                content_text, source_message_id, visibility_state_inherited
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def list_messages(self, conversation_id: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY ordinal",
            (conversation_id,),
        ).fetchall()
        out = []
        for r in rows:
            out.append(Message(
                message_id=r["message_id"],
                conversation_id=r["conversation_id"],
                ordinal=r["ordinal"],
                role=r["role"],
                timestamp=r["timestamp"],
                content_text=r["content_text"],
                content_structured=None,
                source_message_id=r["source_message_id"],
                visibility_state_inherited=r["visibility_state_inherited"],
            ))
        return out

    # --- chunks -------------------------------------------------------------

    def replace_chunks(self, conversation_id: str, chunks: list[Chunk]) -> None:
        self.conn.execute(
            "DELETE FROM chunks WHERE conversation_id = ?", (conversation_id,)
        )
        rows = [
            (
                c.chunk_id, c.conversation_id, c.chunk_index,
                c.start_message_ordinal, c.end_message_ordinal,
                c.chunk_title, c.summary_short, c.project_id,
                c.importance_score, 1 if c.has_open_loops else 0,
            )
            for c in chunks
        ]
        self.conn.executemany(
            """
            INSERT INTO chunks (
                chunk_id, conversation_id, chunk_index,
                start_message_ordinal, end_message_ordinal,
                chunk_title, summary_short, project_id,
                importance_score, has_open_loops
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def list_chunks(self, conversation_id: str) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE conversation_id = ? ORDER BY chunk_index",
            (conversation_id,),
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        r = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(r) if r else None

    # --- derived objects ----------------------------------------------------

    def upsert_derived_object(self, obj: DerivedObject) -> str:
        """Upsert by (kind, canonical_key). Returns the canonical object_id."""
        if obj.canonical_key:
            existing = self.conn.execute(
                "SELECT object_id FROM derived_objects WHERE kind = ? AND canonical_key = ?",
                (obj.kind, obj.canonical_key),
            ).fetchone()
            if existing:
                self.conn.execute(
                    """
                    UPDATE derived_objects
                       SET title = ?, description = ?, project_id = COALESCE(?, project_id),
                           updated_at = ?
                     WHERE object_id = ?
                    """,
                    (obj.title, obj.description, obj.project_id, obj.updated_at, existing["object_id"]),
                )
                return existing["object_id"]
        self.conn.execute(
            """
            INSERT INTO derived_objects (
                object_id, kind, title, description, project_id,
                state, canonical_key, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                obj.object_id, obj.kind, obj.title, obj.description,
                obj.project_id, obj.state, obj.canonical_key,
                obj.created_at, obj.updated_at,
            ),
        )
        return obj.object_id

    def list_derived_objects(
        self, kind: str | None = None, project_id: str | None = None
    ) -> list[DerivedObject]:
        sql = "SELECT * FROM derived_objects WHERE state = 'active'"
        params: list = []
        if kind is not None:
            sql += " AND kind = ?"; params.append(kind)
        if project_id is not None:
            sql += " AND project_id = ?"; params.append(project_id)
        sql += " ORDER BY title"
        rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_derived(r) for r in rows]

    def get_derived_object(self, object_id: str) -> DerivedObject | None:
        r = self.conn.execute(
            "SELECT * FROM derived_objects WHERE object_id = ?", (object_id,)
        ).fetchone()
        return _row_to_derived(r) if r else None

    def delete_orphan_derived_objects(self) -> int:
        """Remove derived objects with no remaining provenance links."""
        cur = self.conn.execute(
            """
            DELETE FROM derived_objects
             WHERE object_id NOT IN (SELECT DISTINCT object_id FROM provenance_links)
            """
        )
        return cur.rowcount or 0

    # --- provenance ---------------------------------------------------------

    def insert_provenance(self, link: ProvenanceLink) -> None:
        self.conn.execute(
            """
            INSERT INTO provenance_links (
                link_id, object_id, conversation_id, chunk_id, excerpt, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (link.link_id, link.object_id, link.conversation_id,
             link.chunk_id, link.excerpt, link.created_at),
        )

    def list_provenance_for_object(self, object_id: str) -> list[ProvenanceLink]:
        rows = self.conn.execute(
            "SELECT * FROM provenance_links WHERE object_id = ?", (object_id,)
        ).fetchall()
        return [_row_to_prov(r) for r in rows]

    def list_provenance_for_conversation(self, conversation_id: str) -> list[ProvenanceLink]:
        rows = self.conn.execute(
            "SELECT * FROM provenance_links WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        return [_row_to_prov(r) for r in rows]

    # --- FTS reindex --------------------------------------------------------

    def reindex_conversation_fts(self, conversation_id: str) -> None:
        """Rebuild FTS rows for one conversation.

        Only conversations in ``FTS_INDEXED_STATES`` ({indexed, private}) get
        FTS rows. All other states have their FTS rows wiped and the function
        returns without re-inserting. This makes the FTS index itself the
        ground truth for "searchable surface exists"; passing
        ``pending_review`` or ``quarantined`` in ``visible_states`` cannot
        surface any row.
        """
        cv_row = self.conn.execute(
            "SELECT rowid, * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if cv_row is None:
            return

        # Always wipe first: state may have just changed.
        self.conn.execute("DELETE FROM fts_conversations WHERE rowid = ?", (cv_row["rowid"],))
        msg_rowids = [
            r["rowid"] for r in self.conn.execute(
                "SELECT rowid FROM messages WHERE conversation_id = ?", (conversation_id,)
            ).fetchall()
        ]
        for rid in msg_rowids:
            self.conn.execute("DELETE FROM fts_messages WHERE rowid = ?", (rid,))
        chunk_rowids = [
            r["rowid"] for r in self.conn.execute(
                "SELECT rowid FROM chunks WHERE conversation_id = ?", (conversation_id,)
            ).fetchall()
        ]
        for rid in chunk_rowids:
            self.conn.execute("DELETE FROM fts_chunks WHERE rowid = ?", (rid,))

        if cv_row["state"] not in FTS_INDEXED_STATES:
            # pending_review / quarantined / any future state: no FTS rows.
            return

        manual = json.loads(cv_row["manual_tags"] or "[]")
        auto = json.loads(cv_row["auto_tags"] or "[]")
        tags_text = " ".join(manual + auto)
        self.conn.execute(
            "INSERT INTO fts_conversations(rowid, title, summary_short, summary_long, tags) VALUES (?, ?, ?, ?, ?)",
            (cv_row["rowid"], cv_row["title"] or "", cv_row["summary_short"] or "",
             cv_row["summary_long"] or "", tags_text),
        )

        msg_rows = self.conn.execute(
            "SELECT rowid, content_text, role FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        for mr in msg_rows:
            self.conn.execute(
                "INSERT INTO fts_messages(rowid, body, role) VALUES (?, ?, ?)",
                (mr["rowid"], mr["content_text"] or "", mr["role"] or ""),
            )

        chunk_rows = self.conn.execute(
            "SELECT rowid, chunk_id, start_message_ordinal, end_message_ordinal, chunk_title, summary_short FROM chunks WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        for cr in chunk_rows:
            body_rows = self.conn.execute(
                "SELECT content_text FROM messages WHERE conversation_id = ? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
                (conversation_id, cr["start_message_ordinal"], cr["end_message_ordinal"]),
            ).fetchall()
            body = "\n\n".join((b["content_text"] or "") for b in body_rows)
            self.conn.execute(
                "INSERT INTO fts_chunks(rowid, chunk_title, summary_short, body) VALUES (?, ?, ?, ?)",
                (cr["rowid"], cr["chunk_title"] or "", cr["summary_short"] or "", body),
            )

    def rebuild_all_fts(self) -> None:
        """Drop all FTS rows, then reindex only FTS-eligible conversations."""
        self.conn.execute("DELETE FROM fts_conversations")
        self.conn.execute("DELETE FROM fts_messages")
        self.conn.execute("DELETE FROM fts_chunks")
        placeholders = ",".join("?" for _ in FTS_INDEXED_STATES)
        rows = self.conn.execute(
            f"SELECT conversation_id FROM conversations WHERE state IN ({placeholders})",
            tuple(FTS_INDEXED_STATES),
        ).fetchall()
        for row in rows:
            self.reindex_conversation_fts(row["conversation_id"])

    # --- generic ------------------------------------------------------------

    def vacuum(self) -> None:
        self.conn.commit()
        self.conn.execute("VACUUM")


# --- helpers ----------------------------------------------------------------

def _row_to_conversation(row) -> Conversation:
    return Conversation(
        conversation_id=row["conversation_id"],
        source=row["source"],
        source_conversation_id=row["source_conversation_id"],
        source_export_fingerprint=row["source_export_fingerprint"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        imported_at=row["imported_at"],
        state=row["state"],
        message_count=row["message_count"],
        summary_short=row["summary_short"] or "",
        summary_long=row["summary_long"],
        manual_tags=json.loads(row["manual_tags"] or "[]"),
        auto_tags=json.loads(row["auto_tags"] or "[]"),
        primary_project_id=row["primary_project_id"],
        importance_score=row["importance_score"],
        resurfacing_score=row["resurfacing_score"],
        has_open_loops=bool(row["has_open_loops"]),
        schema_version=row["schema_version"],
        parser_version=row["parser_version"],
        notes_local=row["notes_local"] or "",
    )


def _row_to_chunk(row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        conversation_id=row["conversation_id"],
        chunk_index=row["chunk_index"],
        start_message_ordinal=row["start_message_ordinal"],
        end_message_ordinal=row["end_message_ordinal"],
        chunk_title=row["chunk_title"] or "",
        summary_short=row["summary_short"] or "",
        project_id=row["project_id"],
        importance_score=row["importance_score"],
        has_open_loops=bool(row["has_open_loops"]),
    )


def _row_to_derived(row) -> DerivedObject:
    return DerivedObject(
        object_id=row["object_id"],
        kind=row["kind"],
        title=row["title"],
        description=row["description"] or "",
        project_id=row["project_id"],
        state=row["state"],
        canonical_key=row["canonical_key"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_prov(row) -> ProvenanceLink:
    return ProvenanceLink(
        link_id=row["link_id"],
        object_id=row["object_id"],
        conversation_id=row["conversation_id"],
        chunk_id=row["chunk_id"],
        excerpt=row["excerpt"],
        created_at=row["created_at"],
    )


# --- connection management --------------------------------------------------

def open_store(vault: Vault) -> Store:
    """Open (and bootstrap if needed) the SQLite store for a vault."""
    vault.db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(vault.db_path)
    conn.row_factory = sqlite3.Row
    # Enable secure_delete + WAL on every connection (PRAGMAs are connection-scoped).
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA secure_delete = ON")
    store = Store(conn=conn)
    store.bootstrap()
    return store


@contextmanager
def transaction(store: Store) -> Iterator[Store]:
    """Run a block in a single transaction."""
    try:
        store.conn.execute("BEGIN")
        yield store
        store.conn.commit()
    except Exception:
        store.conn.rollback()
        raise
