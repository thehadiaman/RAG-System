"""SQLite database for documents, pages, chunks and chats.

Schema changes are numbered migrations. SQLite's `user_version` records how
many have been applied, so an existing database is upgraded in place on
startup. Only ever append to MIGRATIONS; never edit one that has shipped.

Each `session()` opens its own short-lived connection, which keeps things
safe when background PDF processing and web requests use the database at
the same time.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

NOW = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"

MIGRATIONS: list[str] = [
    # 1: documents, pages, chunks, chats, messages
    f"""
    CREATE TABLE documents (
        id              TEXT PRIMARY KEY,
        filename        TEXT NOT NULL,
        sha256          TEXT NOT NULL UNIQUE,
        size_bytes      INTEGER NOT NULL,
        page_count      INTEGER,
        status          TEXT NOT NULL DEFAULT 'uploaded',
        extractor       TEXT,
        progress_done   INTEGER NOT NULL DEFAULT 0,
        progress_total  INTEGER NOT NULL DEFAULT 0,
        progress_note   TEXT,
        error           TEXT,
        created_at      TEXT NOT NULL DEFAULT ({NOW}),
        updated_at      TEXT NOT NULL DEFAULT ({NOW})
    );

    CREATE TABLE pages (
        document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page_no         INTEGER NOT NULL,
        text            TEXT NOT NULL DEFAULT '',
        has_text_layer  INTEGER NOT NULL DEFAULT 1,
        image_path      TEXT,
        PRIMARY KEY (document_id, page_no)
    );

    CREATE TABLE chunks (
        id              TEXT PRIMARY KEY,
        document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        position        INTEGER NOT NULL,
        page_no         INTEGER NOT NULL,
        kind            TEXT NOT NULL CHECK (kind IN ('text', 'table', 'figure')),
        section         TEXT,
        text            TEXT NOT NULL,
        image_path      TEXT
    );
    CREATE INDEX chunks_by_document ON chunks(document_id, position);

    CREATE TABLE chats (
        id              TEXT PRIMARY KEY,
        title           TEXT NOT NULL,
        model_id        TEXT,
        created_at      TEXT NOT NULL DEFAULT ({NOW}),
        updated_at      TEXT NOT NULL DEFAULT ({NOW})
    );

    CREATE TABLE messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id         TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
        role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content         TEXT NOT NULL,
        model_id        TEXT,
        sources         TEXT,  -- JSON list of the chunks an answer cited
        created_at      TEXT NOT NULL DEFAULT ({NOW})
    );
    CREATE INDEX messages_by_chat ON messages(chat_id, id);
    """,
    # 2: page details and figures (charts, pictures) found in PDFs
    """
    ALTER TABLE pages ADD COLUMN char_count INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE pages ADD COLUMN table_count INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE pages ADD COLUMN needs_ocr INTEGER NOT NULL DEFAULT 0;

    CREATE TABLE figures (
        id              TEXT PRIMARY KEY,
        document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page_no         INTEGER NOT NULL,
        position        INTEGER NOT NULL,  -- order on the page
        source          TEXT NOT NULL,     -- which signal found it: layout, picture or drawing
        x0 REAL NOT NULL, y0 REAL NOT NULL, x1 REAL NOT NULL, y1 REAL NOT NULL,
        image_path      TEXT NOT NULL,
        caption         TEXT,
        text_inside     TEXT,
        description     TEXT,              -- written by the vision model
        described_by    TEXT
    );
    CREATE INDEX figures_by_document ON figures(document_id, page_no, position);
    """,
    # 3: results of the vision step
    """
    ALTER TABLE figures ADD COLUMN kind TEXT;  -- chart, table, diagram, photo, decorative or other
    ALTER TABLE figures ADD COLUMN describe_error TEXT;
    ALTER TABLE pages ADD COLUMN text_source TEXT NOT NULL DEFAULT 'pdf';  -- 'pdf' or 'vision'
    ALTER TABLE pages ADD COLUMN read_error TEXT;
    ALTER TABLE documents ADD COLUMN warning TEXT;
    """,
]


class DatabaseError(Exception):
    pass


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def migrate(self) -> int:
        """Create or upgrade the schema. Returns the schema version."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            # WAL lets readers keep working while a background job writes.
            conn.execute("PRAGMA journal_mode = WAL")
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            if current > len(MIGRATIONS):
                raise DatabaseError(
                    f"{self.path} has schema version {current}, but this app only knows "
                    f"{len(MIGRATIONS)}. It was made by a newer version of the app."
                )
            for number, script in enumerate(MIGRATIONS[current:], start=current + 1):
                # One transaction per migration: it applies fully or not at all.
                conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")
            return len(MIGRATIONS)
        finally:
            conn.close()

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits when the block succeeds and rolls back if it raises."""
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def ping(self) -> int:
        """Schema version, read from disk (proves the file opens)."""
        with self.session() as conn:
            return conn.execute("PRAGMA user_version").fetchone()[0]
