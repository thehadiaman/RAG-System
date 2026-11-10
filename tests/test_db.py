import sqlite3

import pytest

from app.store.db import MIGRATIONS, Database, DatabaseError


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "app.db")
    database.migrate()
    return database


def add_document(conn, doc_id="doc1", sha="abc"):
    conn.execute(
        "INSERT INTO documents (id, filename, sha256, size_bytes) VALUES (?, ?, ?, ?)",
        (doc_id, "report.pdf", sha, 1234),
    )


def test_fresh_database_gets_every_migration(tmp_path):
    db = Database(tmp_path / "new" / "app.db")

    assert db.migrate() == len(MIGRATIONS)
    assert db.ping() == len(MIGRATIONS)
    with db.session() as conn:
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"documents", "pages", "chunks", "chats", "messages"} <= tables


def test_migrating_twice_is_safe(db):
    with db.session() as conn:
        add_document(conn)

    assert db.migrate() == len(MIGRATIONS)
    with db.session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_refuses_a_database_made_by_a_newer_app(tmp_path):
    path = tmp_path / "app.db"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 99")

    with pytest.raises(DatabaseError, match="newer version"):
        Database(path).migrate()


def test_uses_write_ahead_logging(db):
    with db.session() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_rows_can_be_saved_and_read_back(db):
    with db.session() as conn:
        add_document(conn)

    with db.session() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = 'doc1'").fetchone()

    assert row["filename"] == "report.pdf"
    assert row["status"] == "uploaded"
    assert row["created_at"].endswith("Z")


def test_deleting_a_document_removes_its_pages_and_chunks(db):
    with db.session() as conn:
        add_document(conn)
        conn.execute("INSERT INTO pages (document_id, page_no, text) VALUES ('doc1', 1, 'Revenue grew')")
        conn.execute(
            "INSERT INTO chunks (id, document_id, position, page_no, kind, text) "
            "VALUES ('c1', 'doc1', 0, 1, 'text', 'Revenue grew')"
        )

    with db.session() as conn:
        conn.execute("DELETE FROM documents WHERE id = 'doc1'")

    with db.session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


def test_deleting_a_chat_removes_its_messages(db):
    with db.session() as conn:
        conn.execute("INSERT INTO chats (id, title) VALUES ('chat1', 'Q3 results')")
        conn.execute("INSERT INTO messages (chat_id, role, content) VALUES ('chat1', 'user', 'Hi')")
        conn.execute("DELETE FROM chats WHERE id = 'chat1'")

        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_a_failed_session_saves_nothing(db):
    with pytest.raises(RuntimeError), db.session() as conn:
        add_document(conn)
        raise RuntimeError("something broke halfway")

    with db.session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_chunk_kind_must_be_known(db):
    with pytest.raises(sqlite3.IntegrityError), db.session() as conn:
        add_document(conn)
        conn.execute(
            "INSERT INTO chunks (id, document_id, position, page_no, kind, text) "
            "VALUES ('c1', 'doc1', 0, 1, 'video', 'x')"
        )


def add_chunk(conn, chunk_id, text, section="Results", doc_id="doc1"):
    conn.execute(
        "INSERT INTO chunks (id, document_id, position, page_no, kind, section, text) VALUES (?, ?, 0, 1, 'text', ?, ?)",
        (chunk_id, doc_id, section, text),
    )


def keyword_hits(conn, query):
    rows = conn.execute(
        "SELECT c.id FROM chunks_fts f JOIN chunks c ON c.seq = f.rowid WHERE chunks_fts MATCH ? ORDER BY c.id",
        (query,),
    ).fetchall()
    return [row["id"] for row in rows]


def test_new_chunks_are_added_to_the_keyword_index(db):
    with db.session() as conn:
        add_document(conn)
        add_chunk(conn, "c1", "Revenue grew 12.5% to $4.2B in 4Q'25")
        add_chunk(conn, "c2", "Headcount fell slightly", section="People")

        assert keyword_hits(conn, "revenues") == ["c1"]  # stemming: revenues ~ revenue
        assert keyword_hits(conn, '"12.5"') == ["c1"]
        assert keyword_hits(conn, "people") == ["c2"]  # section names are searchable too
        assert keyword_hits(conn, "dividend") == []


def test_deleting_a_document_also_clears_its_keywords(db):
    with db.session() as conn:
        add_document(conn)
        add_chunk(conn, "c1", "Revenue grew")

    with db.session() as conn:
        conn.execute("DELETE FROM documents WHERE id = 'doc1'")

    with db.session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 0
        assert keyword_hits(conn, "revenue") == []
