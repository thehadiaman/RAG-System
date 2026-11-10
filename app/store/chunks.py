"""Chunks in SQLite. Inserting or deleting a row here also updates the keyword index (see migration 4)."""

from __future__ import annotations

from app.ingest.chunker import Chunk
from app.store.db import Database
from app.store.documents import DocumentGone, new_id


def replace(db: Database, doc_id: str, chunks: list[Chunk]) -> list[str]:
    """Swap a document's chunks for a new set, in one transaction. Returns the new ids, in order."""
    ids = [new_id() for _ in chunks]
    with db.session() as conn:
        if not conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone():
            raise DocumentGone(doc_id)
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.executemany(
            """INSERT INTO chunks (id, document_id, position, page_no, kind, section, text, figure_id, image_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (chunk_id, doc_id, position, c.page_no, c.kind, c.section, c.text, c.figure_id, c.image_path)
                for position, (chunk_id, c) in enumerate(zip(ids, chunks, strict=True))
            ],
        )
    return ids


def for_document(db: Database, doc_id: str, page_no: int | None = None, kind: str | None = None) -> list[dict]:
    query = "SELECT * FROM chunks WHERE document_id = ?"
    params: list = [doc_id]
    if page_no is not None:
        query += " AND page_no = ?"
        params.append(page_no)
    if kind is not None:
        query += " AND kind = ?"
        params.append(kind)
    with db.session() as conn:
        rows = conn.execute(query + " ORDER BY position", params).fetchall()
    return [dict(row) for row in rows]


def counts(db: Database, doc_id: str) -> dict[str, int]:
    """How many chunks of each kind a document has."""
    with db.session() as conn:
        rows = conn.execute(
            "SELECT kind, COUNT(*) AS n FROM chunks WHERE document_id = ? GROUP BY kind", (doc_id,)
        ).fetchall()
    return {row["kind"]: row["n"] for row in rows}
