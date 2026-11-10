"""Reading and writing documents, pages and figures in SQLite.

File paths are stored relative to the data folder, so the folder can move.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.ingest.pdf import TABLE_SEPARATOR, PageResult
from app.store.db import NOW, Database

# Document status, in the order a document moves through them.
QUEUED = "queued"
EXTRACTING = "extracting"  # reading text, tables and figures from the PDF
DESCRIBING = "describing"  # the vision model reads figures and scanned pages
EXTRACTED = "extracted"
FAILED = "failed"
IN_PROGRESS = (QUEUED, EXTRACTING, DESCRIBING)


class DocumentGone(Exception):
    """The document was deleted while something was working on it."""


def new_id() -> str:
    return uuid.uuid4().hex


def create(
    db: Database,
    doc_id: str,
    filename: str,
    sha256: str,
    size_bytes: int,
    page_count: int,
    extractor: str = "none",
) -> dict:
    with db.session() as conn:
        conn.execute(
            """INSERT INTO documents (id, filename, sha256, size_bytes, page_count, status, extractor)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, filename, sha256, size_bytes, page_count, QUEUED, extractor),
        )
    return get(db, doc_id)


def get(db: Database, doc_id: str) -> dict | None:
    with db.session() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def find_by_sha256(db: Database, sha256: str) -> dict | None:
    with db.session() as conn:
        row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
    return dict(row) if row else None


def list_all(db: Database) -> list[dict]:
    with db.session() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def ids_with_status(db: Database, statuses: tuple[str, ...]) -> list[str]:
    marks = ",".join("?" * len(statuses))
    with db.session() as conn:
        rows = conn.execute(
            f"SELECT id FROM documents WHERE status IN ({marks}) ORDER BY created_at", statuses
        ).fetchall()
    return [row["id"] for row in rows]


def set_status(
    db: Database,
    doc_id: str,
    status: str,
    *,
    note: str | None = None,
    done: int | None = None,
    total: int | None = None,
    error: str | None = None,
    warning: str | None = None,
) -> None:
    with db.session() as conn:
        cursor = conn.execute(
            f"""UPDATE documents SET
                   status = ?,
                   progress_note = ?,
                   progress_done = COALESCE(?, progress_done),
                   progress_total = COALESCE(?, progress_total),
                   error = ?,
                   warning = ?,
                   updated_at = {NOW}
               WHERE id = ?""",
            (status, note, done, total, error, warning, doc_id),
        )
        if cursor.rowcount == 0:
            raise DocumentGone(doc_id)


def clear_extraction(db: Database, doc_id: str) -> None:
    """Remove pages and figures from an earlier, unfinished run."""
    with db.session() as conn:
        conn.execute("DELETE FROM figures WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM pages WHERE document_id = ?", (doc_id,))


def save_page(
    db: Database,
    doc_id: str,
    data_dir: Path,
    result: PageResult,
    preview: Path,
    figure_paths: list[Path],
) -> None:
    with db.session() as conn:
        if not conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone():
            raise DocumentGone(doc_id)
        conn.execute(
            """INSERT OR REPLACE INTO pages
                   (document_id, page_no, text, has_text_layer, image_path, char_count, table_count, needs_ocr)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                result.page_no,
                result.text,
                int(result.char_count > 0),
                _relative(preview, data_dir),
                result.char_count,
                result.table_count,
                int(result.needs_ocr),
            ),
        )
        for position, (figure, path) in enumerate(zip(result.figures, figure_paths, strict=True), start=1):
            conn.execute(
                """INSERT INTO figures
                       (id, document_id, page_no, position, source, x0, y0, x1, y1, image_path, caption, text_inside)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_id(),
                    doc_id,
                    result.page_no,
                    position,
                    figure.source,
                    *figure.bbox,
                    _relative(path, data_dir),
                    figure.caption,
                    figure.text_inside,
                ),
            )


def pages(db: Database, doc_id: str) -> list[dict]:
    with db.session() as conn:
        rows = conn.execute(
            """SELECT p.page_no, p.char_count, p.table_count, p.needs_ocr, p.image_path,
                      (SELECT COUNT(*) FROM figures f
                        WHERE f.document_id = p.document_id AND f.page_no = p.page_no) AS figure_count
                 FROM pages p WHERE p.document_id = ? ORDER BY p.page_no""",
            (doc_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def page(db: Database, doc_id: str, page_no: int) -> dict | None:
    with db.session() as conn:
        row = conn.execute(
            "SELECT * FROM pages WHERE document_id = ? AND page_no = ?", (doc_id, page_no)
        ).fetchone()
    return dict(row) if row else None


def figures(db: Database, doc_id: str, page_no: int | None = None) -> list[dict]:
    query = "SELECT * FROM figures WHERE document_id = ?"
    params: tuple = (doc_id,)
    if page_no is not None:
        query += " AND page_no = ?"
        params += (page_no,)
    with db.session() as conn:
        rows = conn.execute(query + " ORDER BY page_no, position", params).fetchall()
    return [dict(row) for row in rows]


def figure(db: Database, doc_id: str, figure_id: str) -> dict | None:
    with db.session() as conn:
        row = conn.execute(
            "SELECT * FROM figures WHERE document_id = ? AND id = ?", (doc_id, figure_id)
        ).fetchone()
    return dict(row) if row else None


def set_extractor(db: Database, doc_id: str, extractor: str) -> None:
    with db.session() as conn:
        if conn.execute("UPDATE documents SET extractor = ? WHERE id = ?", (extractor, doc_id)).rowcount == 0:
            raise DocumentGone(doc_id)


# ---- Vision step ----


def figures_to_describe(db: Database, doc_id: str) -> list[dict]:
    """Figures without a description yet (so a restarted run skips finished ones)."""
    with db.session() as conn:
        rows = conn.execute(
            "SELECT * FROM figures WHERE document_id = ? AND description IS NULL ORDER BY page_no, position",
            (doc_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def scanned_pages_to_read(db: Database, doc_id: str) -> list[dict]:
    with db.session() as conn:
        rows = conn.execute(
            "SELECT * FROM pages WHERE document_id = ? AND needs_ocr = 1 AND text_source = 'pdf' ORDER BY page_no",
            (doc_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_figure_description(db: Database, figure_id: str, kind: str, description: str, described_by: str) -> None:
    with db.session() as conn:
        cursor = conn.execute(
            """UPDATE figures SET kind = ?, description = ?, described_by = ?, describe_error = NULL
               WHERE id = ?""",
            (kind, description, described_by, figure_id),
        )
        if cursor.rowcount == 0:
            raise DocumentGone(figure_id)


def save_figure_error(db: Database, figure_id: str, error: str) -> None:
    with db.session() as conn:
        conn.execute("UPDATE figures SET describe_error = ? WHERE id = ?", (error, figure_id))


def save_page_transcription(db: Database, doc_id: str, page_no: int, text: str) -> None:
    """A scanned page's text, read by the vision model, replaces its (empty) PDF text."""
    with db.session() as conn:
        cursor = conn.execute(
            """UPDATE pages SET text = ?, char_count = ?, table_count = ?, text_source = 'vision', read_error = NULL
               WHERE document_id = ? AND page_no = ?""",
            (text, len(text), len(TABLE_SEPARATOR.findall(text)), doc_id, page_no),
        )
        if cursor.rowcount == 0:
            raise DocumentGone(doc_id)


def save_page_error(db: Database, doc_id: str, page_no: int, error: str) -> None:
    with db.session() as conn:
        conn.execute(
            "UPDATE pages SET read_error = ? WHERE document_id = ? AND page_no = ?", (error, doc_id, page_no)
        )


def reset_descriptions(db: Database, doc_id: str) -> None:
    """Forget the vision step's results, so it can run again (maybe with another model)."""
    with db.session() as conn:
        conn.execute(
            """UPDATE figures SET kind = NULL, description = NULL, described_by = NULL, describe_error = NULL
               WHERE document_id = ?""",
            (doc_id,),
        )
        conn.execute(
            """UPDATE pages SET text = '', char_count = 0, table_count = 0, text_source = 'pdf', read_error = NULL
               WHERE document_id = ? AND text_source = 'vision'""",
            (doc_id,),
        )
        conn.execute("UPDATE pages SET read_error = NULL WHERE document_id = ?", (doc_id,))


def delete(db: Database, doc_id: str) -> bool:
    with db.session() as conn:
        return conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,)).rowcount > 0


def _relative(path: Path, data_dir: Path) -> str:
    return path.resolve().relative_to(data_dir.resolve()).as_posix()
