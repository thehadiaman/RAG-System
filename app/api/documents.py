import asyncio
import hashlib
import shutil
import sqlite3
from pathlib import Path as FilePath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.deps import get_db, get_ingest, get_vectors
from app.config import Settings, get_settings
from app.ingest.jobs import IngestQueue, document_dir
from app.ingest.pdf import PdfError, check_pdf
from app.store import documents
from app.store.db import Database
from app.store.vectors import VectorStore

router = APIRouter(prefix="/documents", tags=["documents"])

DocId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]
READ_SIZE = 1024 * 1024


class Progress(BaseModel):
    done: int
    total: int
    note: str | None = None


class DocumentOut(BaseModel):
    id: str
    filename: str
    size_bytes: int
    page_count: int | None
    status: str
    progress: Progress
    error: str | None
    created_at: str
    updated_at: str


class UploadOut(DocumentOut):
    duplicate: bool  # this exact file was uploaded before; nothing new was stored


class PageSummary(BaseModel):
    page_no: int
    char_count: int
    table_count: int
    figure_count: int
    needs_ocr: bool  # scanned page: will be read by the vision model
    image_url: str


class FigureOut(BaseModel):
    id: str
    page_no: int
    position: int
    source: str
    bbox: tuple[float, float, float, float]
    caption: str | None
    text_inside: str | None
    description: str | None
    image_url: str


class DocumentDetail(DocumentOut):
    pages: list[PageSummary]
    figures: list[FigureOut]


class PageDetail(BaseModel):
    page_no: int
    text: str
    char_count: int
    table_count: int
    needs_ocr: bool
    image_url: str
    figures: list[FigureOut]


# ---- Conversions ----


def _document(row: dict) -> dict:
    return {
        **{key: row[key] for key in ("id", "filename", "size_bytes", "page_count", "status", "error")},
        "progress": Progress(done=row["progress_done"], total=row["progress_total"], note=row["progress_note"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _page_url(doc_id: str, page_no: int) -> str:
    return f"/api/documents/{doc_id}/pages/{page_no}/image"


def _figure(row: dict) -> FigureOut:
    return FigureOut(
        **{key: row[key] for key in ("id", "page_no", "position", "source", "caption", "text_inside", "description")},
        bbox=(row["x0"], row["y0"], row["x1"], row["y1"]),
        image_url=f"/api/documents/{row['document_id']}/figures/{row['id']}/image",
    )


def _get_or_404(db: Database, doc_id: str) -> dict:
    row = documents.get(db, doc_id)
    if row is None:
        raise HTTPException(404, "Document not found.")
    return row


# ---- Endpoints ----


@router.post("", status_code=201)
async def upload(
    file: UploadFile,
    response: Response,
    db: Database = Depends(get_db),
    ingest: IngestQueue = Depends(get_ingest),
    settings: Settings = Depends(get_settings),
) -> UploadOut:
    """Upload a PDF. It is read in the background; poll GET /api/documents/{id} for progress."""
    docs_root = settings.data_dir / "documents"
    docs_root.mkdir(parents=True, exist_ok=True)
    temp = docs_root / f".upload-{documents.new_id()}.pdf"
    limit = settings.max_upload_mb * 1024 * 1024
    digest = hashlib.sha256()
    size = 0
    try:
        with temp.open("wb") as out:
            while chunk := await file.read(READ_SIZE):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"This file is larger than the {settings.max_upload_mb} MB limit.")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "This file is empty.")
        try:
            page_count = await asyncio.to_thread(check_pdf, temp)
        except PdfError as exc:
            raise HTTPException(400, str(exc)) from exc

        sha256 = digest.hexdigest()
        existing = documents.find_by_sha256(db, sha256)
        if existing:
            response.status_code = 200
            return UploadOut(**_document(existing), duplicate=True)

        doc_id = documents.new_id()
        folder = document_dir(settings.data_dir, doc_id)
        folder.mkdir(parents=True)
        temp.rename(folder / "original.pdf")
        filename = FilePath(file.filename or "document.pdf").name[:255]
        try:
            row = documents.create(db, doc_id, filename, sha256, size, page_count)
        except sqlite3.IntegrityError:
            # The same file was uploaded twice at the same moment; keep the first.
            shutil.rmtree(folder, ignore_errors=True)
            response.status_code = 200
            return UploadOut(**_document(documents.find_by_sha256(db, sha256)), duplicate=True)

        ingest.enqueue(doc_id)
        return UploadOut(**_document(row), duplicate=False)
    finally:
        temp.unlink(missing_ok=True)


@router.get("")
async def list_documents(db: Database = Depends(get_db)) -> list[DocumentOut]:
    return [DocumentOut(**_document(row)) for row in documents.list_all(db)]


@router.get("/{doc_id}")
async def get_document(doc_id: DocId, db: Database = Depends(get_db)) -> DocumentDetail:
    row = _get_or_404(db, doc_id)
    pages = [
        PageSummary(**{**page, "needs_ocr": bool(page["needs_ocr"])}, image_url=_page_url(doc_id, page["page_no"]))
        for page in documents.pages(db, doc_id)
    ]
    return DocumentDetail(**_document(row), pages=pages, figures=[_figure(f) for f in documents.figures(db, doc_id)])


@router.get("/{doc_id}/pages/{page_no}")
async def get_page(doc_id: DocId, page_no: int, db: Database = Depends(get_db)) -> PageDetail:
    _get_or_404(db, doc_id)
    page = documents.page(db, doc_id, page_no)
    if page is None:
        raise HTTPException(404, "Page not found (it may not have been read yet).")
    return PageDetail(
        page_no=page_no,
        text=page["text"],
        char_count=page["char_count"],
        table_count=page["table_count"],
        needs_ocr=bool(page["needs_ocr"]),
        image_url=_page_url(doc_id, page_no),
        figures=[_figure(f) for f in documents.figures(db, doc_id, page_no)],
    )


@router.get("/{doc_id}/pages/{page_no}/image")
async def page_image(
    doc_id: DocId, page_no: int, db: Database = Depends(get_db), settings: Settings = Depends(get_settings)
) -> FileResponse:
    page = documents.page(db, doc_id, page_no)
    if page is None or not page["image_path"]:
        raise HTTPException(404, "Page image not found.")
    return FileResponse(settings.data_dir / page["image_path"])


@router.get("/{doc_id}/figures/{figure_id}/image")
async def figure_image(
    doc_id: DocId,
    figure_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")],
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    figure = documents.figure(db, doc_id, figure_id)
    if figure is None:
        raise HTTPException(404, "Figure not found.")
    return FileResponse(settings.data_dir / figure["image_path"])


@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: DocId,
    db: Database = Depends(get_db),
    vectors: VectorStore = Depends(get_vectors),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Removes the document, its pages, figures, search vectors and files."""
    if not documents.delete(db, doc_id):
        raise HTTPException(404, "Document not found.")
    await asyncio.to_thread(vectors.delete_document, doc_id)
    await asyncio.to_thread(shutil.rmtree, document_dir(settings.data_dir, doc_id), True)
    return Response(status_code=204)
