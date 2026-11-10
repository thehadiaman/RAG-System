import asyncio
import hashlib
import shutil
import sqlite3
from pathlib import Path as FilePath
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Form, HTTPException, Path, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.deps import get_db, get_ingest, get_vectors, get_vision
from app.config import Settings, get_settings
from app.ingest.jobs import IngestQueue, document_dir
from app.ingest.pdf import PdfError, check_pdf
from app.ingest.vision import Extractor, VisionService
from app.store import chunks as chunk_store
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
    extractor: str | None  # who reads charts and scanned pages: local, aws or none
    progress: Progress
    error: str | None  # why processing failed
    warning: str | None  # processing finished, but something was skipped
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
    kind: str | None  # chart, table, diagram, photo, decorative or other (set by the vision model)
    description: str | None
    described_by: str | None
    describe_error: str | None
    image_url: str


class DocumentDetail(DocumentOut):
    pages: list[PageSummary]
    figures: list[FigureOut]
    chunk_counts: dict[str, int]  # searchable pieces by kind: text, table, figure


class ChunkOut(BaseModel):
    id: str
    position: int
    page_no: int
    kind: str
    section: str
    text: str
    figure_id: str | None
    image_url: str | None


class PageDetail(BaseModel):
    page_no: int
    text: str
    text_source: str  # "pdf", or "vision" for scanned pages read by the vision model
    char_count: int
    table_count: int
    needs_ocr: bool
    read_error: str | None
    image_url: str
    figures: list[FigureOut]


# ---- Conversions ----


def _document(row: dict) -> dict:
    return {
        **{
            key: row[key]
            for key in ("id", "filename", "size_bytes", "page_count", "status", "extractor", "error", "warning")
        },
        "progress": Progress(done=row["progress_done"], total=row["progress_total"], note=row["progress_note"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _page_url(doc_id: str, page_no: int) -> str:
    return f"/api/documents/{doc_id}/pages/{page_no}/image"


def _figure(row: dict) -> FigureOut:
    return FigureOut(
        **{
            key: row[key]
            for key in (
                "id", "page_no", "position", "source", "caption", "text_inside",
                "kind", "description", "described_by", "describe_error",
            )
        },
        bbox=(row["x0"], row["y0"], row["x1"], row["y1"]),
        image_url=f"/api/documents/{row['document_id']}/figures/{row['id']}/image",
    )


def _get_or_404(db: Database, doc_id: str) -> dict:
    row = documents.get(db, doc_id)
    if row is None:
        raise HTTPException(404, "Document not found.")
    return row


# ---- Endpoints ----


async def _check_extractor(vision: VisionService, extractor: str) -> None:
    problem = await vision.problem(extractor)
    if problem:
        raise HTTPException(400, f"{vision.label(extractor)} can't be used: {problem}")


@router.post("", status_code=201)
async def upload(
    file: UploadFile,
    response: Response,
    extractor: Annotated[Extractor | None, Form()] = None,
    db: Database = Depends(get_db),
    ingest: IngestQueue = Depends(get_ingest),
    vision: VisionService = Depends(get_vision),
    settings: Settings = Depends(get_settings),
) -> UploadOut:
    """Upload a PDF. It is read in the background; poll GET /api/documents/{id} for progress.

    `extractor` picks who reads charts and scanned pages: "local" (Qwen2.5-VL),
    "aws" (Nova Lite) or "none". Leave it out to use DEFAULT_EXTRACTOR.
    """
    extractor = extractor or settings.default_extractor
    await _check_extractor(vision, extractor)

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
            row = documents.create(db, doc_id, filename, sha256, size, page_count, extractor)
        except sqlite3.IntegrityError:
            # The same file was uploaded twice at the same moment; keep the first.
            shutil.rmtree(folder, ignore_errors=True)
            response.status_code = 200
            return UploadOut(**_document(documents.find_by_sha256(db, sha256)), duplicate=True)

        ingest.enqueue(doc_id)
        return UploadOut(**_document(row), duplicate=False)
    finally:
        temp.unlink(missing_ok=True)


@router.post("/{doc_id}/describe", status_code=202)
async def describe_again(
    doc_id: DocId,
    extractor: Extractor,
    db: Database = Depends(get_db),
    ingest: IngestQueue = Depends(get_ingest),
    vision: VisionService = Depends(get_vision),
) -> DocumentOut:
    """Run the vision step again, for example with the other model, replacing earlier results."""
    row = _get_or_404(db, doc_id)
    if row["status"] in documents.IN_PROGRESS:
        raise HTTPException(409, "This document is still being processed. Try again when it's done.")
    if row["status"] == documents.FAILED:
        raise HTTPException(409, "This document failed to process. Delete it and upload it again.")
    await _check_extractor(vision, extractor)

    documents.reset_descriptions(db, doc_id)
    documents.set_extractor(db, doc_id, extractor)
    documents.set_status(db, doc_id, documents.DESCRIBING, note="Waiting for the vision model", done=0, total=0)
    ingest.enqueue(doc_id)
    return DocumentOut(**_document(documents.get(db, doc_id)))


@router.post("/{doc_id}/index", status_code=202)
async def index_again(
    doc_id: DocId,
    db: Database = Depends(get_db),
    ingest: IngestQueue = Depends(get_ingest),
) -> DocumentOut:
    """Rebuild the document's search index, for example after the embedding model was unavailable."""
    row = _get_or_404(db, doc_id)
    if row["status"] in documents.IN_PROGRESS:
        raise HTTPException(409, "This document is still being processed. Try again when it's done.")
    if row["status"] == documents.FAILED:
        raise HTTPException(409, "This document failed to process. Delete it and upload it again.")

    documents.set_status(
        db, doc_id, documents.INDEXING, note="Waiting to be indexed", done=0, total=0, warning=row["warning"]
    )
    ingest.enqueue(doc_id)
    return DocumentOut(**_document(documents.get(db, doc_id)))


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
    return DocumentDetail(
        **_document(row),
        pages=pages,
        figures=[_figure(f) for f in documents.figures(db, doc_id)],
        chunk_counts=chunk_store.counts(db, doc_id),
    )


@router.get("/{doc_id}/chunks")
async def get_chunks(
    doc_id: DocId,
    page_no: int | None = None,
    kind: Literal["text", "table", "figure"] | None = None,
    db: Database = Depends(get_db),
) -> list[ChunkOut]:
    """The searchable pieces a document was cut into, optionally for one page or one kind."""
    _get_or_404(db, doc_id)
    return [
        ChunkOut(
            **{key: row[key] for key in ("id", "position", "page_no", "kind", "section", "text", "figure_id")},
            image_url=f"/api/documents/{doc_id}/figures/{row['figure_id']}/image" if row["figure_id"] else None,
        )
        for row in chunk_store.for_document(db, doc_id, page_no, kind)
    ]


@router.get("/{doc_id}/pages/{page_no}")
async def get_page(doc_id: DocId, page_no: int, db: Database = Depends(get_db)) -> PageDetail:
    _get_or_404(db, doc_id)
    page = documents.page(db, doc_id, page_no)
    if page is None:
        raise HTTPException(404, "Page not found (it may not have been read yet).")
    return PageDetail(
        page_no=page_no,
        text=page["text"],
        text_source=page["text_source"],
        char_count=page["char_count"],
        table_count=page["table_count"],
        needs_ocr=bool(page["needs_ocr"]),
        read_error=page["read_error"],
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
