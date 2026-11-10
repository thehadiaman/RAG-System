"""Processes uploaded PDFs in the background, one document at a time.

Steps: read the PDF (text, tables, figures), let a vision model read the
figures and scanned pages, then cut everything into chunks and index them for
search. One document at a time because reading is CPU-heavy and the local
vision model needs the whole GPU. Documents left unfinished when the app
stopped are picked up again on the next start, from the step they were in.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
from pathlib import Path

from app.ingest.describe import describe_document
from app.ingest.index import index_document
from app.ingest.pdf import PageResult, extract_document
from app.ingest.vision import VisionService
from app.llm.embedder import Embedder, EmbedderError
from app.store import documents
from app.store.db import Database
from app.store.documents import (
    DESCRIBING,
    EXTRACTED,
    EXTRACTING,
    FAILED,
    INDEXING,
    PENDING_ON_START,
    QUEUED,
    READY,
    DocumentGone,
)
from app.store.vectors import VectorStore

log = logging.getLogger(__name__)


class _Stopping(Exception):
    """The app is shutting down; leave the document to be resumed on the next start."""


def document_dir(data_dir: Path, doc_id: str) -> Path:
    return data_dir / "documents" / doc_id


class IngestQueue:
    def __init__(
        self,
        db: Database,
        data_dir: Path,
        preview_dpi: int = 100,
        figure_dpi: int = 200,
        vision: VisionService | None = None,
        embedder: Embedder | None = None,
        vectors: VectorStore | None = None,
    ) -> None:
        self.db = db
        self.data_dir = data_dir
        self.preview_dpi = preview_dpi
        self.figure_dpi = figure_dpi
        self.vision = vision
        self.embedder = embedder
        self.vectors = vectors
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="ingest-worker")
        for doc_id in documents.ids_with_status(self.db, PENDING_ON_START):
            log.info("Resuming document %s", doc_id)
            self.enqueue(doc_id)

    async def stop(self) -> None:
        self._stopping.set()
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

    def enqueue(self, doc_id: str) -> None:
        self._queue.put_nowait(doc_id)

    async def wait_until_idle(self) -> None:
        await self._queue.join()

    async def _run(self) -> None:
        while True:
            doc_id = await self._queue.get()
            try:
                await self.process(doc_id)
            except Exception:  # noqa: BLE001 - one bad document must not stop the worker
                log.exception("Unexpected error processing %s", doc_id)
            finally:
                self._queue.task_done()

    async def process(self, doc_id: str) -> None:
        doc = documents.get(self.db, doc_id)
        if doc is None:
            return
        total = doc["page_count"] or 0
        status = doc["status"]
        warning = doc["warning"]
        try:
            # Start from the step the document was in.
            if status in (QUEUED, EXTRACTING):
                await self._extract(doc_id, total)
            if status in (QUEUED, EXTRACTING, DESCRIBING):
                warning = await self._describe(doc_id, doc["extractor"] or "none")
            final, warning = await self._index(doc_id, warning)
            documents.set_status(self.db, doc_id, final, note=None, done=total, total=total, warning=warning)
            log.info("Finished %s as %s (%d pages)%s", doc_id, final, total, f"; warning: {warning}" if warning else "")
        except _Stopping:
            log.info("Stopped while processing %s; it will resume on the next start", doc_id)
        except DocumentGone:
            log.info("Document %s was deleted while it was being processed", doc_id)
        except Exception as exc:  # noqa: BLE001 - recorded on the document for the user to see
            log.exception("Failed to process %s", doc_id)
            try:
                documents.set_status(self.db, doc_id, FAILED, error=str(exc) or type(exc).__name__)
            except DocumentGone:
                pass

    async def _extract(self, doc_id: str, total: int) -> None:
        folder = document_dir(self.data_dir, doc_id)

        def on_page(result: PageResult, preview: Path, figure_paths: list[Path]) -> None:
            if self._stopping.is_set():
                raise _Stopping
            documents.save_page(self.db, doc_id, self.data_dir, result, preview, figure_paths)
            documents.set_status(
                self.db,
                doc_id,
                EXTRACTING,
                note=f"Reading page {result.page_no} of {total}",
                done=result.page_no,
                total=total,
            )

        # Start clean, in case an earlier run stopped halfway.
        documents.clear_extraction(self.db, doc_id)
        for sub in ("pages", "figures"):
            shutil.rmtree(folder / sub, ignore_errors=True)
        documents.set_status(self.db, doc_id, EXTRACTING, note="Reading pages", done=0, total=total)

        await asyncio.to_thread(
            extract_document, folder / "original.pdf", folder, on_page, self.preview_dpi, self.figure_dpi
        )

    async def _describe(self, doc_id: str, extractor: str) -> str | None:
        """Runs the vision step. Returns a warning for the user, or None if all went well."""
        if extractor == "none":
            return None
        if self.vision is None:
            return "Charts and scanned pages were not read: no vision model is set up."
        problem = await self.vision.problem(extractor)
        if problem:
            return f"Charts and scanned pages were not read: {problem}"

        documents.set_status(self.db, doc_id, DESCRIBING, note="Loading the vision model", done=0, total=0)

        def on_progress(done: int, total: int) -> None:
            documents.set_status(
                self.db,
                doc_id,
                DESCRIBING,
                note=f"Reading figures and scanned pages: {done} of {total}",
                done=done,
                total=total,
            )

        outcome = await describe_document(
            self.db,
            self.data_dir,
            doc_id,
            self.vision.describer(extractor),
            self.vision.max_megapixels,
            on_progress,
            should_stop=self._stopping.is_set,
        )
        if outcome.interrupted:
            raise _Stopping
        return outcome.warning()

    async def _index(self, doc_id: str, warning: str | None) -> tuple[str, str | None]:
        """Runs the indexing step. Returns the document's final status and warning."""
        if self.embedder is None or self.vectors is None:
            return EXTRACTED, warning

        def show(note: str, done: int = 0, total: int = 0) -> None:
            if self._stopping.is_set():
                raise _Stopping
            documents.set_status(self.db, doc_id, INDEXING, note=note, done=done, total=total, warning=warning)

        show("Preparing the search index")
        try:
            await index_document(
                self.db,
                self.vectors,
                self.embedder,
                doc_id,
                lambda done, total: show(f"Indexing: {done} of {total} chunks", done, total),
            )
        except EmbedderError as exc:
            # The text is safe; it just can't be searched until indexing is run again.
            problem = f"Not searchable yet: {exc}"
            return EXTRACTED, f"{warning} {problem}" if warning else problem
        return READY, warning
