"""Processes uploaded PDFs in the background, one document at a time.

One at a time because extraction is CPU-heavy, and the vision step (next phase)
needs the GPU, which only fits one model. Documents left unfinished when the
app stopped are picked up again on the next start.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
from pathlib import Path

from app.ingest.pdf import PageResult, extract_document
from app.store import documents
from app.store.db import Database
from app.store.documents import EXTRACTED, EXTRACTING, FAILED, IN_PROGRESS, DocumentGone

log = logging.getLogger(__name__)


class _Stopping(Exception):
    """The app is shutting down; leave the document to be resumed on the next start."""


def document_dir(data_dir: Path, doc_id: str) -> Path:
    return data_dir / "documents" / doc_id


class IngestQueue:
    def __init__(self, db: Database, data_dir: Path, preview_dpi: int = 100, figure_dpi: int = 200) -> None:
        self.db = db
        self.data_dir = data_dir
        self.preview_dpi = preview_dpi
        self.figure_dpi = figure_dpi
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="ingest-worker")
        for doc_id in documents.ids_with_status(self.db, IN_PROGRESS):
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

        try:
            # Start clean, in case an earlier run stopped halfway.
            documents.clear_extraction(self.db, doc_id)
            for sub in ("pages", "figures"):
                shutil.rmtree(folder / sub, ignore_errors=True)
            documents.set_status(self.db, doc_id, EXTRACTING, note="Reading pages", done=0, total=total)

            await asyncio.to_thread(
                extract_document, folder / "original.pdf", folder, on_page, self.preview_dpi, self.figure_dpi
            )
            documents.set_status(self.db, doc_id, EXTRACTED, note=None, done=total, total=total)
            log.info("Extracted %s (%d pages)", doc_id, total)
        except _Stopping:
            log.info("Stopped while reading %s; it will resume on the next start", doc_id)
        except DocumentGone:
            log.info("Document %s was deleted while it was being read", doc_id)
        except Exception as exc:  # noqa: BLE001 - recorded on the document for the user to see
            log.exception("Failed to read %s", doc_id)
            try:
                documents.set_status(self.db, doc_id, FAILED, error=str(exc) or type(exc).__name__)
            except DocumentGone:
                pass
