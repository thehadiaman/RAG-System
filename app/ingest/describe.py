"""The vision step: describe every figure and transcribe every scanned page of a document.

A failed item is retried once, then recorded and skipped, so one bad image
doesn't sink the document. After several failures in a row the step stops
(the model is probably down) and the document gets a warning instead.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.ingest.pdf import image_for_vision, page_image_for_vision
from app.ingest.vision import Describer, VisionError, figure_kind, figure_prompt, page_heading, scanned_page_prompt
from app.llm.messages import ImageInput
from app.store import documents
from app.store.db import Database
from app.store.documents import DocumentGone

log = logging.getLogger(__name__)

STOP_AFTER_FAILURES_IN_A_ROW = 3
RETRY_DELAY = 2.0
SCANNED_PAGE_MEGAPIXELS = 2.0  # whole pages need more pixels than figures to keep small text readable


@dataclass
class DescribeOutcome:
    total: int = 0
    done: int = 0  # read successfully
    failed: int = 0
    stopped_because: str | None = None  # set when the step gave up early
    interrupted: bool = False  # the app is shutting down

    @property
    def skipped(self) -> int:
        return self.total - self.done - self.failed

    def warning(self) -> str | None:
        if self.stopped_because:
            return (
                f"Stopped reading figures after repeated errors ({self.stopped_because}). "
                f"{self.failed + self.skipped} of {self.total} figures and scanned pages were not read; "
                "run the vision step again to retry."
            )
        if self.failed:
            return f"{self.failed} of {self.total} figures and scanned pages couldn't be read."
        return None


async def describe_document(
    db: Database,
    data_dir: Path,
    doc_id: str,
    describer: Describer,
    max_megapixels: float,
    on_progress: Callable[[int, int], None],
    should_stop: Callable[[], bool] = lambda: False,
    retry_delay: float | None = None,
) -> DescribeOutcome:
    retry_delay = RETRY_DELAY if retry_delay is None else retry_delay
    figures = documents.figures_to_describe(db, doc_id)
    pages = documents.scanned_pages_to_read(db, doc_id)
    outcome = DescribeOutcome(total=len(figures) + len(pages))
    if outcome.total == 0:
        return outcome

    try:
        await describer.prepare()
    except VisionError as exc:
        outcome.stopped_because = str(exc)
        return outcome

    pdf_path = data_dir / "documents" / doc_id / "original.pdf"
    headings: dict[int, str] = {}
    in_a_row = 0
    processed = 0
    gone = False

    def heading(page_no: int) -> str:
        if page_no not in headings:
            page = documents.page(db, doc_id, page_no)
            headings[page_no] = page_heading(page["text"]) if page else ""
        return headings[page_no]

    async def ask(image: ImageInput, prompt: str) -> str:
        try:
            return await describer.describe(image, prompt)
        except VisionError as exc:
            log.info("Vision call failed, retrying once: %s", exc)
            await asyncio.sleep(retry_delay)
            return await describer.describe(image, prompt)

    async def read_figure(figure: dict) -> None:
        data = await asyncio.to_thread(image_for_vision, data_dir / figure["image_path"], max_megapixels)
        prompt = figure_prompt(figure["page_no"], heading(figure["page_no"]), figure["caption"], figure["text_inside"])
        text = await ask(ImageInput(data, "png"), prompt)
        documents.save_figure_description(db, figure["id"], figure_kind(text), text, describer.label)

    async def read_page(page: dict) -> None:
        data = await asyncio.to_thread(page_image_for_vision, pdf_path, page["page_no"], SCANNED_PAGE_MEGAPIXELS)
        text = await ask(ImageInput(data, "png"), scanned_page_prompt(page["page_no"]))
        documents.save_page_transcription(db, doc_id, page["page_no"], text)

    def record_failure(kind: str, item: dict, error: str) -> None:
        if kind == "figure":
            documents.save_figure_error(db, item["id"], error)
        else:
            documents.save_page_error(db, doc_id, item["page_no"], error)

    semaphore = asyncio.Semaphore(max(1, describer.concurrency))

    async def run(kind: str, item: dict) -> None:
        nonlocal in_a_row, processed, gone
        async with semaphore:
            if gone or outcome.stopped_because or outcome.interrupted:
                return
            if should_stop():
                outcome.interrupted = True
                return
            try:
                await (read_figure(item) if kind == "figure" else read_page(item))
                outcome.done += 1
                in_a_row = 0
            except DocumentGone:
                gone = True
                return
            except Exception as exc:  # noqa: BLE001 - recorded on the item; the document carries on
                error = str(exc) or type(exc).__name__
                log.warning("Couldn't read %s on page %s: %s", kind, item["page_no"], error)
                outcome.failed += 1
                in_a_row += 1
                record_failure(kind, item, error)
                if in_a_row >= STOP_AFTER_FAILURES_IN_A_ROW:
                    outcome.stopped_because = error
            processed += 1
            try:
                on_progress(processed, outcome.total)
            except DocumentGone:
                gone = True

    work = [("page", page) for page in pages] + [("figure", figure) for figure in figures]
    await asyncio.gather(*(run(kind, item) for kind, item in work))

    if gone:
        raise DocumentGone(doc_id)
    return outcome
