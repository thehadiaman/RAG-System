"""Reads a PDF page by page: markdown text, tables, page previews, figures and scanned pages.

Figures are found from three signals, because each one misses some cases:
- picture areas from pymupdf4llm's layout model,
- images embedded in the page,
- dense groups of vector drawings (many reports draw their charts as shapes).
Small icons, thin banners, logos repeated on many pages and background images
are dropped, and overlapping areas are merged into one figure.

Everything here is blocking. PyMuPDF must never run on two threads at once,
so every use of it goes through PDF_LOCK, held one page at a time so that
other work (an upload check, a vision render) only waits briefly.
"""

from __future__ import annotations

import collections
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import pymupdf4llm

log = logging.getLogger(__name__)

# Figure size limits, in PDF points (72 points = 1 inch).
MIN_FIGURE_SIDE = 60
MIN_FIGURE_AREA = 0.025  # share of the page
MAX_FIGURE_ASPECT = 8  # wider or taller than this is a banner or rule
MAX_FIGURES_PER_PAGE = 8
FIGURE_PADDING = 12  # include nearby axis labels and legends

# A group of drawn shapes counts as a chart when it has at least this many
# drawing items and more items than words (text panels have the reverse).
CHART_MIN_DRAWING_ITEMS = 20

# An image on at least this share of pages (and at least 3) is a logo or footer.
REPEATED_IMAGE_SHARE = 0.25

# Scanned page: almost no real text, and mostly one picture (or text drawn as shapes).
SCANNED_MAX_CHARS = 20
SCANNED_MIN_IMAGE_COVER = 0.5
SCANNED_MIN_DRAWING_ITEMS = 200

# Charts and photos have hundreds of colours; plain bars and panels have a handful.
PLAIN_BLOCK_MAX_COLOURS = 16

# Background image: covers most of the page and has plenty of real text on top.
BACKGROUND_MIN_COVER = 0.8
BACKGROUND_MIN_CHARS = 200

MAX_FIGURE_PIXELS = 1600  # longest side of a cropped figure

CAPTION_WORDS = re.compile(r"^\s*(figure|fig\.|chart|graph|exhibit|table|source|note)\b", re.I)
PICTURE_TEXT = re.compile(r"<!-- Start of picture text -->(.*?)<!-- End of picture text -->", re.S)
OMITTED_PICTURE = re.compile(r"\*\*==> picture .*? intentionally omitted <==\*\*")
TABLE_SEPARATOR = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+\s*$", re.M)


PDF_LOCK = threading.Lock()


class PdfError(Exception):
    """The file isn't a PDF we can read."""


@dataclass
class Figure:
    bbox: tuple[float, float, float, float]
    source: str  # "layout", "picture" or "drawing": which signal found it
    caption: str = ""
    text_inside: str = ""


@dataclass
class PageResult:
    page_no: int  # 1-based
    text: str  # markdown
    char_count: int
    table_count: int
    needs_ocr: bool
    figures: list[Figure] = field(default_factory=list)


# ---- Opening and validating ----


def open_pdf(path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - any parser failure means "can't read it"
        raise PdfError(f"This file couldn't be read as a PDF ({exc}).") from exc
    if doc.needs_pass:
        doc.close()
        raise PdfError("This PDF is password protected. Remove the password and upload it again.")
    if doc.page_count == 0:
        doc.close()
        raise PdfError("This PDF has no pages.")
    return doc


def check_pdf(path: Path) -> int:
    """Raises PdfError if the file isn't a readable PDF. Returns the page count."""
    with path.open("rb") as f:
        if not f.read(1024).lstrip().startswith(b"%PDF-"):
            raise PdfError("This file isn't a PDF.")
    with PDF_LOCK:
        doc = open_pdf(path)
        try:
            return doc.page_count
        finally:
            doc.close()


# ---- Whole document ----


def extract_document(
    pdf_path: Path,
    out_dir: Path,
    on_page: Callable[[PageResult, Path, list[Path]], None],
    preview_dpi: int = 100,
    figure_dpi: int = 200,
) -> int:
    """Extract every page, calling `on_page(result, preview_path, figure_paths)` after each one.

    Returns the number of pages.
    """
    pages_dir = out_dir / "pages"
    figures_dir = out_dir / "figures"
    pages_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with PDF_LOCK:
        doc = open_pdf(pdf_path)
        repeated = repeated_image_xrefs(doc)
    try:
        for number in range(doc.page_count):
            with PDF_LOCK:
                page = doc[number]
                result = extract_page(doc, page, repeated)

                preview_path = pages_dir / f"{result.page_no:04d}.jpg"
                render_page(page, preview_path, preview_dpi)

                figure_paths = []
                for i, figure in enumerate(result.figures, start=1):
                    path = figures_dir / f"p{result.page_no:04d}-{i:02d}.png"
                    crop(page, pymupdf.Rect(figure.bbox), path, figure_dpi)
                    figure_paths.append(path)

            on_page(result, preview_path, figure_paths)
        return doc.page_count
    finally:
        with PDF_LOCK:
            doc.close()


def repeated_image_xrefs(doc: pymupdf.Document) -> set[int]:
    """Images placed on many pages: logos, footers, slide backgrounds."""
    pages_per_image: collections.Counter[int] = collections.Counter()
    for page in doc:
        for xref in {info["xref"] for info in page.get_image_info(xrefs=True) if info["xref"]}:
            pages_per_image[xref] += 1
    threshold = max(3, doc.page_count * REPEATED_IMAGE_SHARE)
    return {xref for xref, count in pages_per_image.items() if count >= threshold}


# ---- One page ----


def extract_page(doc: pymupdf.Document, page: pymupdf.Page, repeated_xrefs: set[int]) -> PageResult:
    chunk = pymupdf4llm.to_markdown(
        doc,
        pages=[page.number],
        page_chunks=True,
        use_ocr=False,  # scanned pages go to the vision model instead
        header=False,  # drop running headers and footers
        footer=False,
        show_progress=False,
    )[0]
    text = clean_markdown(chunk["text"])
    plain = page.get_text().strip()
    needs_ocr = is_scanned(page, len(plain))

    figures = [] if needs_ocr else find_figures(page, chunk.get("page_boxes", []), repeated_xrefs)
    for figure in figures:
        rect = pymupdf.Rect(figure.bbox)
        figure.caption = caption_near(page, rect)
        figure.text_inside = _squash(page.get_text("text", clip=rect))[:2000]

    return PageResult(
        page_no=page.number + 1,
        text=text,
        char_count=len(plain),
        table_count=len(TABLE_SEPARATOR.findall(text)),
        needs_ocr=needs_ocr,
        figures=figures,
    )


def clean_markdown(markdown: str) -> str:
    """Keep text found inside pictures (it holds chart labels and numbers) in a readable form."""

    def picture_text(match: re.Match) -> str:
        parts = [part.strip() for part in match.group(1).split("<br>") if part.strip()]
        return f"[Picture text: {' · '.join(parts)}]" if parts else ""

    markdown = PICTURE_TEXT.sub(picture_text, markdown)
    markdown = OMITTED_PICTURE.sub("", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def is_scanned(page: pymupdf.Page, char_count: int) -> bool:
    if char_count >= SCANNED_MAX_CHARS:
        return False
    page_area = page.rect.get_area()
    image_cover = max(
        ((pymupdf.Rect(info["bbox"]) & page.rect).get_area() / page_area for info in page.get_image_info()),
        default=0.0,
    )
    if image_cover >= SCANNED_MIN_IMAGE_COVER:
        return True
    drawn_items = sum(len(d["items"]) for d in page.get_drawings())
    return drawn_items >= SCANNED_MIN_DRAWING_ITEMS


# ---- Finding figures ----


def find_figures(page: pymupdf.Page, layout_boxes: list[dict], repeated_xrefs: set[int]) -> list[Figure]:
    candidates: list[tuple[pymupdf.Rect, str]] = []
    repeated_areas: list[pymupdf.Rect] = []

    for info in page.get_image_info(xrefs=True):
        rect = pymupdf.Rect(info["bbox"]) & page.rect
        if rect.is_empty:
            continue
        if info["xref"] in repeated_xrefs:
            repeated_areas.append(rect)
        elif not _is_background(page, rect):
            candidates.append((rect, "picture"))

    for box in layout_boxes:
        if box.get("class") == "picture":
            candidates.append((pymupdf.Rect(box["bbox"]), "layout"))

    drawings = page.get_drawings()
    for rect in page.cluster_drawings(drawings=drawings):
        if _looks_like_chart(page, rect, drawings):
            candidates.append((rect, "drawing"))

    # Drop tables (they are read as text) and anything sitting on a repeated logo or banner.
    excluded = _table_areas(page, layout_boxes) + repeated_areas
    kept = [
        (rect, source)
        for rect, source in candidates
        if _big_enough(page, rect) and not any(_mostly_inside(rect, area) for area in excluded)
    ]
    merged = [item for item in _merge_overlapping(kept) if not _is_plain_block(page, item[0])]
    merged.sort(key=lambda item: item[0].get_area(), reverse=True)
    merged = merged[:MAX_FIGURES_PER_PAGE]
    merged.sort(key=lambda item: (round(item[0].y0), item[0].x0))  # reading order

    return [
        Figure(bbox=_rounded((rect + (-FIGURE_PADDING, -FIGURE_PADDING, FIGURE_PADDING, FIGURE_PADDING)) & page.rect), source=source)
        for rect, source in merged
    ]


def _looks_like_chart(page: pymupdf.Page, rect: pymupdf.Rect, drawings: list[dict]) -> bool:
    grown = rect + (-1, -1, 1, 1)
    items = sum(len(d["items"]) for d in drawings if pymupdf.Rect(d["rect"]) in grown)
    if items < CHART_MIN_DRAWING_ITEMS:
        return False
    words = len(page.get_text("words", clip=rect))
    return items >= words


def _table_areas(page: pymupdf.Page, layout_boxes: list[dict]) -> list[pymupdf.Rect]:
    areas = [pymupdf.Rect(box["bbox"]) for box in layout_boxes if box.get("class") == "table"]
    try:
        areas += [pymupdf.Rect(table.bbox) for table in page.find_tables().tables]
    except Exception as exc:  # noqa: BLE001 - table finding is a best-effort filter
        log.debug("find_tables failed on page %d: %s", page.number + 1, exc)
    return [area for area in areas if _big_enough(page, area)]


def _is_background(page: pymupdf.Page, rect: pymupdf.Rect) -> bool:
    if rect.get_area() / page.rect.get_area() < BACKGROUND_MIN_COVER:
        return False
    return len(page.get_text("text", clip=rect).strip()) >= BACKGROUND_MIN_CHARS


def _is_plain_block(page: pymupdf.Page, rect: pymupdf.Rect) -> bool:
    """A coloured bar or panel: almost no distinct colours in a small thumbnail."""
    thumbnail = page.get_pixmap(dpi=30, clip=rect)
    return thumbnail.color_count() < PLAIN_BLOCK_MAX_COLOURS


def _big_enough(page: pymupdf.Page, rect: pymupdf.Rect) -> bool:
    if rect.is_empty or min(rect.width, rect.height) < MIN_FIGURE_SIDE:
        return False
    if max(rect.width, rect.height) / min(rect.width, rect.height) > MAX_FIGURE_ASPECT:
        return False
    return rect.get_area() / page.rect.get_area() >= MIN_FIGURE_AREA


def _mostly_inside(rect: pymupdf.Rect, other: pymupdf.Rect, share: float = 0.5) -> bool:
    return (rect & other).get_area() >= share * rect.get_area()


def _merge_overlapping(items: list[tuple[pymupdf.Rect, str]]) -> list[tuple[pymupdf.Rect, str]]:
    """Merge areas where the overlap covers at least half of the smaller one."""
    merged = [(pymupdf.Rect(rect), source) for rect, source in items]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, source_a = merged[i]
                b, source_b = merged[j]
                smaller = min(a.get_area(), b.get_area())
                if smaller and (a & b).get_area() >= 0.5 * smaller:
                    source = source_a if a.get_area() >= b.get_area() else source_b
                    merged[i] = (a | b, source)
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def caption_near(page: pymupdf.Page, rect: pymupdf.Rect, max_gap: float = 40) -> str:
    """A caption-like line just above or below the figure, else the closest text above it."""
    nearby = []
    for x0, y0, x1, y1, text, _, block_type in page.get_text("blocks"):
        if block_type != 0 or not text.strip():
            continue
        if x1 < rect.x0 or x0 > rect.x1:  # no horizontal overlap
            continue
        # The figure box includes padding, so a caption can overlap it by that much.
        gap_above = rect.y0 - y1
        gap_below = y0 - rect.y1
        if -FIGURE_PADDING <= gap_above <= max_gap or -FIGURE_PADDING <= gap_below <= max_gap:
            above = gap_above >= -FIGURE_PADDING
            nearby.append((min(abs(gap_above), abs(gap_below)), above, _squash(text)))
    for _, _, text in sorted(nearby):
        if CAPTION_WORDS.match(text):
            return text[:300]
    above = sorted(item for item in nearby if item[1])
    return above[0][2][:300] if above else ""


# ---- Rendering ----


# Vision images are limited by their total pixel count, not their longest side.
# A model's cost depends on the area, and a side limit would shrink wide or tall
# charts until their labels are unreadable.


def page_image_for_vision(pdf_path: Path, page_no: int, max_megapixels: float) -> bytes:
    """A whole page as PNG, as sharp as the pixel budget allows (at most 200 dpi)."""
    with PDF_LOCK:
        doc = open_pdf(pdf_path)
        try:
            page = doc[page_no - 1]
            square_inches = (page.rect.width / 72) * (page.rect.height / 72)
            dpi = min(200, int((max_megapixels * 1_000_000 / square_inches) ** 0.5))
            return page.get_pixmap(dpi=dpi).tobytes("png")
        finally:
            doc.close()


def image_for_vision(path: Path, max_megapixels: float) -> bytes:
    """An image file as PNG, scaled down if it has more than `max_megapixels` pixels."""
    with PDF_LOCK:
        pix = pymupdf.Pixmap(str(path))
        pixels = pix.width * pix.height
        if pixels > max_megapixels * 1_000_000:
            scale = (max_megapixels * 1_000_000 / pixels) ** 0.5
            pix = pymupdf.Pixmap(pix, max(1, int(pix.width * scale)), max(1, int(pix.height * scale)))
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)  # drop transparency
        return pix.tobytes("png")


def render_page(page: pymupdf.Page, path: Path, dpi: int) -> None:
    page.get_pixmap(dpi=dpi).save(path, jpg_quality=80)


def crop(page: pymupdf.Page, rect: pymupdf.Rect, path: Path, dpi: int) -> None:
    longest_side_inches = max(rect.width, rect.height) / 72
    dpi = min(dpi, int(MAX_FIGURE_PIXELS / longest_side_inches))
    page.get_pixmap(dpi=dpi, clip=rect).save(path)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _rounded(rect: pymupdf.Rect) -> tuple[float, float, float, float]:
    return (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
