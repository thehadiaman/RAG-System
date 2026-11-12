"""Cuts a document's pages and figures into search-sized chunks.

- Text follows the page's own structure: a heading starts a new section, and
  paragraphs are packed into chunks with a little overlap so a thought that
  spans two chunks can still be found.
- Tables are never mixed with text. A table that is too big is split by rows,
  with the header repeated so every piece makes sense on its own.
- Each figure is one chunk: the vision model's description plus the PDF's own
  caption and text inside the figure (exact numbers the model may have missed).

Sizes are in characters; about 4 characters make one token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingest.pdf import TABLE_SEPARATOR
from app.ingest.vision import page_heading

MAX_TEXT_CHARS = 1600
OVERLAP_CHARS = 200
MAX_TABLE_CHARS = 2400
MIN_FIGURE_TEXT_CHARS = 20  # a figure with less text than this and no description says nothing
UNREAD_FIGURE = "Figure (not read by a vision model)."

HEADING = re.compile(r"^#{1,6}\s+(.*)$")
PICTURE_TEXT = re.compile(r"^\[Picture text: .*\]$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
FOOTNOTE_MARK = re.compile(r"<sup>.*?</sup>")
HTML_TAG = re.compile(r"<[^>]+>")


@dataclass
class Chunk:
    page_no: int
    kind: str  # "text", "table" or "figure"
    section: str  # the heading this chunk sits under
    text: str
    figure_id: str | None = None
    image_path: str | None = None


@dataclass
class Block:
    kind: str  # "heading", "table", "text" or "picture_text"
    text: str


# ---- Whole document ----


def chunk_document(pages: list[dict], figures: list[dict]) -> list[Chunk]:
    """`pages` and `figures` are database rows, in page order."""
    figures_by_page: dict[int, list[dict]] = {}
    for figure in figures:
        figures_by_page.setdefault(figure["page_no"], []).append(figure)

    chunks: list[Chunk] = []
    section = ""
    for page in pages:
        page_figures = figures_by_page.get(page["page_no"], [])
        figure_chunks = []
        figure_section = clean_heading(page_heading(page["text"])) or section
        for figure in page_figures:
            chunk = figure_chunk(figure, figure_section)
            if chunk:
                figure_chunks.append(chunk)

        # Once every figure on the page has a description, the raw text the PDF had
        # inside those pictures would only repeat it.
        described = bool(page_figures) and all(f["description"] for f in page_figures)
        text_chunks, section = chunk_page(
            page["page_no"],
            page["text"],
            section,
            drop_picture_text=described,
            keep_lone_headings=not figure_chunks,
        )
        chunks += text_chunks + figure_chunks
    return chunks


# ---- One page of text ----


def split_blocks(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []
    table: list[str] = []

    def end_paragraph() -> None:
        if paragraph:
            blocks.append(Block("text", "\n".join(paragraph)))
            paragraph.clear()

    def end_table() -> None:
        if table:
            blocks.append(Block("table", "\n".join(table)))
            table.clear()

    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("|"):
            end_paragraph()
            table.append(line)
            continue
        end_table()
        if not line:
            end_paragraph()
        elif HEADING.match(line):
            end_paragraph()
            blocks.append(Block("heading", line))
        elif PICTURE_TEXT.match(line):
            end_paragraph()
            blocks.append(Block("picture_text", line))
        else:
            paragraph.append(line)
    end_paragraph()
    end_table()
    return blocks


def chunk_page(
    page_no: int,
    markdown: str,
    section: str,
    drop_picture_text: bool = False,
    keep_lone_headings: bool = True,
) -> tuple[list[Chunk], str]:
    """Chunks for one page, and the section in force at its end (it carries to the next page)."""
    chunks: list[Chunk] = []
    parts: list[str] = []  # pieces of the text chunk being built
    parts_section = section
    lone_headings: list[str] = []

    def flush() -> str:
        """Emit the chunk being built. Returns its text, or "" if nothing was emitted."""
        nonlocal parts
        text = "\n\n".join(parts).strip()
        only_headings = all(HEADING.match(part) for part in parts)
        parts = []
        if not text:
            return ""
        if only_headings:
            # A heading with nothing under it lives on as the section of what follows.
            lone_headings.append(text)
            return ""
        chunks.append(Chunk(page_no, "text", parts_section, text))
        return text

    def add_text(text: str) -> None:
        nonlocal parts, parts_section
        for piece in split_long_text(text):
            size = sum(len(part) + 2 for part in parts)
            if parts and size + len(piece) > MAX_TEXT_CHARS:
                finished = flush()
                parts_section = section
                overlap = tail(finished)
                parts = [overlap] if overlap else []
            parts.append(piece)

    for block in split_blocks(markdown):
        if block.kind == "heading":
            flush()
            section = clean_heading(block.text)
            parts_section = section
            parts.append(block.text)
        elif block.kind == "table":
            flush()
            parts_section = section
            chunks += [Chunk(page_no, "table", section, piece) for piece in split_table(block.text)]
        elif block.kind == "picture_text":
            if not drop_picture_text:
                add_text(block.text)
        else:
            add_text(block.text)
    flush()

    if not chunks and lone_headings and keep_lone_headings:
        # A page with only a title (a cover or divider slide) is still worth finding.
        chunks.append(Chunk(page_no, "text", section, "\n\n".join(lone_headings)))
    return chunks, section


def clean_heading(line: str) -> str:
    """A heading as plain text: no #, bold marks or footnote markers like <sup>1</sup>."""
    match = HEADING.match(line)
    text = match.group(1) if match else line
    text = FOOTNOTE_MARK.sub("", text)
    text = HTML_TAG.sub("", text)
    return text.replace("**", "").strip()[:200]


def split_long_text(text: str) -> list[str]:
    """Break a paragraph longer than a chunk at sentence ends."""
    if len(text) <= MAX_TEXT_CHARS:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(text):
        while len(sentence) > MAX_TEXT_CHARS:  # one enormous "sentence": cut it hard
            pieces.append(sentence[:MAX_TEXT_CHARS])
            sentence = sentence[MAX_TEXT_CHARS:]
        if current and len(current) + 1 + len(sentence) > MAX_TEXT_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def tail(text: str) -> str:
    """The last sentence or two of a chunk, repeated at the start of the next one."""
    if not text:
        return ""
    last_paragraph = text.split("\n\n")[-1]
    if HEADING.match(last_paragraph):
        return ""
    overlap = ""
    for sentence in reversed(SENTENCE_END.split(last_paragraph)):
        if len(sentence) + len(overlap) + 1 > OVERLAP_CHARS:
            break
        overlap = f"{sentence} {overlap}".strip()
    return overlap


# ---- Tables ----


def split_table(table: str) -> list[str]:
    if len(table) <= MAX_TABLE_CHARS:
        return [table]
    lines = table.split("\n")
    has_header = len(lines) > 1 and TABLE_SEPARATOR.match(lines[1])
    header = lines[:2] if has_header else []
    rows = lines[2:] if has_header else lines
    header_size = sum(len(line) + 1 for line in header)

    pieces: list[str] = []
    current: list[str] = []
    size = header_size
    for row in rows:
        if current and size + len(row) + 1 > MAX_TABLE_CHARS:
            pieces.append("\n".join(header + current))
            current, size = [], header_size
        current.append(row)
        size += len(row) + 1
    if current:
        pieces.append("\n".join(header + current))
    return pieces


# ---- Figures ----


def figure_chunk(figure: dict, section: str) -> Chunk | None:
    """One chunk per figure, or None when there is nothing worth searching."""
    if figure.get("kind") == "decorative":
        return None
    description = (figure.get("description") or "").strip()
    caption = (figure.get("caption") or "").strip()
    inside = (figure.get("text_inside") or "").strip()

    if not description and len(caption) + len(inside) < MIN_FIGURE_TEXT_CHARS:
        return None

    parts = [description or UNREAD_FIGURE]
    if caption:
        parts.append(f"Caption: {caption}")
    if inside:
        parts.append(f"Text in the figure (from the PDF): {inside}")
    return Chunk(
        page_no=figure["page_no"],
        kind="figure",
        section=section,
        text="\n\n".join(parts),
        figure_id=figure["id"],
        image_path=figure["image_path"],
    )


# ---- Text sent to the embedding model ----

MAX_EMBED_CHARS = 6000  # stay inside the embedding model's context


def embedding_text(filename: str, chunk: Chunk) -> str:
    """The chunk with a short header saying where it is from, so it stands on its own."""
    title = filename.rsplit(".", 1)[0]
    header = f"{title} › {chunk.section}" if chunk.section else title
    # The same placeholder on every unread figure would pull their vectors together.
    text = chunk.text.removeprefix(UNREAD_FIGURE).lstrip()
    return f"{header}\n{text}"[:MAX_EMBED_CHARS]
