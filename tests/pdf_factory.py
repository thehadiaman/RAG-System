"""Builds small PDFs with known content for the extraction tests."""

from pathlib import Path

import pymupdf

A4 = (595, 842)


def gradient_png(width: int, height: int) -> bytes:
    """A colourful image (a plain one would be dropped as a decorative block)."""
    samples = bytes(
        value
        for y in range(height)
        for x in range(width)
        for value in (x * 255 // width, y * 255 // height, (x + y) * 255 // (width + height))
    )
    return pymupdf.Pixmap(pymupdf.csRGB, width, height, samples, False).tobytes("png")


def add_heading_and_text(page: pymupdf.Page) -> None:
    page.insert_text((72, 100), "Annual Report 2025", fontsize=24)
    body = (
        "Revenue grew strongly this year thanks to higher demand in the Gulf region. "
        "Operating margin improved and the company opened new offices in Dubai and Abu Dhabi. "
    ) * 4
    page.insert_textbox(pymupdf.Rect(72, 130, 523, 400), body, fontsize=11)


def add_table(page: pymupdf.Page) -> None:
    page.insert_text((72, 90), "Results by segment", fontsize=16)
    rows = [
        ["Segment", "2024", "2025", "Change"],
        ["Retail", "1,200", "1,350", "+12.5%"],
        ["Wholesale", "800", "860", "+7.5%"],
        ["Online", "400", "520", "+30.0%"],
        ["Total", "2,400", "2,730", "+13.8%"],
    ]
    x0, y0, col_w, row_h = 72, 120, 110, 26
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x0 + c * col_w, y0 + r * row_h, x0 + (c + 1) * col_w, y0 + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 6, rect.y1 - 8), cell, fontsize=10)


def add_bar_chart(page: pymupdf.Page) -> None:
    """A bar chart drawn with shapes, like charts exported from Excel or PowerPoint."""
    page.insert_text((72, 90), "Revenue trend", fontsize=16)
    left, bottom, width, height = 110, 520, 380, 320
    values = [120, 150, 135, 180, 210, 240, 230, 270]
    for i in range(6):  # gridlines and value labels
        y = bottom - i * height / 5
        page.draw_line((left, y), (left + width, y), color=(0.8, 0.8, 0.8), width=0.5)
        page.insert_text((left - 30, y + 3), str(i * 60), fontsize=8)
    page.draw_line((left, bottom), (left, bottom - height), color=(0, 0, 0), width=1)
    page.draw_line((left, bottom), (left + width, bottom), color=(0, 0, 0), width=1)
    bar_w = width / len(values) * 0.6
    for i, value in enumerate(values):
        x = left + 12 + i * width / len(values)
        top = bottom - value / 300 * height
        page.draw_rect(pymupdf.Rect(x, top, x + bar_w, bottom), color=None, fill=(0.1, 0.3 + i * 0.07, 0.7))
        page.draw_line((x + bar_w / 2, bottom), (x + bar_w / 2, bottom + 4), color=(0, 0, 0), width=0.5)
        page.insert_text((x, bottom + 14), str(2018 + i), fontsize=8)
    page.insert_text((left, bottom + 40), "Figure 1: Revenue by year (AED million)", fontsize=10)


def add_photo_and_logo(page: pymupdf.Page) -> None:
    page.insert_text((72, 90), "Our new headquarters", fontsize=16)
    page.insert_image(pymupdf.Rect(72, 120, 472, 370), stream=gradient_png(400, 250))
    page.insert_text((72, 390), "Figure 2: The new Dubai office", fontsize=10)
    page.insert_image(pymupdf.Rect(500, 760, 525, 785), stream=gradient_png(25, 25))  # tiny logo
    page.insert_textbox(pymupdf.Rect(72, 420, 523, 600), "The office opened in March. " * 8, fontsize=11)


def add_scanned_page(page: pymupdf.Page) -> None:
    """No text layer at all, just one big picture of a page."""
    page.insert_image(page.rect, stream=gradient_png(300, 420))


def build_sample_pdf(path: Path) -> Path:
    """Five pages: text, table, bar chart, photo, scanned. Every page has the same banner image."""
    doc = pymupdf.open()
    banner = gradient_png(240, 120)
    banner_xref = 0
    for add in (add_heading_and_text, add_table, add_bar_chart, add_photo_and_logo, add_scanned_page):
        page = doc.new_page(width=A4[0], height=A4[1])
        add(page)
        rect = pymupdf.Rect(420, 700, 540, 760)  # same image on every page: should be ignored
        banner_xref = page.insert_image(rect, stream=banner) if not banner_xref else page.insert_image(rect, xref=banner_xref)
    doc.save(path)
    return path


def build_text_pdf(path: Path, text: str = "Hello from a tiny PDF.") -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    return path


def build_password_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Secret")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    return path
