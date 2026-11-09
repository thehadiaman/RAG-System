import pymupdf
import pytest

from app.ingest.pdf import MAX_FIGURE_PIXELS, PdfError, check_pdf, clean_markdown, extract_document
from tests.pdf_factory import build_password_pdf, build_sample_pdf, build_text_pdf


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    """The sample PDF, extracted once for all tests in this file."""
    folder = tmp_path_factory.mktemp("sample")
    pdf = build_sample_pdf(folder / "sample.pdf")
    pages = {}

    def on_page(result, preview, figure_paths):
        pages[result.page_no] = (result, preview, figure_paths)

    count = extract_document(pdf, folder / "out", on_page)
    return count, pages


def test_every_page_is_reported(extracted):
    count, pages = extracted
    assert count == 5
    assert sorted(pages) == [1, 2, 3, 4, 5]


def test_text_page_keeps_its_heading_and_has_no_figures(extracted):
    result, _, _ = extracted[1][1]
    assert result.text.startswith("# Annual Report 2025")
    assert "offices in Dubai and Abu Dhabi" in result.text
    assert result.figures == []
    assert not result.needs_ocr


def test_table_becomes_a_markdown_table_not_a_figure(extracted):
    result, _, _ = extracted[1][2]
    assert result.table_count == 1
    assert "|Retail|1,200|1,350|+12.5%|" in result.text
    assert result.figures == []


def test_chart_drawn_with_shapes_is_found_with_its_caption(extracted):
    result, _, figure_paths = extracted[1][3]
    assert len(result.figures) == 1
    figure = result.figures[0]
    assert figure.caption == "Figure 1: Revenue by year (AED million)"
    assert "2018" in figure.text_inside and "2025" in figure.text_inside
    assert figure_paths[0].is_file()


def test_photo_is_found_but_tiny_logo_and_repeated_banner_are_not(extracted):
    result, _, _ = extracted[1][4]
    assert len(result.figures) == 1
    assert result.figures[0].caption == "Figure 2: The new Dubai office"
    x0, y0, x1, y1 = result.figures[0].bbox
    assert y1 < 500  # the photo, not the logo or banner near the bottom


def test_scanned_page_is_flagged_for_the_vision_model(extracted):
    result, _, _ = extracted[1][5]
    assert result.needs_ocr
    assert result.char_count == 0
    assert result.figures == []


def test_previews_and_figure_crops_are_written(extracted):
    _, pages = extracted
    for result, preview, figure_paths in pages.values():
        assert preview.suffix == ".jpg" and preview.stat().st_size > 0
        for path in figure_paths:
            pix = pymupdf.Pixmap(str(path))
            assert max(pix.width, pix.height) <= MAX_FIGURE_PIXELS


def test_check_pdf_counts_pages(tmp_path):
    assert check_pdf(build_text_pdf(tmp_path / "ok.pdf")) == 1


def test_check_pdf_rejects_other_files(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_text("just some text")

    with pytest.raises(PdfError, match="isn't a PDF"):
        check_pdf(path)


def test_check_pdf_rejects_password_protected_files(tmp_path):
    with pytest.raises(PdfError, match="password protected"):
        check_pdf(build_password_pdf(tmp_path / "locked.pdf"))


def test_picture_text_is_kept_in_readable_form():
    markdown = "Intro\n\n\n\n<!-- Start of picture text -->Revenue<br>47%<br> <br><!-- End of picture text -->\n"

    assert clean_markdown(markdown) == "Intro\n\n[Picture text: Revenue · 47%]"
