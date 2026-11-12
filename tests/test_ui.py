"""Browser tests of the web UI: headless Chromium against the real app (fake Ollama and AWS behind it).

Set SCREENSHOT_DIR to also save screenshots.
"""

import os
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright  # noqa: E402

from tests.fake_ollama import FakeOllama  # noqa: E402
from tests.pdf_factory import build_sample_pdf  # noqa: E402

SHOTS = os.environ.get("SCREENSHOT_DIR")
# The browser logs every refused request as a console error; these refusals are expected in some tests.
EXPECTED_REFUSALS = ("status of 400", "status of 409", "status of 413")
READY_TIMEOUT = 45_000  # ms; reading the sample PDF takes a few seconds


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def page(browser, live_server):
    context = browser.new_context(base_url=live_server, viewport={"width": 1280, "height": 860})
    page = context.new_page()
    problems = []
    page.on("pageerror", lambda error: problems.append(f"page error: {error}"))
    page.on(
        "console",
        lambda msg: problems.append(f"console: {msg.text}")
        if msg.type == "error" and not any(refusal in msg.text for refusal in EXPECTED_REFUSALS)
        else None,
    )
    page.goto("/")
    expect(page.locator("body[data-ready=true]")).to_be_attached()
    yield page
    context.close()
    assert problems == [], problems


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    return build_sample_pdf(tmp_path / "annual-report.pdf")


def shot(page, name, full_page=True):
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/{name}.png", full_page=full_page)


def upload_and_wait(page, pdf, extractor="none"):
    page.locator("#upload-box").evaluate("box => { box.open = true; }")
    page.check(f"input[name=extractor][value={extractor}]")
    page.set_input_files("#file-input", str(pdf))
    row = page.locator(".doc", has_text=pdf.name)
    expect(row.locator(".doc-status")).to_have_text("Ready", timeout=READY_TIMEOUT)
    return row


# ---- Header and Documents panel ----


def test_models_are_listed_with_what_they_can_do(page):
    options = page.locator("#model-select option")
    expect(options).to_have_count(3)
    expect(options.nth(0)).to_have_text("llama3.2:3b · 3B")
    expect(options.nth(1)).to_have_text("qwen2.5vl:3b · 3B · reads images")
    expect(options.nth(2)).to_contain_text("Qwen2.5-VL (AWS): not set up")
    expect(options.nth(2)).to_be_disabled()
    expect(page.locator("#model-note")).to_have_text("Text only")
    expect(page.locator("#banner")).to_be_hidden()


def test_model_choice_is_remembered(page):
    page.select_option("#model-select", "ollama:qwen2.5vl:3b")
    expect(page.locator("#model-note")).to_have_text("Can also look at chart images")

    page.reload()

    expect(page.locator("#model-select")).to_have_value("ollama:qwen2.5vl:3b")


def test_unavailable_extractor_is_disabled_with_the_reason(page):
    expect(page.locator("input[name=extractor][value=local]")).to_be_checked()
    expect(page.locator("input[name=extractor][value=aws]")).to_be_disabled()
    expect(page.locator(".extractor.unavailable")).to_contain_text("AWS keys are not set")
    expect(page.locator("#documents-empty")).to_be_visible()
    shot(page, "01-empty")


def test_upload_shows_progress_and_becomes_ready(page, sample_pdf):
    row = upload_and_wait(page, sample_pdf)

    expect(row.locator(".doc-meta")).to_contain_text("5 pages")
    expect(row.locator(".doc-meta")).to_contain_text("Text only")
    expect(page.locator("#documents-count")).to_have_text("1")
    expect(page.locator("#documents-empty")).to_be_hidden()


def test_details_show_pages_with_their_text_and_figures(page, sample_pdf):
    upload_and_wait(page, sample_pdf).locator(".doc-main").click()
    dialog = page.locator("dialog#details")
    expect(dialog).to_be_visible()
    expect(dialog.locator("#details-title")).to_have_text("annual-report.pdf")
    expect(dialog.locator(".page-thumb")).to_have_count(5)
    expect(dialog.locator(".page-thumb", has_text="Page 5")).to_contain_text("scanned")
    expect(dialog.locator(".details-summary")).to_contain_text("2 figures, 1 table, 3 text passages")
    expect(dialog.locator(".details-summary")).to_contain_text("2 (2 unread)")
    shot(page, "02-details-pages")

    dialog.locator(".page-thumb", has_text="Page 2").click()
    expect(dialog.locator(".page-text pre")).to_contain_text("|Retail|1,200|1,350|+12.5%|")
    dialog.get_by_role("button", name="Next page").click()
    expect(dialog.locator(".page-view-head strong")).to_have_text("Page 3 of 5")
    shot(page, "03-details-page-text")

    dialog.get_by_role("tab", name="Figures (2)").click()
    expect(dialog.locator(".figure")).to_have_count(2)
    expect(dialog.locator(".figure").first).to_contain_text("Figure 1: Revenue by year (AED million)")
    expect(dialog.locator(".figure").first).to_contain_text("Not read by a vision model")

    dialog.get_by_role("button", name="Close").click()
    expect(dialog).to_be_hidden()


def test_charts_read_by_the_local_model_show_their_descriptions(page, sample_pdf):
    upload_and_wait(page, sample_pdf, extractor="local").locator(".doc-main").click()
    dialog = page.locator("dialog#details")

    dialog.get_by_role("tab", name="Figures (2)").click()

    first = dialog.locator(".figure").first
    expect(first.locator(".badge")).to_have_text("chart")
    expect(first).to_contain_text("read by qwen2.5vl:3b")
    expect(first.locator("pre")).to_contain_text("Type: chart")
    shot(page, "04-details-figures")


def test_charts_can_be_read_again(page, sample_pdf):
    upload_and_wait(page, sample_pdf).locator(".doc-main").click()
    dialog = page.locator("dialog#details")

    dialog.get_by_role("button", name="Read charts again").click()

    expect(page.locator(".toast", has_text="Reading the charts again.")).to_be_visible()
    expect(dialog.locator(".details-head .doc-status")).to_have_text("Ready", timeout=READY_TIMEOUT)
    dialog.get_by_role("tab", name="Figures (2)").click()
    expect(dialog.locator(".figure").first.locator(".badge")).to_have_text("chart")


def test_delete_asks_first_then_removes_the_document(page, sample_pdf):
    upload_and_wait(page, sample_pdf).locator(".doc-main").click()
    page.once("dialog", lambda confirm: confirm.accept())

    page.locator("dialog#details").get_by_role("button", name="Delete").click()

    expect(page.locator(".doc")).to_have_count(0)
    expect(page.locator("#documents-empty")).to_be_visible()
    expect(page.locator(".toast", has_text="Deleted annual-report.pdf")).to_be_visible()


def test_files_that_are_not_pdfs_are_refused(page, tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello")
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("not really a pdf")

    page.set_input_files("#file-input", str(text_file))
    expect(page.locator(".toast-error", has_text="notes.txt isn't a PDF")).to_be_visible()

    page.set_input_files("#file-input", str(fake_pdf))
    expect(page.locator(".toast-error", has_text="fake.pdf")).to_contain_text("This file isn't a PDF.")
    expect(page.locator(".doc")).to_have_count(0)


@pytest.mark.parametrize("fake_ollama", [FakeOllama(models={"llama3.2:3b": ["completion"]})])
def test_banner_says_how_to_fix_a_missing_embedding_model(page):
    banner = page.locator("#banner")

    expect(banner).to_be_visible()
    expect(banner).to_contain_text("ollama pull nomic-embed-text")
    expect(page.locator("input[name=extractor][value=local]")).to_be_disabled()  # no vision model either


def test_layout_in_dark_mode_and_on_a_narrow_screen(browser, live_server, sample_pdf):
    """Screenshots for a visual check; the assertions only make sure nothing breaks."""
    context = browser.new_context(base_url=live_server, viewport={"width": 1280, "height": 860}, color_scheme="dark")
    page = context.new_page()
    page.goto("/")
    expect(page.locator("body[data-ready=true]")).to_be_attached()
    upload_and_wait(page, sample_pdf)
    shot(page, "05-dark")

    page.set_viewport_size({"width": 390, "height": 844})
    expect(page.locator(".doc")).to_be_visible()
    width = page.evaluate("document.documentElement.scrollWidth")
    assert width <= 390, f"page scrolls sideways on a phone ({width}px wide)"
    shot(page, "06-narrow")
    context.close()
