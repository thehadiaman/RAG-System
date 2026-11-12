"""Browser tests of the chat: asking, streaming, citations, sources, stopping and saved chats."""

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect  # noqa: E402

from app.answer import NOTHING_INDEXED  # noqa: E402
from tests.fake_ollama import REWRITE_REPLY  # noqa: E402
from tests.test_ui import browser, page, sample_pdf, shot, upload_and_wait  # noqa: E402, F401

ANSWER_TIMEOUT = 20_000


def ask(page, question):
    page.fill("#question", question)
    page.keyboard.press("Enter")


def last_answer(page):
    return page.locator(".msg-assistant").last


def wait_for_answer(page):
    answer = last_answer(page)
    expect(answer.locator(".msg-meta")).to_be_visible(timeout=ANSWER_TIMEOUT)
    return answer


@pytest.fixture
def ready(page, sample_pdf):
    upload_and_wait(page, sample_pdf)
    return page


def test_question_is_answered_with_clickable_citations(ready):
    page = ready
    ask(page, "How did the retail segment do?")

    expect(page.locator(".msg-user .bubble")).to_have_text("How did the retail segment do?")
    answer = wait_for_answer(page)
    expect(answer.locator(".answer")).to_contain_text("Retail revenue rose to 1,350")
    expect(answer.locator(".answer button.cite")).to_have_count(3)  # [1], [1][2]
    expect(answer.locator(".source-cards").first.locator(".source-card.cited")).to_have_count(2)
    expect(answer.locator(".msg-meta")).to_contain_text("llama3.2:3b · ")
    expect(answer.locator(".msg-meta")).not_to_contain_text("ollama:")

    # The new chat is saved, titled from the question, and open.
    item = page.locator(".chat-item.active")
    expect(item).to_contain_text("How did the retail segment do?")
    expect(page.locator("#chat-title")).to_have_text("How did the retail segment do?")
    assert "#chat=" in page.url
    shot(page, "07-chat-answer")

    answer.locator(".answer button.cite").first.click()
    panel = page.locator("#source-panel")
    expect(panel).to_be_visible()
    expect(panel.locator("h2")).to_have_text("Source 1")
    expect(panel.locator(".source-where")).to_contain_text("annual-report.pdf · page")
    expect(panel.locator("pre.passage")).not_to_be_empty()
    expect(panel).not_to_contain_text("null")
    shot(page, "08-source-panel")

    panel.get_by_role("button", name="Show in document").click()
    expect(page.locator("dialog#details .page-view")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.locator("dialog#details")).to_be_hidden()
    page.keyboard.press("Escape")
    expect(panel).to_be_hidden()


def test_source_panel_shows_the_text_exactly_as_the_model_read_it(ready, fake_ollama):
    page = ready
    ask(page, "How did the retail segment do?")
    answer = wait_for_answer(page)

    answer.locator(".answer button.cite").first.click()

    passage = page.locator("#source-panel pre.passage").inner_text()
    source = page.request.get(page.url.replace("/#chat=", "/api/chats/")).json()["messages"][1]["sources"][0]
    assert passage == source["text"]


def test_follow_up_shows_what_was_searched(ready):
    page = ready
    ask(page, "How did retail revenue change in 2025?")
    wait_for_answer(page)

    ask(page, "and Wholesale?")

    answer = page.locator(".msg-assistant").nth(1)
    expect(answer.locator(".searched-for")).to_contain_text(REWRITE_REPLY, timeout=ANSWER_TIMEOUT)
    wait_for_answer(page)
    expect(page.locator(".msg")).to_have_count(4)


def test_saved_chat_comes_back_after_reloading(ready):
    page = ready
    ask(page, "How did retail revenue change in 2025?")
    wait_for_answer(page)
    ask(page, "and Wholesale?")
    wait_for_answer(page)

    page.reload()
    expect(page.locator("body[data-ready=true]")).to_be_attached()

    expect(page.locator(".msg")).to_have_count(4)
    expect(page.locator(".msg-assistant").nth(1).locator(".searched-for")).to_contain_text(REWRITE_REPLY)
    expect(page.locator(".msg-assistant").first.locator(".answer button.cite")).to_have_count(3)
    expect(page.locator(".chat-item.active")).to_have_count(1)


def test_stop_keeps_what_was_written(ready, fake_ollama):
    page = ready
    fake_ollama.stream_delay = 0.4
    ask(page, "How did retail do?")

    expect(last_answer(page).locator(".answer")).not_to_be_empty(timeout=ANSWER_TIMEOUT)
    page.locator("#stop").click()

    answer = last_answer(page)
    expect(answer.locator(".msg-note")).to_contain_text("Stopped")
    expect(page.locator("#send")).to_be_visible()
    chat_id = page.url.split("#chat=")[1]

    # The server saves the stopped answer when it notices the connection closed.
    def saved_status():
        messages = page.request.get(f"/api/chats/{chat_id}").json()["messages"]
        return messages[-1]["status"] if len(messages) == 2 else None

    page.wait_for_function("() => true")  # let the browser settle
    for _ in range(50):
        if saved_status() == "stopped":
            break
        page.wait_for_timeout(100)
    assert saved_status() == "stopped"

    page.reload()
    expect(last_answer(page).locator(".msg-note")).to_contain_text("Stopped")


def test_failed_answer_says_why(ready, fake_ollama):
    page = ready
    fake_ollama.stream_error = "CUDA out of memory"

    ask(page, "How did retail do?")

    note = last_answer(page).locator(".note-bad")
    expect(note).to_contain_text("CUDA out of memory", timeout=ANSWER_TIMEOUT)


def test_tables_render_and_html_in_answers_is_harmless(ready, fake_ollama):
    page = ready
    fake_ollama.chat_reply = lambda body: (
        '<img src="x" onerror="alert(1)"><script>alert(2)</script> **Totals** [1]\n\n'
        "| Segment | 2025 |\n|---|---|\n| Retail | 1,350 |\n\n[Click](javascript:alert(3))"
        if body["messages"][-1]["content"].startswith("Sources:")
        else "x"
    )
    page.on("dialog", lambda dialog: pytest.fail(f"a script ran: {dialog.message}"))

    ask(page, "Show the totals")

    answer = wait_for_answer(page).locator(".answer")
    expect(answer.locator("table td").first).to_have_text("Retail")
    expect(answer.locator("strong")).to_have_text("Totals")
    expect(answer.locator("img")).to_have_count(0)
    expect(answer.locator("script")).to_have_count(0)
    assert answer.locator("a").get_attribute("href") in (None, "")  # the javascript: link was removed
    shot(page, "09-chat-table")


def test_with_no_documents_the_answer_says_so(page):
    ask(page, "What was revenue?")

    expect(wait_for_answer(page).locator(".answer")).to_have_text(NOTHING_INDEXED)


def test_unusable_model_puts_the_question_back(page, sample_pdf):
    upload_and_wait(page, sample_pdf)
    page.evaluate(
        """() => { const s = document.getElementById('model-select');
                   s.append(new Option('gone', 'ollama:mistral')); s.value = 'ollama:mistral'; }"""
    )

    ask(page, "How did retail do?")

    expect(page.locator(".toast-error")).to_contain_text("ollama pull mistral")
    expect(page.locator("#question")).to_have_value("How did retail do?")
    expect(page.locator(".msg")).to_have_count(0)


def test_chats_can_be_switched_renamed_and_deleted(ready):
    page = ready
    ask(page, "First question about retail")
    wait_for_answer(page)
    page.get_by_role("button", name="New chat").click()
    expect(page.locator(".welcome")).to_be_visible()
    ask(page, "Second question about wholesale")
    wait_for_answer(page)
    expect(page.locator(".chat-item")).to_have_count(2)

    page.locator(".chat-item", has_text="First question").locator(".chat-open").click()
    expect(page.locator(".msg-user .bubble")).to_have_text("First question about retail")

    page.once("dialog", lambda prompt: prompt.accept("Retail review"))
    page.get_by_role("button", name="Rename First question about retail").click()
    expect(page.locator(".chat-item", has_text="Retail review")).to_be_visible()

    page.once("dialog", lambda confirm: confirm.accept())
    page.get_by_role("button", name="Delete Retail review").click()
    expect(page.locator(".chat-item")).to_have_count(1)
    expect(page.locator(".welcome")).to_be_visible()  # the open chat was the deleted one
    shot(page, "10-chat-list")


def test_chat_can_be_limited_to_some_documents(ready):
    page = ready
    page.locator("#doc-filter summary").click()
    page.locator(".doc-filter-option input").first.check()
    expect(page.locator("#doc-filter summary")).to_have_text("annual-report.pdf")

    ask(page, "How did retail do?")
    wait_for_answer(page)

    chat_id = page.url.split("#chat=")[1]
    chat = page.request.get(f"/api/chats/{chat_id}").json()
    assert len(chat["document_ids"]) == 1
