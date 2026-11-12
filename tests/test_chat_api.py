import json

import pytest

from app.answer import NOTHING_INDEXED
from tests.fake_ollama import ANSWER_REPLY
from tests.pdf_factory import build_sample_pdf
from tests.test_documents_api import upload, wait_for


def ask(client, **body):
    """POST /api/chat and return the Server-Sent Events as (event, data) pairs."""
    with client.stream("POST", "/api/chat", json=body) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


@pytest.fixture
def ready_doc(client, tmp_path):
    doc_id = upload(client, build_sample_pdf(tmp_path / "annual-report.pdf"), extractor="none").json()["id"]
    wait_for(client, doc_id)
    return doc_id


def test_answer_is_streamed_with_numbered_sources(client, ready_doc):
    events = ask(client, question="How did the retail segment do?", model="ollama:llama3.2:3b")

    kinds = [kind for kind, _ in events]
    assert kinds[0] == "sources"
    assert kinds[-1] == "done"
    assert "token" in kinds

    sources = events[0][1]["sources"]
    assert [s["number"] for s in sources] == list(range(1, len(sources) + 1))
    assert sources[0]["filename"] == "annual-report.pdf"
    assert all(s["page_image_url"].startswith(f"/api/documents/{ready_doc}/pages/") for s in sources)
    assert client.get(sources[0]["page_image_url"]).status_code == 200

    done = events[-1][1]
    assert done["answer"] == ANSWER_REPLY
    assert done["cited"] == [1, 2]
    assert "".join(data["text"] for kind, data in events if kind == "token") == ANSWER_REPLY


def test_vision_model_is_sent_chart_images(client, ready_doc, fake_ollama):
    events = ask(client, question="Revenue by year chart", model="ollama:qwen2.5vl:3b")

    sent = [s for s in events[0][1]["sources"] if s["image_sent_to_model"]]
    assert sent and all(s["kind"] == "figure" for s in sent)
    message = fake_ollama.calls_to("/api/chat")[-1]["messages"][-1]
    assert len(message["images"]) == len(sent)


def test_answer_can_be_limited_to_some_documents(client, ready_doc):
    events = ask(client, question="retail", model="ollama:llama3.2:3b", document_ids=[ready_doc])

    assert {s["document_id"] for s in events[0][1]["sources"]} == {ready_doc}


def test_with_nothing_indexed_the_answer_says_so(client, fake_ollama):
    events = ask(client, question="How did retail do?", model="ollama:llama3.2:3b")

    assert events[0][1]["sources"] == []
    assert events[-1][1]["answer"] == NOTHING_INDEXED
    assert fake_ollama.calls_to("/api/chat") == []


@pytest.mark.parametrize(
    ("model", "status", "message"),
    [
        ("ollama:mistral", 400, "ollama pull mistral"),
        ("aws:qwen2.5-vl", 400, "AWS keys are not set"),
        ("gpt-4", 400, "Unknown model"),
    ],
)
def test_unusable_model_is_refused_before_streaming(client, model, status, message):
    response = client.post("/api/chat", json={"question": "q", "model": model})

    assert response.status_code == status
    assert message in response.json()["detail"]


def test_chat_checks_its_input(client):
    assert client.post("/api/chat", json={"question": " ", "model": "ollama:llama3.2:3b"}).status_code == 422
    assert client.post("/api/chat", json={"question": "q"}).status_code == 422
    unknown = client.post("/api/chat", json={"question": "q", "model": "ollama:llama3.2:3b", "document_ids": ["0" * 32]})
    assert unknown.status_code == 404
