import pytest

from app.api.chats import answer_and_save
from app.store import chats
from tests.fake_ollama import ANSWER_REPLY, REWRITE_REPLY
from tests.pdf_factory import build_sample_pdf
from tests.test_chat_api import parse_events
from tests.test_documents_api import upload, wait_for

MODEL = "ollama:llama3.2:3b"


@pytest.fixture
def ready_doc(client, tmp_path):
    doc_id = upload(client, build_sample_pdf(tmp_path / "annual-report.pdf"), extractor="none").json()["id"]
    wait_for(client, doc_id)
    return doc_id


def new_chat(client, **body):
    response = client.post("/api/chats", json={"model": MODEL, **body})
    assert response.status_code == 201, response.text
    return response.json()


def ask(client, chat_id, question, **body):
    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"question": question, **body}) as response:
        assert response.status_code == 200, response.read()
        return parse_events("".join(response.iter_text()))


# ---- Chats ----


def test_new_chat_starts_empty(client):
    chat = new_chat(client)

    assert chat["title"] == "New chat"
    assert chat["model"] == MODEL
    assert chat["document_ids"] is None
    assert chat["message_count"] == 0
    assert client.get(f"/api/chats/{chat['id']}").json()["messages"] == []


def test_chats_are_listed_most_recent_first(client, ready_doc):
    first = new_chat(client)
    second = new_chat(client)
    ask(client, first["id"], "How did retail do?")

    assert [c["id"] for c in client.get("/api/chats").json()] == [first["id"], second["id"]]


def test_chat_can_be_renamed_and_limited_to_documents(client, ready_doc):
    chat = new_chat(client)

    renamed = client.patch(f"/api/chats/{chat['id']}", json={"title": "Q3 review", "document_ids": [ready_doc]}).json()
    assert renamed["title"] == "Q3 review"
    assert renamed["document_ids"] == [ready_doc]

    cleared = client.patch(f"/api/chats/{chat['id']}", json={"document_ids": None}).json()
    assert cleared["document_ids"] is None
    assert cleared["title"] == "Q3 review"  # fields not sent stay as they were


def test_chat_settings_are_checked(client):
    assert client.post("/api/chats", json={"model": "ollama:mistral"}).status_code == 400
    assert client.post("/api/chats", json={"document_ids": ["0" * 32]}).status_code == 404
    chat = new_chat(client)
    assert client.patch(f"/api/chats/{chat['id']}", json={"model": "gpt-4"}).status_code == 400
    assert client.get(f"/api/chats/{'0' * 32}").status_code == 404
    assert client.get("/api/chats/not-an-id").status_code == 422


def test_deleting_a_chat_removes_its_messages(client, ready_doc):
    chat = new_chat(client)
    ask(client, chat["id"], "How did retail do?")

    assert client.delete(f"/api/chats/{chat['id']}").status_code == 204
    assert client.get(f"/api/chats/{chat['id']}").status_code == 404
    assert client.delete(f"/api/chats/{chat['id']}").status_code == 404


# ---- Asking ----


def test_question_and_answer_are_saved_with_sources(client, ready_doc):
    chat = new_chat(client)

    events = ask(client, chat["id"], "How did the retail segment do this year?")

    done = events[-1][1]
    assert events[-1][0] == "done"
    assert done["answer"] == ANSWER_REPLY

    saved = client.get(f"/api/chats/{chat['id']}").json()
    assert saved["title"] == "How did the retail segment do this year?"
    question, answer = saved["messages"]
    assert (question["role"], question["content"]) == ("user", "How did the retail segment do this year?")
    assert answer["id"] == done["message_id"]
    assert answer["role"] == "assistant"
    assert answer["content"] == ANSWER_REPLY
    assert answer["status"] == "complete"
    assert answer["cited"] == [1, 2]
    assert answer["model"] == MODEL
    assert answer["search_query"] == "How did the retail segment do this year?"
    assert answer["seconds"] is not None
    assert answer["sources"][0]["number"] == 1
    assert answer["sources"][0]["filename"] == "annual-report.pdf"


def test_long_first_question_makes_a_short_title(client, ready_doc):
    chat = new_chat(client)
    ask(client, chat["id"], "Can you tell me how the retail and wholesale segments performed compared with last year?")

    title = client.get(f"/api/chats/{chat['id']}").json()["title"]
    assert title.endswith("…") and len(title) <= 61


def test_follow_up_is_understood_from_the_conversation(client, ready_doc, fake_ollama):
    chat = new_chat(client)
    ask(client, chat["id"], "How did retail revenue change in 2025?")

    events = ask(client, chat["id"], "and Wholesale?")

    assert events[0] == ("status", {"message": "Understanding your follow-up question."})
    sources = next(data for kind, data in events if kind == "sources")
    assert sources["search_query"] == REWRITE_REPLY

    rewrite_prompt = fake_ollama.calls_to("/api/chat")[-2]["messages"][-1]["content"]
    assert "Q: How did retail revenue change in 2025?" in rewrite_prompt
    assert "Follow-up question: and Wholesale?" in rewrite_prompt

    messages = client.get(f"/api/chats/{chat['id']}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[3]["search_query"] == REWRITE_REPLY


def test_model_can_change_between_questions(client, ready_doc):
    chat = new_chat(client)
    ask(client, chat["id"], "How did retail do?", model="ollama:qwen2.5vl:3b")

    saved = client.get(f"/api/chats/{chat['id']}").json()
    assert saved["model"] == "ollama:qwen2.5vl:3b"
    assert saved["messages"][1]["model"] == "ollama:qwen2.5vl:3b"


def test_chat_without_a_model_asks_for_one(client):
    chat = client.post("/api/chats", json={}).json()

    response = client.post(f"/api/chats/{chat['id']}/messages", json={"question": "Hi"})

    assert response.status_code == 400
    assert "no model yet" in response.json()["detail"]


def test_failed_answer_is_saved_with_what_was_written(client, ready_doc, fake_ollama):
    fake_ollama.stream_error = "CUDA out of memory"
    chat = new_chat(client)

    events = ask(client, chat["id"], "How did retail do?")

    assert events[-1][0] == "error"
    answer = client.get(f"/api/chats/{chat['id']}").json()["messages"][1]
    assert answer["status"] == "error"
    assert "CUDA out of memory" in answer["error"]
    assert answer["content"] == ANSWER_REPLY
    assert answer["id"] == events[-1][1]["message_id"]


def test_failed_answer_is_not_used_as_context_for_the_next_question(client, ready_doc, fake_ollama):
    fake_ollama.stream_error = "CUDA out of memory"
    chat = new_chat(client)
    ask(client, chat["id"], "How did retail do?")
    fake_ollama.stream_error = None

    events = ask(client, chat["id"], "and Wholesale?")

    assert events[0][0] == "sources"  # no complete earlier answer, so no rewrite


def test_old_answers_keep_their_sources_after_the_document_is_deleted(client, ready_doc):
    chat = new_chat(client)
    ask(client, chat["id"], "How did retail do?")

    client.delete(f"/api/documents/{ready_doc}")

    answer = client.get(f"/api/chats/{chat['id']}").json()["messages"][1]
    assert answer["sources"][0]["text"]
    assert answer["sources"][0]["filename"] == "annual-report.pdf"


def test_answer_stopped_part_way_is_saved_as_stopped(client, ready_doc):
    """Closing the stream early (the user pressed stop) keeps what was written so far."""
    app = client.app
    chat = new_chat(client)
    chats.add_message(app.state.db, chat["id"], "user", "How did retail do?")

    async def stop_after_first_token():
        model = await app.state.answerer.resolve_model(MODEL)
        stream = answer_and_save(app.state.answerer, app.state.db, chat["id"], "How did retail do?", model, None, [])
        async for line in stream:
            if line.startswith("event: token"):
                break
        await stream.aclose()

    client.portal.call(stop_after_first_token)

    answer = client.get(f"/api/chats/{chat['id']}").json()["messages"][1]
    assert answer["status"] == "stopped"
    assert answer["content"] == ANSWER_REPLY.split(" ")[0] + " "
    assert answer["sources"]


def test_number_check_is_saved_with_the_answer(client, ready_doc):
    chat = new_chat(client)

    done = ask(client, chat["id"], "How did retail do?")[-1][1]

    answer = client.get(f"/api/chats/{chat['id']}").json()["messages"][1]
    assert answer["number_check"] == done["number_check"]
    assert answer["number_check"]["summary"]["checked"] == 2
