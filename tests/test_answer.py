import asyncio

import pytest

from app.answer import (
    NOT_FOUND,
    NOTHING_INDEXED,
    SYSTEM_PROMPT,
    Answerer,
    AnswerError,
    Source,
    Turn,
    build_prompt,
    cited_numbers,
    clean_rewrite,
    format_history,
    rewrite_prompt,
    select_sources,
)
from app.llm.messages import ImageInput, solid_png
from app.search import Hit, SearchResult
from tests.fake_bedrock import FakeBedrock
from tests.fake_ollama import ANSWER_REPLY, REWRITE_REPLY, FakeOllama


def chunk(n, kind="text", text=None, page_no=1, section="Results", image_path=None):
    return {
        "id": f"c{n}",
        "document_id": "d" * 32,
        "filename": "annual-report.pdf",
        "page_no": page_no,
        "kind": kind,
        "section": section,
        "text": text or f"Chunk number {n}.",
        "figure_id": f"f{n}" if kind == "figure" else None,
        "image_path": image_path,
    }


# ---- Sources and prompt ----


def test_sources_keep_search_order_and_get_numbers():
    sources = select_sources([chunk(1), chunk(2), chunk(3)], max_sources=2, max_chars=10_000)

    assert [(s.number, s.hit["id"]) for s in sources] == [(1, "c1"), (2, "c2")]


def test_sources_are_trimmed_to_the_size_budget():
    long = " ".join(["word"] * 2000)  # ~10,000 characters
    sources = select_sources([chunk(1, text=long), chunk(2, text=long), chunk(3)], max_sources=8, max_chars=4000)

    assert len(sources[0].text) <= 3002 and sources[0].text.endswith(" …")
    assert sum(len(s.text) for s in sources) <= 4000 + 2
    assert len(sources) == 2


def test_prompt_numbers_the_sources_and_says_where_they_are_from():
    sources = [
        Source(1, chunk(1, text="Revenue rose 12%."), "Revenue rose 12%."),
        Source(2, chunk(2, kind="table", page_no=4, section=""), "|A|B|"),
    ]

    prompt = build_prompt("How did revenue do?", sources)

    assert prompt.startswith("Sources:\n\n[1] annual-report.pdf, page 1, text, section: Results\nRevenue rose 12%.")
    assert "[2] annual-report.pdf, page 4, table\n|A|B|" in prompt
    assert "Question: How did revenue do?" in prompt
    assert "images attached" not in prompt


def test_prompt_says_which_source_each_image_belongs_to():
    image = ImageInput(solid_png(4, 4, (0, 0, 0)), "png")
    sources = [Source(1, chunk(1), "a"), Source(2, chunk(2, "figure"), "b", image), Source(3, chunk(3, "figure"), "c", image)]

    assert "image 1 is source [2], image 2 is source [3]" in build_prompt("q", sources)


@pytest.mark.parametrize(
    ("answer", "cited"),
    [
        ("Revenue rose [1].", [1]),
        ("Up [2][1], and again [2].", [2, 1]),
        ("Both [1, 3] and [2; 4].", [1, 3, 2, 4]),
        ("A made-up source [9] and a year [2025] are ignored.", []),
        ("No citations at all.", []),
    ],
)
def test_cited_numbers_are_read_from_the_answer(answer, cited):
    assert cited_numbers(answer, source_count=4) == cited


def test_system_prompt_sets_the_ground_rules():
    assert NOT_FOUND in SYSTEM_PROMPT
    assert "[2]" in SYSTEM_PROMPT
    assert "exactly" in SYSTEM_PROMPT


# ---- Follow-up questions ----

HISTORY = [
    Turn("How much did Retail revenue grow in 2025?", "Retail revenue rose to 1,350 [1], up 12.5% [2]."),
    Turn("What about Online?", "Online grew 30.0% [3]."),
]


def test_history_drops_old_citation_numbers_and_long_answers():
    long = Turn("Q3?", "word " * 300)

    text = format_history([HISTORY[0], long])

    assert text.startswith("Q: How much did Retail revenue grow in 2025?\nA: Retail revenue rose to 1,350, up 12.5%.")
    assert "[1]" not in text
    assert text.endswith(" …")
    assert len(text) < 700


def test_history_keeps_only_the_last_turns():
    turns = [Turn(f"Question {i}?", f"Answer {i}.") for i in range(5)]

    text = format_history(turns)

    assert "Question 2?" not in text
    assert "Question 3?" in text and "Question 4?" in text


def test_prompt_includes_the_conversation_after_the_sources():
    prompt = build_prompt("And Online?", [Source(1, chunk(1), "a")], HISTORY)

    assert prompt.index("Sources:") < prompt.index("Earlier in this conversation") < prompt.index("Question: And Online?")
    assert "Q: What about Online?\nA: Online grew 30.0%." in prompt


def test_rewrite_prompt_shows_the_conversation_and_the_follow_up():
    prompt = rewrite_prompt("and Wholesale?", HISTORY)

    assert prompt.startswith("Conversation so far:\nQ: How much did Retail revenue grow in 2025?")
    assert "Follow-up question: and Wholesale?" in prompt


@pytest.mark.parametrize(
    ("reply", "question"),
    [
        ("What was Wholesale revenue in 2025?", "What was Wholesale revenue in 2025?"),
        ('Standalone question: "What was Wholesale revenue in 2025?"', "What was Wholesale revenue in 2025?"),
        ("**What was Wholesale revenue?**\nI kept the year.", "What was Wholesale revenue?"),
        ("", ""),
        ("x" * 500, ""),
    ],
)
def test_rewritten_question_is_cleaned(reply, question):
    assert clean_rewrite(reply) == question


# ---- Answerer ----


class StubSearcher:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    async def search(self, question, document_ids=None, limit=8):
        self.calls.append((question, document_ids, limit))
        return SearchResult([Hit(h) for h in self.hits], "hybrid")


@pytest.fixture
def figure_file(tmp_path):
    path = tmp_path / "documents" / "d" / "figures" / "p0003-01.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(solid_png(64, 48, (10, 120, 200)))
    return "documents/d/figures/p0003-01.png"


def answerer(tmp_path, hits, ollama=None, bedrock=None, **settings):
    fake = ollama or FakeOllama()
    client = fake.client(keep_loaded=("nomic-embed-text",))
    return Answerer(StubSearcher(hits), client, bedrock or FakeBedrock(), tmp_path, **settings)


async def run(answering, question, model_id, history=None, document_ids=None):
    model = await answering.resolve_model(model_id)
    return [(e.type, e.data) async for e in answering.answer(question, model, document_ids, history)]


def sources_of(events):
    return next(data["sources"] for kind, data in events if kind == "sources")


@pytest.mark.anyio
async def test_text_model_is_resolved_from_ollama(tmp_path):
    model = await answerer(tmp_path, []).resolve_model("ollama:llama3.2:3b")

    assert (model.provider, model.name, model.vision) == ("ollama", "llama3.2:3b", False)
    assert (await answerer(tmp_path, []).resolve_model("ollama:qwen2.5vl:3b")).vision is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model_id", "message"),
    [
        ("ollama:mistral", "isn't installed in Ollama. Run: ollama pull mistral"),
        ("ollama:nomic-embed-text", "can't chat"),
        ("gpt-4", "Unknown model"),
        ("aws:qwen2.5-vl", "AWS keys are not set"),
    ],
)
async def test_unusable_models_are_refused_with_a_reason(tmp_path, model_id, message):
    with pytest.raises(AnswerError, match=message):
        await answerer(tmp_path, []).resolve_model(model_id)


@pytest.mark.anyio
async def test_answer_streams_sources_then_text_then_citations(tmp_path):
    fake = FakeOllama(loaded=["qwen2.5vl:3b"])
    answering = answerer(tmp_path, [chunk(1), chunk(2, "table")], ollama=fake, num_ctx=8192)

    events = await run(answering, "How did retail do?", "ollama:llama3.2:3b")

    types = [t for t, _ in events]
    assert types[0] == "sources" and types[-1] == "done"
    assert events[0][1]["search_query"] == "How did retail do?"
    assert [d["message"] for t, d in events if t == "status"] == [
        "Loading llama3.2:3b. This can take up to a minute.",
        "Writing the answer.",
    ]
    assert "".join(d["text"] for t, d in events if t == "token") == ANSWER_REPLY
    done = events[-1][1]
    assert done["answer"] == ANSWER_REPLY
    assert done["cited"] == [1, 2]
    assert done["model"] == "ollama:llama3.2:3b"
    assert done["first_token_seconds"] is not None

    # The other model was unloaded first, and the chat model got the settings it is used with.
    assert "qwen2.5vl:3b" not in fake.loaded
    call = fake.calls_to("/api/chat")[0]
    assert call["stream"] is True
    assert call["options"] == {"num_ctx": 8192, "temperature": 0.1, "num_predict": 1024}
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call["messages"][1]["content"].startswith("Sources:\n\n[1] annual-report.pdf, page 1, text")
    assert "images" not in call["messages"][1]


@pytest.mark.anyio
async def test_follow_up_is_rewritten_before_searching(tmp_path):
    fake = FakeOllama()
    answering = answerer(tmp_path, [chunk(1)], ollama=fake)

    events = await run(answering, "and Wholesale?", "ollama:llama3.2:3b", history=HISTORY)

    assert events[0] == ("status", {"message": "Understanding your follow-up question."})
    assert events[1][1]["search_query"] == REWRITE_REPLY
    assert answering.searcher.calls[0][0] == REWRITE_REPLY

    rewrite_call, answer_call = fake.calls_to("/api/chat")
    assert rewrite_call["stream"] is False
    assert "Follow-up question: and Wholesale?" in rewrite_call["messages"][-1]["content"]
    assert rewrite_call["options"]["num_ctx"] == 8192  # same as the answer, so no reload in between
    prompt = answer_call["messages"][-1]["content"]
    assert f"Question: {REWRITE_REPLY}" in prompt
    assert "Q: What about Online?" in prompt


@pytest.mark.anyio
async def test_first_question_is_not_rewritten(tmp_path):
    fake = FakeOllama()

    await run(answerer(tmp_path, [chunk(1)], ollama=fake), "How did retail do?", "ollama:llama3.2:3b")

    assert len(fake.calls_to("/api/chat")) == 1


@pytest.mark.anyio
async def test_failed_rewrite_falls_back_to_the_previous_question_plus_the_new_one(tmp_path):
    fake = FakeOllama()
    fake.chat_failures = 1  # only the rewrite call fails
    answering = answerer(tmp_path, [chunk(1)], ollama=fake)

    events = await run(answering, "and Wholesale?", "ollama:llama3.2:3b", history=HISTORY)

    assert events[1][1]["search_query"] == "What about Online? and Wholesale?"
    assert events[-1][0] == "done"


@pytest.mark.anyio
async def test_vision_model_also_gets_chart_images(tmp_path, figure_file):
    fake = FakeOllama()
    hits = [chunk(1), chunk(2, "figure", image_path=figure_file), chunk(3, "figure", image_path=figure_file),
            chunk(4, "figure", image_path=figure_file)]
    answering = answerer(tmp_path, hits, ollama=fake, max_images=2)

    events = await run(answering, "Show the trend", "ollama:qwen2.5vl:3b")

    assert [s.image is not None for s in sources_of(events)] == [False, True, True, False]
    message = fake.calls_to("/api/chat")[0]["messages"][1]
    assert len(message["images"]) == 2
    assert "image 1 is source [2], image 2 is source [3]" in message["content"]


@pytest.mark.anyio
async def test_text_model_gets_no_images(tmp_path, figure_file):
    events = await run(answerer(tmp_path, [chunk(1, "figure", image_path=figure_file)]), "q", "ollama:llama3.2:3b")

    assert sources_of(events)[0].image is None


@pytest.mark.anyio
async def test_missing_figure_file_just_means_no_image(tmp_path):
    events = await run(answerer(tmp_path, [chunk(1, "figure", image_path="documents/gone.png")]), "q", "ollama:qwen2.5vl:3b")

    assert sources_of(events)[0].image is None


@pytest.mark.anyio
async def test_no_search_results_means_no_model_call(tmp_path):
    fake = FakeOllama()

    events = await run(answerer(tmp_path, [], ollama=fake), "q", "ollama:llama3.2:3b")

    assert [t for t, _ in events] == ["sources", "token", "done"]
    assert events[1][1]["text"] == NOTHING_INDEXED
    assert fake.calls_to("/api/chat") == []


@pytest.mark.anyio
async def test_a_failure_while_writing_ends_with_an_error_event(tmp_path):
    fake = FakeOllama()
    fake.stream_error = "CUDA out of memory"

    events = await run(answerer(tmp_path, [chunk(1)], ollama=fake), "q", "ollama:llama3.2:3b")

    kind, data = events[-1]
    assert kind == "error"
    assert "CUDA out of memory" in data["message"]
    assert data["partial_answer"] == ANSWER_REPLY


@pytest.mark.anyio
async def test_waits_for_the_gpu_while_a_pdf_is_being_read(tmp_path):
    fake = FakeOllama()
    answering = answerer(tmp_path, [chunk(1)], ollama=fake)
    model = await answering.resolve_model("ollama:llama3.2:3b")

    await answering.ollama.gpu.acquire()  # the vision step is busy with a figure
    events = answering.answer("q", model)
    assert (await anext(events)).type == "sources"
    waiting = await anext(events)
    assert waiting.data["message"] == "Waiting for the GPU: a PDF is being read by the vision model."

    next_event = asyncio.ensure_future(anext(events))
    await asyncio.sleep(0.01)
    assert not next_event.done()  # still waiting
    answering.ollama.gpu.release()
    assert (await next_event).type == "status"
    rest = [e.type async for e in events]
    assert rest[-1] == "done"


@pytest.mark.anyio
async def test_aws_qwen_answers_with_images(tmp_path, figure_file):
    bedrock = FakeBedrock(configured=True, qwen_reply="Revenue was $9.5B [1].")
    answering = answerer(tmp_path, [chunk(1, "figure", image_path=figure_file)], bedrock=bedrock)

    events = await run(answering, "q", "aws:qwen2.5-vl")

    assert events[-1][1]["answer"].strip() == "Revenue was $9.5B [1]."
    assert events[-1][1]["cited"] == [1]
    model, messages, kwargs = bedrock.calls[0]
    assert model == "qwen_stream"
    assert len(messages[0].images) == 1
    assert kwargs["system"] == SYSTEM_PROMPT


@pytest.mark.anyio
async def test_aws_qwen_rewrites_follow_ups_too(tmp_path):
    bedrock = FakeBedrock(configured=True, qwen_reply="What was Wholesale revenue in 2025?")
    answering = answerer(tmp_path, [chunk(1)], bedrock=bedrock)

    events = await run(answering, "and Wholesale?", "aws:qwen2.5-vl", history=HISTORY)

    assert next(d for t, d in events if t == "sources")["search_query"] == "What was Wholesale revenue in 2025?"
    assert [call[0] for call in bedrock.calls] == ["qwen", "qwen_stream"]


@pytest.mark.anyio
async def test_finished_answer_comes_with_its_number_check(tmp_path):
    hits = [chunk(1, text="Retail revenue was 1,350 in 2025."), chunk(2, text="Online grew 25.0%.")]

    events = await run(answerer(tmp_path, hits), "How did retail do?", "ollama:llama3.2:3b")

    check = events[-1][1]["number_check"]
    # The fake answer says "1,350 [1]" (right) and "online grew 30.0% [1][2]" (in no source).
    assert check["summary"] == {"checked": 2, "ok": 1, "other_source": 0, "not_found": 1}
    assert [(n["text"], n["status"]) for n in check["numbers"]] == [("1,350", "ok"), ("30.0%", "not_found")]
