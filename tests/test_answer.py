import asyncio

import pytest

from app.answer import (
    NOT_FOUND,
    NOTHING_INDEXED,
    SYSTEM_PROMPT,
    Answerer,
    AnswerError,
    Source,
    build_prompt,
    cited_numbers,
    select_sources,
)
from app.llm.messages import ImageInput, solid_png
from app.search import Hit, SearchResult
from tests.fake_bedrock import FakeBedrock
from tests.fake_ollama import ANSWER_REPLY, FakeOllama


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


async def collect(answering, prepared):
    return [(e.type, e.data) for e in [event async for event in answering.stream(prepared)]]


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

    prepared = await answering.prepare("How did retail do?", "ollama:llama3.2:3b")
    events = await collect(answering, prepared)

    types = [t for t, _ in events]
    assert types[0] == "sources" and types[-1] == "done"
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
async def test_vision_model_also_gets_chart_images(tmp_path, figure_file):
    fake = FakeOllama()
    hits = [chunk(1), chunk(2, "figure", image_path=figure_file), chunk(3, "figure", image_path=figure_file),
            chunk(4, "figure", image_path=figure_file)]
    answering = answerer(tmp_path, hits, ollama=fake, max_images=2)

    prepared = await answering.prepare("Show the trend", "ollama:qwen2.5vl:3b")
    await collect(answering, prepared)

    assert [s.image is not None for s in prepared.sources] == [False, True, True, False]
    message = fake.calls_to("/api/chat")[0]["messages"][1]
    assert len(message["images"]) == 2
    assert "image 1 is source [2], image 2 is source [3]" in message["content"]


@pytest.mark.anyio
async def test_text_model_gets_no_images(tmp_path, figure_file):
    answering = answerer(tmp_path, [chunk(1, "figure", image_path=figure_file)])

    prepared = await answering.prepare("q", "ollama:llama3.2:3b")

    assert prepared.sources[0].image is None


@pytest.mark.anyio
async def test_missing_figure_file_just_means_no_image(tmp_path):
    answering = answerer(tmp_path, [chunk(1, "figure", image_path="documents/gone.png")])

    prepared = await answering.prepare("q", "ollama:qwen2.5vl:3b")

    assert prepared.sources[0].image is None


@pytest.mark.anyio
async def test_no_search_results_means_no_model_call(tmp_path):
    fake = FakeOllama()
    answering = answerer(tmp_path, [], ollama=fake)

    events = await collect(answering, await answering.prepare("q", "ollama:llama3.2:3b"))

    assert [t for t, _ in events] == ["sources", "token", "done"]
    assert events[1][1]["text"] == NOTHING_INDEXED
    assert fake.calls_to("/api/chat") == []


@pytest.mark.anyio
async def test_a_failure_while_writing_ends_with_an_error_event(tmp_path):
    fake = FakeOllama()
    fake.stream_error = "CUDA out of memory"
    answering = answerer(tmp_path, [chunk(1)], ollama=fake)

    events = await collect(answering, await answering.prepare("q", "ollama:llama3.2:3b"))

    kind, data = events[-1]
    assert kind == "error"
    assert "CUDA out of memory" in data["message"]
    assert data["partial_answer"] == ANSWER_REPLY


@pytest.mark.anyio
async def test_waits_for_the_gpu_while_a_pdf_is_being_read(tmp_path):
    fake = FakeOllama()
    answering = answerer(tmp_path, [chunk(1)], ollama=fake)
    prepared = await answering.prepare("q", "ollama:llama3.2:3b")

    await answering.ollama.gpu.acquire()  # the vision step is busy with a figure
    events = answering.stream(prepared)
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

    events = await collect(answering, await answering.prepare("q", "aws:qwen2.5-vl"))

    assert events[-1][1]["answer"].strip() == "Revenue was $9.5B [1]."
    assert events[-1][1]["cited"] == [1]
    model, messages, kwargs = bedrock.calls[0]
    assert model == "qwen_stream"
    assert len(messages[0].images) == 1
    assert kwargs["system"] == SYSTEM_PROMPT
