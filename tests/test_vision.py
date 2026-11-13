import base64

import pytest

from app.ingest.vision import (
    AwsDescriber,
    LocalDescriber,
    VisionService,
    clean_reply,
    figure_kind,
    figure_prompt,
    page_heading,
)
from app.llm.messages import ImageInput, solid_png
from tests.fake_bedrock import FakeBedrock
from tests.fake_ollama import FakeOllama

IMAGE = ImageInput(solid_png(8, 8, (200, 0, 0)), "png")


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        ("Type: chart\nTitle: Revenue", "chart"),
        ("**Type:** Table\n", "table"),
        ("type: graph", "chart"),
        ("Type: logo", "decorative"),
        ("Type: Infographic", "diagram"),
        ("Type: spaceship", "other"),
        ("No type line at all", "other"),
    ],
)
def test_figure_kind_comes_from_the_type_line(reply, kind):
    assert figure_kind(reply) == kind


def test_code_fences_around_a_reply_are_removed():
    assert clean_reply("```markdown\nType: chart\n```") == "Type: chart"
    assert clean_reply("  Type: photo  ") == "Type: photo"


def test_repeated_rows_from_a_looping_model_are_dropped():
    looping = "Type: chart\nData:\n| Month | Yield |\n|---|---|\n" + "| Jan-25 | 4.6 |\n| Apr-25 | 4.2 |\n" * 300

    cleaned = clean_reply(looping)

    assert cleaned == "Type: chart\nData:\n| Month | Yield |\n|---|---|\n| Jan-25 | 4.6 |\n| Apr-25 | 4.2 |"


def test_runaway_descriptions_are_cut_short():
    long = "\n".join(f"| Row {i} | {i} |" for i in range(2000))

    cleaned = clean_reply(long)

    assert len(cleaned) < 6100
    assert cleaned.endswith("(cut short: the description was too long)")


def test_page_heading_prefers_a_markdown_heading():
    assert page_heading("intro line\n## Revenue by region\nmore") == "Revenue by region"
    assert page_heading("Short first line\nbody") == "Short first line"
    assert page_heading("") == ""


def test_figure_prompt_carries_the_pdf_text():
    prompt = figure_prompt(3, "Results", "Figure 1: Revenue", "2018 2019 120 150")

    assert "page 3" in prompt
    assert "Page heading: Results" in prompt
    assert "Caption: Figure 1: Revenue" in prompt
    assert "Text inside the figure: 2018 2019 120 150" in prompt


def test_figure_prompt_marks_missing_context():
    assert "Caption: (none)" in figure_prompt(1, "", "", "")


# ---- Which extractors can be used ----


def service(fake: FakeOllama, bedrock: FakeBedrock | None = None, model: str = "qwen2.5vl:3b") -> VisionService:
    return VisionService(fake.client(), bedrock or FakeBedrock(), local_model=model)


@pytest.mark.anyio
async def test_local_extractor_is_ready_when_the_vision_model_is_installed():
    assert await service(FakeOllama()).problem("local") is None


@pytest.mark.anyio
async def test_local_extractor_says_how_to_install_a_missing_model():
    problem = await service(FakeOllama(), model="qwen2.5vl:7b").problem("local")
    assert problem == "qwen2.5vl:7b isn't installed. Run: ollama pull qwen2.5vl:7b"


@pytest.mark.anyio
async def test_local_extractor_needs_a_model_that_can_read_images():
    problem = await service(FakeOllama(), model="llama3.2:3b").problem("local")
    assert "can't read images" in problem


@pytest.mark.anyio
async def test_local_extractor_reports_ollama_down():
    assert "Is Ollama running?" in await service(FakeOllama(reachable=False)).problem("local")


@pytest.mark.anyio
async def test_aws_extractor_needs_aws_settings():
    assert "AWS keys are not set" in await service(FakeOllama()).problem("aws")
    assert await service(FakeOllama(), FakeBedrock(configured=True)).problem("aws") is None


@pytest.mark.anyio
async def test_text_only_is_always_available():
    assert await service(FakeOllama(reachable=False)).problem("none") is None


# ---- Backends ----


@pytest.mark.anyio
async def test_local_model_gets_the_gpu_with_the_same_context_size_it_will_use():
    fake = FakeOllama(loaded=["llama3.2:3b", "nomic-embed-text:latest"])
    describer = LocalDescriber(fake.client(), "qwen2.5vl:3b", num_ctx=8192)

    await describer.prepare()

    assert fake.loaded == ["qwen2.5vl:3b"]
    assert fake.calls_to("/api/generate")[-1]["options"] == {"num_ctx": 8192, "temperature": 0, "num_predict": 1500}


@pytest.mark.anyio
async def test_local_model_gets_the_image_and_instructions():
    fake = FakeOllama()

    reply = await LocalDescriber(fake.client(), "qwen2.5vl:3b", num_ctx=8192).describe(IMAGE, "Describe it")

    assert reply.startswith("Type: chart")
    call = fake.calls_to("/api/chat")[0]
    assert call["options"] == {"num_ctx": 8192, "temperature": 0, "num_predict": 1500}
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["content"] == "Describe it"
    assert call["messages"][1]["images"] == [base64.b64encode(IMAGE.data).decode()]


@pytest.mark.anyio
async def test_nova_gets_the_image_and_instructions():
    bedrock = FakeBedrock(configured=True, nova_reply="Type: photo\nTitle: Office")

    reply = await AwsDescriber(bedrock).describe(IMAGE, "Describe it")

    assert reply == "Type: photo\nTitle: Office"
    _, messages, kwargs = bedrock.calls[0]
    assert messages[0].images == (IMAGE,)
    assert kwargs["temperature"] == 0
    assert kwargs["system"]


@pytest.mark.anyio
async def test_reading_a_figure_waits_while_a_chat_answer_holds_the_gpu():
    import asyncio

    fake = FakeOllama()
    describer = LocalDescriber(fake.client(), "qwen2.5vl:3b", num_ctx=8192)

    await describer.ollama.gpu.acquire()  # a chat answer is being written
    task = asyncio.ensure_future(describer.describe(IMAGE, "Describe it"))
    await asyncio.sleep(0.01)
    assert not task.done()
    assert fake.calls_to("/api/chat") == []

    describer.ollama.gpu.release()
    assert (await task).startswith("Type: chart")
