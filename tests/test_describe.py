import pytest

from app.ingest.jobs import IngestQueue, document_dir
from app.ingest.vision import VisionService
from app.store import documents
from app.store.db import Database
from tests.fake_bedrock import FakeBedrock
from tests.fake_ollama import FakeOllama
from tests.pdf_factory import build_sample_pdf

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr("app.ingest.describe.RETRY_DELAY", 0)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "app.db")
    database.migrate()
    return database


def add_sample(db, data_dir, extractor):
    """The 5-page sample: 2 figures (pages 3 and 4) and 1 scanned page (page 5)."""
    doc_id = documents.new_id()
    folder = document_dir(data_dir, doc_id)
    folder.mkdir(parents=True)
    build_sample_pdf(folder / "original.pdf")
    documents.create(db, doc_id, "sample.pdf", doc_id, 1000, 5, extractor)
    return doc_id


def queue(db, data_dir, ollama: FakeOllama | None = None, bedrock: FakeBedrock | None = None) -> IngestQueue:
    vision = VisionService(
        (ollama or FakeOllama()).client(keep_loaded=("nomic-embed-text",)),
        bedrock or FakeBedrock(),
        local_model="qwen2.5vl:3b",
    )
    return IngestQueue(db, data_dir, vision=vision)


async def test_local_model_describes_figures_and_reads_scanned_pages(db, tmp_path):
    fake = FakeOllama()
    doc_id = add_sample(db, tmp_path, "local")

    await queue(db, tmp_path, fake).process(doc_id)

    doc = documents.get(db, doc_id)
    assert doc["status"] == "extracted"
    assert doc["warning"] is None

    figures = documents.figures(db, doc_id)
    assert len(figures) == 2
    assert all(f["kind"] == "chart" and f["described_by"] == "qwen2.5vl:3b" for f in figures)
    assert figures[0]["description"].startswith("Type: chart")

    scanned = documents.page(db, doc_id, 5)
    assert scanned["text_source"] == "vision"
    assert "Total revenue was AED 5 million." in scanned["text"]

    prompts = [call["messages"][-1]["content"] for call in fake.calls_to("/api/chat")]
    assert len(prompts) == 3
    assert any("Caption: Figure 1: Revenue by year (AED million)" in p for p in prompts)


async def test_one_bad_item_does_not_stop_the_rest(db, tmp_path):
    fake = FakeOllama()
    fake.chat_failures = 2  # the first item fails, and so does its retry
    doc_id = add_sample(db, tmp_path, "local")

    await queue(db, tmp_path, fake).process(doc_id)

    doc = documents.get(db, doc_id)
    assert doc["status"] == "extracted"
    assert doc["warning"] == "1 of 3 figures and scanned pages couldn't be read."
    assert documents.page(db, doc_id, 5)["read_error"]  # scanned pages go first
    assert all(f["description"] for f in documents.figures(db, doc_id))


async def test_gives_up_after_repeated_failures(db, tmp_path):
    fake = FakeOllama()
    fake.chat_failures = 100
    doc_id = add_sample(db, tmp_path, "local")

    await queue(db, tmp_path, fake).process(doc_id)

    doc = documents.get(db, doc_id)
    assert doc["status"] == "extracted"  # the PDF's own text is still usable
    assert doc["warning"].startswith("Stopped reading figures after repeated errors")


async def test_a_resumed_run_skips_figures_already_described(db, tmp_path):
    fake = FakeOllama()
    doc_id = add_sample(db, tmp_path, "none")
    work = queue(db, tmp_path, fake)
    await work.process(doc_id)  # read the PDF only

    first = documents.figures(db, doc_id)[0]
    documents.save_figure_description(db, first["id"], "chart", "Type: chart", "earlier run")
    documents.set_extractor(db, doc_id, "local")
    documents.set_status(db, doc_id, documents.DESCRIBING)

    await work.process(doc_id)

    assert len(fake.calls_to("/api/chat")) == 2  # the other figure and the scanned page
    assert documents.figures(db, doc_id)[0]["described_by"] == "earlier run"


async def test_aws_extractor_uses_nova(db, tmp_path):
    bedrock = FakeBedrock(configured=True, nova_reply="Type: chart\nTitle: Revenue")
    doc_id = add_sample(db, tmp_path, "aws")

    await queue(db, tmp_path, bedrock=bedrock).process(doc_id)

    assert len(bedrock.calls) == 3
    assert {f["described_by"] for f in documents.figures(db, doc_id)} == {"amazon.nova-lite-v1:0"}


async def test_missing_vision_model_leaves_a_warning(db, tmp_path):
    fake = FakeOllama(models={"nomic-embed-text:latest": ["embedding"]})
    doc_id = add_sample(db, tmp_path, "local")

    await queue(db, tmp_path, fake).process(doc_id)

    doc = documents.get(db, doc_id)
    assert doc["status"] == "extracted"
    assert "qwen2.5vl:3b isn't installed" in doc["warning"]
    assert fake.calls_to("/api/chat") == []
