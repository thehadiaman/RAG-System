import pytest

from app.llm.embedder import Embedder, EmbedderError
from tests.fake_ollama import FakeOllama

pytestmark = pytest.mark.anyio


def make(fake: FakeOllama, model: str = "nomic-embed-text", batch_size: int = 32) -> Embedder:
    return Embedder(fake.client(), model, batch_size)


async def test_documents_and_questions_get_nomic_prefixes():
    fake = FakeOllama()
    embedder = make(fake)

    await embedder.embed_documents(["Revenue grew 12%"])
    await embedder.embed_query("How much did revenue grow?")

    inputs = [call["input"] for call in fake.calls_to("/api/embed")]
    assert inputs == [["search_document: Revenue grew 12%"], ["search_query: How much did revenue grow?"]]


async def test_other_embedding_models_get_no_prefix():
    fake = FakeOllama(models={"mxbai-embed-large:latest": ["embedding"]})

    await make(fake, model="mxbai-embed-large").embed_query("Hi")

    assert fake.calls_to("/api/embed")[0]["input"] == ["Hi"]


async def test_runs_on_cpu_and_stays_loaded():
    fake = FakeOllama()

    await make(fake).embed_query("Hi")

    call = fake.calls_to("/api/embed")[0]
    assert call["options"] == {"num_gpu": 0}
    assert call["keep_alive"] == -1


async def test_texts_are_sent_in_batches_and_keep_their_order():
    fake = FakeOllama()
    texts = ["one", "two", "three", "four", "five"]

    vectors = await make(fake, batch_size=2).embed_documents(texts)

    assert [len(call["input"]) for call in fake.calls_to("/api/embed")] == [2, 2, 1]
    assert len(vectors) == 5
    assert vectors[2] == (await make(fake).embed_documents(["three"]))[0]


async def test_missing_model_says_how_to_install_it():
    fake = FakeOllama(models={"llama3.2:3b": ["completion"]})

    with pytest.raises(EmbedderError, match="Run: ollama pull nomic-embed-text"):
        await make(fake).embed_query("Hi")


async def test_problem_is_none_when_ready():
    assert await make(FakeOllama()).problem() is None


async def test_problem_reports_missing_model():
    fake = FakeOllama(models={"llama3.2:3b": ["completion"]})

    assert await make(fake).problem() == (
        "Embedding model nomic-embed-text isn't installed. Run: ollama pull nomic-embed-text"
    )


async def test_problem_reports_ollama_down():
    assert "Is Ollama running?" in await make(FakeOllama(reachable=False)).problem()
