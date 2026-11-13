"""The real search model, as built into the Docker image."""

import pytest

from app.config import Settings
from app.llm.embedder import MODEL_ID, EmbedderError, OnnxEmbedder

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def embedder():
    return OnnxEmbedder(Settings().embed_model_dir, threads=2)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_the_model_is_built_in_and_ready(embedder):
    assert await embedder.problem() is None
    assert embedder.model == MODEL_ID


async def test_vectors_have_768_numbers_and_unit_length(embedder):
    [vector] = await embedder.embed_documents(["Revenue grew 12% in 2025."])

    assert len(vector) == 768
    assert dot(vector, vector) == pytest.approx(1.0, abs=1e-5)


async def test_questions_land_closest_to_the_passage_that_answers_them(embedder):
    passages = await embedder.embed_documents([
        "Net profit after tax rose 18% to AED 7.8 billion.",
        "The bank opened three new branches in Sharjah.",
        "Total assets grew 21% to AED 416 billion.",
    ])

    for question, best in [("How much did total assets grow?", 2), ("What was the profit?", 0), ("Where were branches opened?", 1)]:
        query = await embedder.embed_query(question)
        scores = [dot(query, passage) for passage in passages]
        assert scores.index(max(scores)) == best, (question, scores)


async def test_questions_and_passages_get_their_own_prefix(embedder):
    [as_passage] = await embedder.embed_documents(["Revenue"])
    as_question = await embedder.embed_query("Revenue")

    assert dot(as_passage, as_question) < 0.99


async def test_a_text_gets_the_same_vector_alone_or_among_others(embedder):
    [alone] = await embedder.embed_documents(["Cost-to-income ratio was 28.4% in Q4'25."])
    together = await embedder.embed_documents(["A short one.", "Cost-to-income ratio was 28.4% in Q4'25.", "x " * 500])

    assert together[1] == pytest.approx(alone, abs=1e-6)


async def test_text_beyond_the_models_window_is_cut_off(embedder):
    start = "Revenue by segment and region. " * 600  # well over 2,048 tokens on its own
    first, second = await embedder.embed_documents([start + "Apples. " * 50, start + "Engines. " * 50])

    assert first == pytest.approx(second, abs=1e-6)


async def test_missing_model_files_say_how_to_fix_it(tmp_path):
    broken = OnnxEmbedder(tmp_path)

    problem = await broken.problem()
    assert "The search model's files are missing" in problem
    assert "docker compose up -d --build" in problem
    with pytest.raises(EmbedderError, match="missing"):
        await broken.embed_query("Hi")


async def test_a_damaged_model_file_is_reported(tmp_path, embedder):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_quantized.onnx").write_bytes(b"not a model")
    (tmp_path / "tokenizer.json").write_bytes((embedder.model_dir / "tokenizer.json").read_bytes())

    assert "Couldn't load the search model" in await OnnxEmbedder(tmp_path).problem()
