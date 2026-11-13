import pytest

from app.llm.ollama import OllamaError, full_name
from tests.fake_ollama import FakeOllama

pytestmark = pytest.mark.anyio


def test_full_name_adds_latest_tag_only_when_missing():
    assert full_name("nomic-embed-text") == "nomic-embed-text:latest"
    assert full_name("llama3.2:3b") == "llama3.2:3b"
    assert full_name("hf.co/org/model") == "hf.co/org/model:latest"
    assert full_name("hf.co/org/model:Q4_K_M") == "hf.co/org/model:Q4_K_M"


async def test_list_models_reports_what_each_model_can_do():
    models = {m.name: m for m in await FakeOllama().client().list_models()}

    assert list(models) == ["llama3.2:3b", "nomic-embed-text:latest", "qwen2.5vl:3b"]

    assert models["llama3.2:3b"].can_chat and not models["llama3.2:3b"].has_vision
    assert models["qwen2.5vl:3b"].can_chat and models["qwen2.5vl:3b"].has_vision
    assert models["nomic-embed-text:latest"].is_embedding
    assert not models["nomic-embed-text:latest"].can_chat

    assert models["llama3.2:3b"].parameter_size == "3B"
    assert models["llama3.2:3b"].quantization == "Q4_K_M"


async def test_capabilities_are_looked_up_once_per_model():
    fake = FakeOllama()
    client = fake.client()

    await client.list_models()
    await client.list_models()

    assert len(fake.calls_to("/api/show")) == len(fake.models)


async def test_unreachable_ollama_raises_a_clear_error():
    client = FakeOllama(reachable=False).client()

    with pytest.raises(OllamaError, match="Can't reach Ollama"):
        await client.list_models()


async def test_use_only_unloads_every_other_model():
    # Search runs inside the app now, so an embedding model left in Ollama is unloaded too.
    fake = FakeOllama(loaded=["llama3.2:3b", "nomic-embed-text:latest"])
    client = fake.client(keep_alive="15m")

    unloaded = await client.use_only("qwen2.5vl:3b")

    assert unloaded == ["llama3.2:3b", "nomic-embed-text:latest"]
    assert fake.loaded == ["qwen2.5vl:3b"]
    load_call = fake.calls_to("/api/generate")[-1]
    assert load_call == {"model": "qwen2.5vl:3b", "keep_alive": "15m"}


async def test_use_only_with_the_model_already_loaded_unloads_nothing():
    fake = FakeOllama(loaded=["llama3.2:3b"])

    unloaded = await fake.client().use_only("llama3.2:3b")

    assert unloaded == []
    assert fake.loaded == ["llama3.2:3b"]


async def test_use_only_matches_names_with_or_without_latest_tag():
    fake = FakeOllama(models={"mistral:latest": ["completion"]}, loaded=["mistral:latest"])

    unloaded = await fake.client().use_only("mistral")

    assert unloaded == []


async def test_has_model_accepts_names_with_or_without_tag():
    client = FakeOllama().client()

    assert await client.has_model("nomic-embed-text")
    assert await client.has_model("llama3.2:3b")
    assert not await client.has_model("mistral")
