import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from tests.fake_bedrock import FakeBedrock
from tests.fake_ollama import FakeOllama


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    path = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(path))
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def fake_ollama():
    return FakeOllama()


@pytest.fixture
def fake_bedrock():
    return FakeBedrock()


@pytest.fixture
def client(data_dir, fake_ollama, fake_bedrock, monkeypatch):
    """The real app, with its data in a temp folder and fake Ollama and AWS behind it."""
    import app.main as main

    monkeypatch.setenv("VISION_LOCAL_MODEL", "qwen2.5vl:3b")
    monkeypatch.setattr("app.ingest.describe.RETRY_DELAY", 0)
    get_settings.cache_clear()
    monkeypatch.setattr(
        main, "make_ollama", lambda settings: fake_ollama.client(keep_loaded=(settings.ollama_embed_model,))
    )
    monkeypatch.setattr(main, "make_bedrock", lambda settings: fake_bedrock)

    with TestClient(main.create_app()) as test_client:
        yield test_client
