import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_ollama
from app.config import get_settings
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
def client(data_dir, fake_ollama):
    from app.main import create_app

    app = create_app()
    ollama = fake_ollama.client()
    app.dependency_overrides[get_ollama] = lambda: ollama
    with TestClient(app) as test_client:
        yield test_client
