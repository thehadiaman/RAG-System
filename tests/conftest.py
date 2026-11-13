import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from app.config import get_settings
from tests.fake_bedrock import FakeBedrock
from tests.fake_embedder import FakeEmbedder
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
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def make_app(data_dir, fake_ollama, fake_bedrock, fake_embedder, monkeypatch):
    """Builds the real app, with its data in a temp folder and fakes for Ollama, AWS and the search model."""
    import app.main as main

    monkeypatch.setenv("VISION_LOCAL_MODEL", "qwen2.5vl:3b")
    monkeypatch.setattr("app.ingest.describe.RETRY_DELAY", 0)
    get_settings.cache_clear()
    monkeypatch.setattr(main, "make_ollama", lambda settings: fake_ollama.client())
    monkeypatch.setattr(main, "make_embedder", lambda settings: fake_embedder)
    monkeypatch.setattr(main, "make_bedrock", lambda settings: fake_bedrock)
    return main.create_app


@pytest.fixture
def client(make_app):
    with TestClient(make_app()) as test_client:
        yield test_client


@pytest.fixture
def live_server(make_app):
    """The app served on a free local port, for browser tests. Yields its base URL."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(make_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("The test server didn't start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=15)
