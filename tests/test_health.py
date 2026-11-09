import pytest

from app import __version__
from app.store.db import MIGRATIONS
from tests.fake_ollama import FakeOllama


def test_health_reports_every_component_ready(client):
    data = client.get("/api/health").json()

    assert data["status"] == "ok"
    assert data["version"] == __version__
    assert data["components"] == {
        "database": {"ok": True, "detail": f"schema version {len(MIGRATIONS)}"},
        "vectors": {"ok": True, "detail": "0 chunks"},
        "embedder": {"ok": True, "detail": "nomic-embed-text"},
    }


@pytest.mark.parametrize("fake_ollama", [FakeOllama(models={"llama3.2:3b": ["completion"]})])
def test_health_is_degraded_without_the_embedding_model(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["components"]["embedder"]["ok"] is False
    assert "ollama pull nomic-embed-text" in data["components"]["embedder"]["detail"]


def test_index_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>PDF RAG</title>" in response.text


def test_unknown_api_route_is_404(client):
    assert client.get("/api/does-not-exist").status_code == 404


def test_startup_creates_the_database_and_vector_store(client, data_dir):
    assert (data_dir / "app.db").is_file()
    assert (data_dir / "chroma").is_dir()
