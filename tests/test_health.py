import pytest

from app import __version__
from app.store.db import MIGRATIONS
from tests.fake_embedder import FakeEmbedder


def test_health_reports_every_component_ready(client):
    data = client.get("/api/health").json()

    assert data["status"] == "ok"
    assert data["version"] == __version__
    assert data["components"] == {
        "database": {"ok": True, "detail": f"schema version {len(MIGRATIONS)}"},
        "vectors": {"ok": True, "detail": "0 chunks"},
        "embedder": {"ok": True, "detail": "fake-search-model (test)"},
    }


@pytest.mark.parametrize("fake_embedder", [FakeEmbedder(problem="The search model's files are missing")])
def test_health_is_degraded_when_the_search_model_is_broken(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["components"]["embedder"]["ok"] is False
    assert data["components"]["embedder"]["detail"] == "The search model's files are missing"


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


@pytest.mark.parametrize(
    ("path", "content_type"),
    [("/js/app.js", "text/javascript"), ("/css/app.css", "text/css"), ("/js/details.js", "text/javascript")],
)
def test_web_ui_files_are_served(client, path, content_type):
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)


@pytest.mark.parametrize("path", ["/", "/js/details.js"])
def test_browsers_check_for_a_newer_web_ui_on_each_load(client, path):
    first = client.get(path)
    assert first.headers["cache-control"] == "no-cache"

    unchanged = client.get(path, headers={"if-none-match": first.headers["etag"]})
    assert unchanged.status_code == 304


def test_unexpected_errors_return_a_clear_message(make_app, monkeypatch):
    from fastapi.testclient import TestClient

    def broken(db):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("app.store.documents.list_all", broken)
    with TestClient(make_app(), raise_server_exceptions=False) as test_client:
        response = test_client.get("/api/documents")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Something went wrong on the server. Details are in the app log: docker compose logs app"
    }


def test_startup_logs_what_is_ready(make_app, caplog):
    import time

    from fastapi.testclient import TestClient

    caplog.set_level("INFO", logger="app")
    with TestClient(make_app()):
        deadline = time.monotonic() + 10
        while "Qwen2.5-VL on AWS" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.05)

    assert "Search model fake-search-model (test): ready" in caplog.text
    assert "Charts read by Qwen2.5-VL on this computer (qwen2.5vl:3b): ready" in caplog.text
    assert "Charts read by Amazon Nova Lite (AWS): AWS keys are not set" in caplog.text
