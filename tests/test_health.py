from app import __version__


def test_health_reports_ok(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_index_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>PDF RAG</title>" in response.text


def test_unknown_api_route_is_404(client):
    assert client.get("/api/does-not-exist").status_code == 404


def test_data_dir_created_on_startup(client, data_dir):
    assert data_dir.is_dir()
