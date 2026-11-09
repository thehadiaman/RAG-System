import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    path = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(path))
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def client(data_dir):
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
