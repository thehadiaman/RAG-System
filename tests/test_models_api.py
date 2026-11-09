import pytest

from tests.fake_ollama import FakeOllama


def test_lists_chat_models_and_hides_embedders(client):
    data = client.get("/api/models").json()

    assert [m["id"] for m in data["models"]] == ["ollama:llama3.2:3b", "ollama:qwen2.5vl:3b"]
    assert data["providers"]["ollama"] == {"reachable": True, "version": "0.30.10", "error": None}


def test_marks_vision_models_and_loaded_models(client, fake_ollama):
    fake_ollama.loaded = ["qwen2.5vl:3b"]

    models = {m["name"]: m for m in client.get("/api/models").json()["models"]}

    assert models["qwen2.5vl:3b"]["vision"] is True
    assert models["qwen2.5vl:3b"]["loaded"] is True
    assert models["llama3.2:3b"]["vision"] is False
    assert models["llama3.2:3b"]["loaded"] is False
    assert models["llama3.2:3b"]["size_gb"] == 2.0
    assert models["llama3.2:3b"]["provider"] == "ollama"


@pytest.mark.parametrize("fake_ollama", [FakeOllama(models={})])
def test_no_models_installed_gives_empty_list(client):
    data = client.get("/api/models").json()

    assert data["models"] == []
    assert data["providers"]["ollama"]["reachable"] is True


@pytest.mark.parametrize("fake_ollama", [FakeOllama(reachable=False)])
def test_ollama_down_still_answers_with_a_reason(client):
    response = client.get("/api/models")

    assert response.status_code == 200
    data = response.json()
    assert data["models"] == []
    assert data["providers"]["ollama"]["reachable"] is False
    assert "Can't reach Ollama" in data["providers"]["ollama"]["error"]
