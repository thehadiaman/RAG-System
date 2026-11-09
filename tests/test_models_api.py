import pytest

from tests.fake_bedrock import FakeBedrock
from tests.fake_ollama import FakeOllama


def test_lists_chat_models_and_hides_embedders(client):
    data = client.get("/api/models").json()

    assert [m["id"] for m in data["models"]] == ["ollama:llama3.2:3b", "ollama:qwen2.5vl:3b"]
    assert data["providers"]["ollama"] == {"available": True, "version": "0.30.10", "error": None}


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
    assert data["providers"]["ollama"]["available"] is True


@pytest.mark.parametrize("fake_ollama", [FakeOllama(reachable=False)])
def test_ollama_down_still_answers_with_a_reason(client):
    response = client.get("/api/models")

    assert response.status_code == 200
    data = response.json()
    assert data["models"] == []
    assert data["providers"]["ollama"]["available"] is False
    assert "Is Ollama running?" in data["providers"]["ollama"]["error"]


def test_aws_qwen_hidden_until_configured(client):
    data = client.get("/api/models").json()

    assert not [m for m in data["models"] if m["provider"] == "aws"]
    assert data["providers"]["aws"]["available"] is False
    assert "AWS keys are not set" in data["providers"]["aws"]["error"]


@pytest.mark.parametrize("fake_bedrock", [FakeBedrock(configured=True)])
def test_aws_qwen_listed_when_configured(client):
    data = client.get("/api/models").json()

    aws = data["models"][-1]
    assert aws["id"] == "aws:qwen2.5-vl"
    assert aws["label"] == "Qwen2.5-VL (AWS)"
    assert aws["vision"] is True
    assert data["providers"]["aws"] == {"available": True, "version": None, "error": None}


@pytest.mark.parametrize(
    ("fake_ollama", "fake_bedrock"), [(FakeOllama(reachable=False), FakeBedrock(configured=True))]
)
def test_aws_model_still_listed_when_ollama_is_down(client):
    data = client.get("/api/models").json()

    assert [m["id"] for m in data["models"]] == ["aws:qwen2.5-vl"]
