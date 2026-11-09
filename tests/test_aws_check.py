import pytest

from tests.fake_bedrock import FakeBedrock


def test_reports_not_configured_without_calling_aws(client, fake_bedrock):
    data = client.get("/api/aws/check").json()

    assert data["region"] == "us-east-1"
    for result in (data["nova_lite"], data["qwen_vl"]):
        assert result["ok"] is False
        assert result["configured"] is False
        assert "AWS keys are not set" in result["error"]
    assert fake_bedrock.calls == []


@pytest.mark.parametrize("fake_bedrock", [FakeBedrock(configured=True, nova_reply="Red", qwen_reply=" Blue ")])
def test_asks_each_model_about_a_red_square(client, fake_bedrock):
    data = client.get("/api/aws/check").json()

    assert data["nova_lite"]["ok"] is True
    assert data["nova_lite"]["saw_red"] is True
    assert data["qwen_vl"]["ok"] is True
    assert data["qwen_vl"]["saw_red"] is False
    assert data["qwen_vl"]["reply"] == "Blue"

    assert [model for model, _, _ in fake_bedrock.calls] == ["nova", "qwen"]
    _, messages, kwargs = fake_bedrock.calls[0]
    assert messages[0].images[0].format == "png"
    assert kwargs["max_tokens"] == 10


@pytest.mark.parametrize("fake_bedrock", [FakeBedrock(configured=True, error="AWS refused access to Nova Lite.")])
def test_reports_aws_errors(client):
    data = client.get("/api/aws/check").json()

    assert data["nova_lite"]["ok"] is False
    assert data["nova_lite"]["configured"] is True
    assert data["nova_lite"]["error"] == "AWS refused access to Nova Lite."
