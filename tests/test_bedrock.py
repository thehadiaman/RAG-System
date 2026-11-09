import base64
import io
import json

import boto3
import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber

from app.llm.bedrock import (
    BedrockClient,
    BedrockError,
    build_converse_request,
    build_qwen_body,
    build_qwen_prompt,
    qwen_text,
)
from app.llm.messages import ChatMessage, ImageInput, solid_png

ARN = "arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123"
NOVA = "amazon.nova-lite-v1:0"
RED = ImageInput(solid_png(4, 4, (255, 0, 0)), "png")
BLUE = ImageInput(solid_png(4, 4, (0, 0, 255)), "png")


def runtime():
    return boto3.client(
        "bedrock-runtime", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
    )


def make_client(client, **kwargs) -> BedrockClient:
    return BedrockClient(
        region="us-east-1", nova_model_id=NOVA, qwen_model_arn=ARN, client=client, retry_delay=0, **kwargs
    )


def json_body(data: dict) -> StreamingBody:
    raw = json.dumps(data).encode()
    return StreamingBody(io.BytesIO(raw), len(raw))


def client_error(code: str, message: str = "") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "Operation")


# ---- Request building ----


def test_converse_request_puts_images_before_text():
    messages = [ChatMessage("user", "What is this?", images=(RED,))]

    request = build_converse_request(NOVA, messages, "Be brief.", max_tokens=50, temperature=0.1)

    assert request == {
        "modelId": NOVA,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": RED.data}}},
                    {"text": "What is this?"},
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 50, "temperature": 0.1},
        "system": [{"text": "Be brief."}],
    }


def test_converse_request_leaves_out_empty_system():
    request = build_converse_request(NOVA, [ChatMessage("user", "Hi")], None, 10, 0)
    assert "system" not in request


def test_qwen_prompt_follows_the_chat_template():
    messages = [
        ChatMessage("user", "Describe the chart.", images=(RED,)),
        ChatMessage("assistant", "A bar chart."),
        ChatMessage("user", "Which bar is tallest?"),
    ]

    assert build_qwen_prompt(messages, "You read charts.") == (
        "<|im_start|>system\nYou read charts.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the chart.<|im_end|>\n"
        "<|im_start|>assistant\nA bar chart.<|im_end|>\n"
        "<|im_start|>user\nWhich bar is tallest?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_qwen_prompt_uses_default_system_when_none_given():
    assert build_qwen_prompt([ChatMessage("user", "Hi")], None).startswith(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    )


def test_qwen_body_sends_images_as_base64_in_prompt_order():
    messages = [ChatMessage("user", "Compare.", images=(RED, BLUE))]

    body = build_qwen_body(messages, None, max_tokens=100, temperature=0.0)

    assert body["images"] == [base64.b64encode(RED.data).decode(), base64.b64encode(BLUE.data).decode()]
    assert body["prompt"].count("<|image_pad|>") == 2
    assert body["max_gen_len"] == 100
    assert body["temperature"] == 0.0


def test_qwen_body_without_images_has_no_images_field():
    assert "images" not in build_qwen_body([ChatMessage("user", "Hi")], None, 10, 0)


@pytest.mark.parametrize(
    "body",
    [
        {"choices": [{"text": "Red"}]},
        {"outputs": [{"text": "Red"}]},
        {"generation": "Red"},
        {"choices": [{"message": {"content": "Red"}}]},
        {"choices": [{"delta": {"content": "Red"}}]},
    ],
)
def test_qwen_text_handles_every_reply_shape(body):
    assert qwen_text(body) == "Red"


def test_qwen_text_rejects_unknown_reply():
    with pytest.raises(BedrockError, match="Unexpected reply"):
        qwen_text({"something": "else"})


# ---- Calls (AWS simulated with botocore's Stubber) ----



@pytest.mark.anyio
async def test_nova_sends_converse_request_and_returns_text():
    client = runtime()
    messages = [ChatMessage("user", "Colour?", images=(RED,))]
    with Stubber(client) as stub:
        stub.add_response(
            "converse",
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "Red"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 1, "totalTokens": 11},
                "metrics": {"latencyMs": 100},
            },
            expected_params=build_converse_request(NOVA, messages, None, 10, 0),
        )

        reply = await make_client(client).nova(messages, max_tokens=10, temperature=0)

    assert reply == "Red"


@pytest.mark.anyio
async def test_nova_access_denied_says_how_to_fix_it():
    client = runtime()
    with Stubber(client) as stub:
        stub.add_client_error("converse", "AccessDeniedException", "not authorized", 403)

        with pytest.raises(BedrockError, match="model access is enabled in the Bedrock console"):
            await make_client(client).nova([ChatMessage("user", "Hi")])


@pytest.mark.anyio
async def test_qwen_invokes_imported_model_and_returns_text():
    client = runtime()
    messages = [ChatMessage("user", "Colour?", images=(RED,))]
    with Stubber(client) as stub:
        stub.add_response(
            "invoke_model",
            {"body": json_body({"choices": [{"text": "Red."}]}), "contentType": "application/json"},
            expected_params={
                "modelId": ARN,
                "body": ANY,
                "contentType": "application/json",
                "accept": "application/json",
            },
        )

        reply = await make_client(client).qwen(messages)

    assert reply == "Red."


@pytest.mark.anyio
async def test_qwen_waits_while_the_imported_model_wakes_up():
    client = runtime()
    with Stubber(client) as stub:
        stub.add_client_error("invoke_model", "ModelNotReadyException", "model is starting", 429)
        stub.add_client_error("invoke_model", "ModelNotReadyException", "model is starting", 429)
        stub.add_response("invoke_model", {"body": json_body({"outputs": [{"text": "Awake"}]}), "contentType": "x"})

        reply = await make_client(client).qwen([ChatMessage("user", "Hi")])

    assert reply == "Awake"


@pytest.mark.anyio
async def test_qwen_gives_up_when_the_model_takes_too_long_to_wake():
    client = runtime()
    with Stubber(client) as stub:
        stub.add_client_error("invoke_model", "ModelNotReadyException", "model is starting", 429)

        with pytest.raises(BedrockError, match="still starting up"):
            await make_client(client, cold_start_wait=0).qwen([ChatMessage("user", "Hi")])


# ---- Streaming (boto3 event streams replaced by plain lists) ----


class FakeRuntime:
    def __init__(self, events=None, error: Exception | None = None):
        self.events = events or []
        self.error = error

    def _events(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event

    def converse_stream(self, **kwargs):
        if self.error:
            raise self.error
        return {"stream": self._events()}

    def invoke_model_with_response_stream(self, **kwargs):
        if self.error:
            raise self.error
        return {"body": self._events()}

    def converse(self, **kwargs):
        raise self.error


def chunk(data: dict) -> dict:
    return {"chunk": {"bytes": json.dumps(data).encode()}}


@pytest.mark.anyio
async def test_nova_stream_yields_text_as_it_arrives():
    fake = FakeRuntime(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": "lo"}, "contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {}}},
        ]
    )

    pieces = [p async for p in make_client(fake).nova_stream([ChatMessage("user", "Hi")])]

    assert pieces == ["Hel", "lo"]


@pytest.mark.anyio
async def test_qwen_stream_reads_both_chunk_formats():
    fake = FakeRuntime([chunk({"choices": [{"text": "Hel"}]}), chunk({"generation": "lo"}), chunk({"choices": []})])

    pieces = [p async for p in make_client(fake).qwen_stream([ChatMessage("user", "Hi")])]

    assert pieces == ["Hel", "lo"]


@pytest.mark.anyio
async def test_error_in_the_middle_of_a_stream_is_explained():
    fake = FakeRuntime([chunk({"choices": [{"text": "Hel"}]}), client_error("ThrottlingException", "slow down")])

    pieces = []
    with pytest.raises(BedrockError, match="limiting requests"):
        async for piece in make_client(fake).qwen_stream([ChatMessage("user", "Hi")]):
            pieces.append(piece)

    assert pieces == ["Hel"]


@pytest.mark.anyio
async def test_stopping_a_stream_early_is_clean():
    fake = FakeRuntime([chunk({"choices": [{"text": str(i)}]}) for i in range(100)])

    stream = make_client(fake).qwen_stream([ChatMessage("user", "Hi")])
    first = await anext(stream)
    await stream.aclose()

    assert first == "0"


@pytest.mark.anyio
async def test_missing_keys_are_explained():
    fake = FakeRuntime(error=NoCredentialsError())

    with pytest.raises(BedrockError, match="AWS keys are not set"):
        await make_client(fake).nova([ChatMessage("user", "Hi")])


# ---- What is configured ----


@pytest.fixture
def no_aws_keys(monkeypatch, tmp_path):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def test_problems_without_keys(no_aws_keys):
    client = BedrockClient(region="us-east-1", nova_model_id=NOVA, qwen_model_arn=ARN)

    assert "AWS keys are not set" in client.nova_problem()
    assert "AWS keys are not set" in client.qwen_problem()


def test_problems_without_qwen_arn(no_aws_keys):
    client = BedrockClient(
        region="us-east-1", nova_model_id=NOVA, qwen_model_arn="", access_key_id="a", secret_access_key="b"
    )

    assert client.nova_problem() is None
    assert client.qwen_problem() == "BEDROCK_QWEN_VL_MODEL_ARN is not set in .env"


def test_keys_from_the_environment_count(no_aws_keys, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "a")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "b")

    assert BedrockClient(region="us-east-1", qwen_model_arn=ARN).qwen_problem() is None
