"""Models on Amazon Bedrock.

- Nova Lite is a standard Bedrock model, called with the Converse API.
- Qwen2.5-VL is imported with Bedrock Custom Model Import. Imported models only
  support InvokeModel, so the prompt is written in Qwen's own chat format and
  images are sent as base64.

boto3 is blocking, so every call runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import aclosing
from functools import cached_property

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, HTTPClientError, NoCredentialsError
from botocore.exceptions import ConnectionError as BotoConnectionError

from app.llm.messages import ChatMessage

log = logging.getLogger(__name__)

QWEN_DEFAULT_SYSTEM = "You are a helpful assistant."
QWEN_IMAGE_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"

NOVA_LABEL = "Nova Lite"
QWEN_LABEL = "Qwen2.5-VL (AWS)"


class BedrockError(Exception):
    """A Bedrock call failed. The message says what to fix."""


# ---- Request building and response parsing (plain functions, easy to test) ----


def build_converse_request(
    model_id: str,
    messages: list[ChatMessage],
    system: str | None,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Converse API request. Images go before the text, as Nova recommends."""
    request = {
        "modelId": model_id,
        "messages": [
            {
                "role": m.role,
                "content": [
                    *({"image": {"format": img.format, "source": {"bytes": img.data}}} for img in m.images),
                    {"text": m.text},
                ],
            }
            for m in messages
        ],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        request["system"] = [{"text": system}]
    return request


def build_qwen_prompt(messages: list[ChatMessage], system: str | None) -> str:
    """Qwen2.5-VL's chat template, written out by hand (no `transformers` needed)."""
    parts = [f"<|im_start|>system\n{system or QWEN_DEFAULT_SYSTEM}<|im_end|>\n"]
    for m in messages:
        parts.append(f"<|im_start|>{m.role}\n{QWEN_IMAGE_TOKENS * len(m.images)}{m.text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_qwen_body(
    messages: list[ChatMessage],
    system: str | None,
    max_tokens: int,
    temperature: float,
) -> dict:
    body = {
        "prompt": build_qwen_prompt(messages, system),
        "max_gen_len": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }
    # Images are matched, in order, to the image tokens in the prompt.
    images = [base64.b64encode(img.data).decode("ascii") for m in messages for img in m.images]
    if images:
        body["images"] = images
    return body


def qwen_text(body: dict) -> str:
    """Text from an imported-model reply. Handles every shape Bedrock has been seen to return."""
    for key in ("choices", "outputs"):
        if key in body:
            pieces = []
            for item in body[key]:
                if item.get("text") is not None:
                    pieces.append(item["text"])
                else:  # OpenAI-style message or streaming delta
                    message = item.get("message") or item.get("delta") or {}
                    pieces.append(message.get("content") or "")
            return "".join(pieces)
    if "generation" in body:
        return body["generation"] or ""
    raise BedrockError(f"Unexpected reply from the Qwen model (fields: {', '.join(body) or 'none'})")


def converse_text(response: dict) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(block.get("text", "") for block in content)


def converse_stream_pieces(events: Iterable[dict]) -> Iterator[str]:
    for event in events:
        text = event.get("contentBlockDelta", {}).get("delta", {}).get("text")
        if text:
            yield text


def qwen_stream_pieces(events: Iterable[dict]) -> Iterator[str]:
    for event in events:
        chunk = event.get("chunk")
        if not chunk:
            continue
        text = qwen_text(json.loads(chunk["bytes"]))
        if text:
            yield text


# ---- Client ----


class BedrockClient:
    def __init__(
        self,
        region: str,
        nova_model_id: str = "",
        qwen_model_arn: str = "",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        cold_start_wait: float = 300.0,
        retry_delay: float = 10.0,
        client=None,
    ) -> None:
        self.region = region
        self.nova_model_id = nova_model_id
        self.qwen_model_arn = qwen_model_arn
        self.cold_start_wait = cold_start_wait
        self.retry_delay = retry_delay
        self._keys = {
            "aws_access_key_id": access_key_id or None,
            "aws_secret_access_key": secret_access_key or None,
            "aws_session_token": session_token or None,
        }
        if client is not None:
            self.__dict__["_runtime"] = client  # fills the cached_property below

    @cached_property
    def _runtime(self):
        # Created on first use, so the app starts fine without any AWS setup.
        config = Config(
            connect_timeout=10,
            read_timeout=300,
            retries={"mode": "standard", "max_attempts": 3},
        )
        return boto3.client("bedrock-runtime", region_name=self.region, config=config, **self._keys)

    # -- What is usable right now (no network calls) --

    def has_credentials(self) -> bool:
        if self._keys["aws_access_key_id"] and self._keys["aws_secret_access_key"]:
            return True
        credentials = boto3.Session().get_credentials()
        return bool(credentials and credentials.access_key)

    def _problem(self, setting_name: str, value: str) -> str | None:
        if not value:
            return f"{setting_name} is not set in .env"
        if not self.has_credentials():
            return "AWS keys are not set (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env)"
        return None

    def nova_problem(self) -> str | None:
        """None if Nova Lite is ready to use, otherwise why it isn't."""
        return self._problem("BEDROCK_NOVA_LITE_MODEL_ID", self.nova_model_id)

    def qwen_problem(self) -> str | None:
        """None if Qwen2.5-VL on AWS is ready to use, otherwise why it isn't."""
        return self._problem("BEDROCK_QWEN_VL_MODEL_ARN", self.qwen_model_arn)

    # -- Nova Lite --

    async def nova(
        self,
        messages: list[ChatMessage],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        request = build_converse_request(self.nova_model_id, messages, system, max_tokens, temperature)
        response = await self._call(NOVA_LABEL, self._runtime.converse, **request)
        return converse_text(response)

    async def nova_stream(
        self,
        messages: list[ChatMessage],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        request = build_converse_request(self.nova_model_id, messages, system, max_tokens, temperature)
        response = await self._call(NOVA_LABEL, self._runtime.converse_stream, **request)
        async with aclosing(self._stream(NOVA_LABEL, lambda: converse_stream_pieces(response["stream"]))) as pieces:
            async for piece in pieces:
                yield piece

    # -- Qwen2.5-VL (imported model) --

    async def qwen(
        self,
        messages: list[ChatMessage],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        body = build_qwen_body(messages, system, max_tokens, temperature)
        response = await self._call(
            QWEN_LABEL,
            self._runtime.invoke_model,
            modelId=self.qwen_model_arn,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = await asyncio.to_thread(response["body"].read)
        return qwen_text(json.loads(payload))

    async def qwen_stream(
        self,
        messages: list[ChatMessage],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        body = build_qwen_body(messages, system, max_tokens, temperature)
        response = await self._call(
            QWEN_LABEL,
            self._runtime.invoke_model_with_response_stream,
            modelId=self.qwen_model_arn,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        async with aclosing(self._stream(QWEN_LABEL, lambda: qwen_stream_pieces(response["body"]))) as pieces:
            async for piece in pieces:
                yield piece

    # -- Plumbing --

    async def _call(self, label: str, method: Callable, **kwargs) -> dict:
        """Run a boto3 call in a thread. Waits while an imported model is starting up."""
        started = time.monotonic()
        while True:
            try:
                return await asyncio.to_thread(method, **kwargs)
            except ClientError as exc:
                if _error_code(exc) != "ModelNotReadyException":
                    raise self._explain(label, exc) from exc
                waited = time.monotonic() - started
                if waited >= self.cold_start_wait:
                    raise BedrockError(
                        f"{label} is still starting up after {waited:.0f} s. "
                        "Imported models sleep when idle; try again in a minute."
                    ) from exc
                log.info("%s is starting up (waited %.0f s so far)", label, waited)
                await asyncio.sleep(self.retry_delay)
            except (BotoCoreError, NoCredentialsError) as exc:
                raise self._explain(label, exc) from exc

    async def _stream(self, label: str, pieces: Callable[[], Iterator[str]]) -> AsyncIterator[str]:
        """Read a blocking boto3 event stream in a thread and hand pieces to async code."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        done = object()

        def put(item) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # event loop already closed
                pass

        def worker() -> None:
            try:
                for piece in pieces():
                    if stop.is_set():
                        break
                    put(piece)
            except Exception as exc:  # noqa: BLE001 - handed to the async side
                put(exc)
            finally:
                put(done)

        loop.run_in_executor(None, worker)
        try:
            while (item := await queue.get()) is not done:
                if isinstance(item, Exception):
                    if isinstance(item, BedrockError):
                        raise item
                    raise self._explain(label, item) from item
                yield item
        finally:
            stop.set()  # the reader stopped early (e.g. user pressed stop)

    def _explain(self, label: str, exc: Exception) -> BedrockError:
        """Turn an AWS error into a message that says what to fix."""
        if isinstance(exc, NoCredentialsError):
            return BedrockError("AWS keys are not set (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env)")
        if isinstance(exc, ClientError):
            code = _error_code(exc)
            message = exc.response.get("Error", {}).get("Message", "")
            if code == "AccessDeniedException":
                return BedrockError(
                    f"AWS refused access to {label}. Check that model access is enabled in the "
                    f"Bedrock console for {self.region} and that your IAM user may call bedrock:InvokeModel. "
                    f"({message})"
                )
            if code == "ResourceNotFoundException":
                return BedrockError(
                    f"{label} wasn't found in {self.region}. Check the model ID or ARN and the region. ({message})"
                )
            if code in ("UnrecognizedClientException", "InvalidSignatureException", "ExpiredTokenException"):
                return BedrockError(f"AWS rejected your keys (wrong or expired). ({message})")
            if code == "ThrottlingException":
                return BedrockError(f"AWS is limiting requests to {label}; try again shortly.")
            if code == "ValidationException":
                return BedrockError(f"{label} rejected the request: {message}")
            return BedrockError(f"{label} failed ({code}): {message}")
        if isinstance(exc, (BotoConnectionError, HTTPClientError)):
            return BedrockError(f"Can't reach AWS Bedrock in {self.region}: {exc}")
        return BedrockError(f"{label} failed: {exc}")


def _error_code(exc: ClientError) -> str:
    return exc.response.get("Error", {}).get("Code", "")
