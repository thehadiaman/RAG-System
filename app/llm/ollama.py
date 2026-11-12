"""Client for the Ollama server running on this machine.

The GPU only has room for one chat or vision model at a time, so `use_only()`
unloads every other model (except ones you ask to keep, like the embedder)
before loading the model that is about to be used.

Anything that uses a local chat or vision model holds the `gpu` lock while it
does, so a chat answer and the PDF-reading step take turns instead of
swapping models under each other.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx2

from app.llm.messages import ChatMessage

log = logging.getLogger(__name__)

# Loading a large model from disk, or embedding a big batch on CPU, can take a while.
LOAD_TIMEOUT = 300.0
EMBED_TIMEOUT = 300.0
# A vision model partly on CPU can take minutes for one large image.
CHAT_TIMEOUT = 600.0


class OllamaError(Exception):
    """Ollama couldn't be reached, or it returned an error."""


def full_name(name: str) -> str:
    """Ollama treats "llama3" and "llama3:latest" as the same model; normalise to the long form."""
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


@dataclass(frozen=True)
class OllamaModel:
    name: str
    size_bytes: int
    family: str
    parameter_size: str
    quantization: str
    capabilities: tuple[str, ...]

    @property
    def is_embedding(self) -> bool:
        if self.capabilities:
            return "embedding" in self.capabilities
        return "embed" in self.name  # fallback if Ollama didn't report capabilities

    @property
    def can_chat(self) -> bool:
        if self.capabilities:
            return "completion" in self.capabilities
        return not self.is_embedding

    @property
    def has_vision(self) -> bool:
        return "vision" in self.capabilities


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        keep_alive: str = "10m",
        timeout: float = 30.0,
        transport: httpx2.AsyncBaseTransport | None = None,
        keep_loaded: tuple[str, ...] = (),
    ) -> None:
        self._http = httpx2.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)
        self.keep_alive = keep_alive
        # Models that use_only() never unloads (the embedder, which runs on CPU).
        self.keep_loaded = tuple(full_name(name) for name in keep_loaded)
        self._capabilities_cache: dict[str, tuple[str, ...]] = {}
        self._switch_lock = asyncio.Lock()
        # Held while a local chat or vision model is in use (see the module docstring).
        self.gpu = asyncio.Lock()

    @property
    def base_url(self) -> str:
        return str(self._http.base_url)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = await self._http.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx2.HTTPStatusError as exc:
            body = exc.response.text[:300]
            raise OllamaError(f"Ollama returned {exc.response.status_code} for {path}: {body}") from exc
        except httpx2.HTTPError as exc:
            detail = str(exc) or type(exc).__name__
            raise OllamaError(f"Can't reach Ollama at {self.base_url} ({detail}). Is Ollama running?") from exc
        return response.json()

    async def version(self) -> str:
        data = await self._request("GET", "/api/version")
        return data.get("version", "unknown")

    async def list_models(self) -> list[OllamaModel]:
        """Every installed model, with what it can do (chat, vision, embedding)."""
        data = await self._request("GET", "/api/tags")
        models = []
        for item in data.get("models", []):
            details = item.get("details") or {}
            models.append(
                OllamaModel(
                    name=item["name"],
                    size_bytes=item.get("size", 0),
                    family=details.get("family", ""),
                    parameter_size=details.get("parameter_size", ""),
                    quantization=details.get("quantization_level", ""),
                    capabilities=await self._capabilities(item["name"], item.get("digest", "")),
                )
            )
        return sorted(models, key=lambda m: m.name)

    async def _capabilities(self, name: str, digest: str) -> tuple[str, ...]:
        # Cached by digest, so a re-pulled model with new weights is looked up again.
        key = digest or name
        if key not in self._capabilities_cache:
            try:
                info = await self._request("POST", "/api/show", json={"model": name})
            except OllamaError as exc:
                log.warning("Couldn't read capabilities of %s: %s", name, exc)
                return ()
            self._capabilities_cache[key] = tuple(info.get("capabilities") or ())
        return self._capabilities_cache[key]

    async def has_model(self, name: str) -> bool:
        data = await self._request("GET", "/api/tags")
        return any(full_name(item["name"]) == full_name(name) for item in data.get("models", []))

    async def embed(
        self,
        model: str,
        inputs: list[str],
        keep_alive: str | int | None = None,
        options: dict | None = None,
    ) -> list[list[float]]:
        payload: dict = {"model": model, "input": inputs}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if options:
            payload["options"] = options
        data = await self._request("POST", "/api/embed", json=payload, timeout=EMBED_TIMEOUT)
        return data["embeddings"]

    async def loaded_models(self) -> list[str]:
        """Names of the models currently in memory."""
        data = await self._request("GET", "/api/ps")
        return [item["name"] for item in data.get("models", [])]

    async def load(self, name: str, options: dict | None = None) -> None:
        # A generate request without a prompt only loads the model. Pass the same
        # options (like num_ctx) later calls will use, or Ollama reloads it then.
        payload: dict = {"model": name, "keep_alive": self.keep_alive}
        if options:
            payload["options"] = options
        await self._request("POST", "/api/generate", json=payload, timeout=LOAD_TIMEOUT)

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
        """One complete reply (no streaming). Images are sent with the message they belong to."""
        payload: dict = {
            "model": model,
            "messages": _chat_messages(messages, system),
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        if options:
            payload["options"] = options
        data = await self._request("POST", "/api/chat", json=payload, timeout=CHAT_TIMEOUT)
        return data.get("message", {}).get("content", "")

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        system: str | None = None,
        options: dict | None = None,
    ) -> AsyncIterator[str]:
        """The reply as it is generated, piece by piece."""
        payload: dict = {
            "model": model,
            "messages": _chat_messages(messages, system),
            "stream": True,
            "keep_alive": self.keep_alive,
        }
        if options:
            payload["options"] = options
        try:
            async with self._http.stream("POST", "/api/chat", json=payload, timeout=CHAT_TIMEOUT) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")[:300]
                    raise OllamaError(f"Ollama returned {response.status_code} for /api/chat: {body}")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise OllamaError(f"Ollama stopped with an error: {data['error']}")
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        break
        except httpx2.HTTPError as exc:
            detail = str(exc) or type(exc).__name__
            raise OllamaError(f"Can't reach Ollama at {self.base_url} ({detail}). Is Ollama running?") from exc

    async def unload(self, name: str) -> None:
        await self._request("POST", "/api/generate", json={"model": name, "keep_alive": 0})

    async def use_only(self, name: str, keep: tuple[str, ...] = (), options: dict | None = None) -> list[str]:
        """Make `name` the only model in memory, apart from `keep_loaded` and `keep`.

        Returns the names of the models that were unloaded.
        """
        wanted = full_name(name)
        spared = {wanted, *self.keep_loaded, *(full_name(k) for k in keep)}

        # One switch at a time, so two requests can't load two models at once.
        async with self._switch_lock:
            unloaded = []
            for loaded in await self.loaded_models():
                if full_name(loaded) in spared:
                    continue
                log.info("Unloading %s to make room for %s", loaded, wanted)
                await self.unload(loaded)
                unloaded.append(loaded)
            await self.load(wanted, options)
            return unloaded


def _chat_messages(messages: list[ChatMessage], system: str | None) -> list[dict]:
    result = [{"role": "system", "content": system}] if system else []
    for m in messages:
        item: dict = {"role": m.role, "content": m.text}
        if m.images:
            item["images"] = [base64.b64encode(img.data).decode("ascii") for img in m.images]
        result.append(item)
    return result
