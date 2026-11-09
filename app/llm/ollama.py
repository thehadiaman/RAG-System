"""Client for the Ollama server running on this machine.

The GPU only has room for one chat or vision model at a time, so `use_only()`
unloads every other model (except ones you ask to keep, like the embedder)
before loading the model that is about to be used.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx2

log = logging.getLogger(__name__)

# Loading a large model from disk can take a while.
LOAD_TIMEOUT = 300.0


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
    ) -> None:
        self._http = httpx2.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)
        self.keep_alive = keep_alive
        self._capabilities_cache: dict[str, tuple[str, ...]] = {}
        self._switch_lock = asyncio.Lock()

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

    async def loaded_models(self) -> list[str]:
        """Names of the models currently in memory."""
        data = await self._request("GET", "/api/ps")
        return [item["name"] for item in data.get("models", [])]

    async def load(self, name: str) -> None:
        # A generate request without a prompt only loads the model.
        await self._request(
            "POST",
            "/api/generate",
            json={"model": name, "keep_alive": self.keep_alive},
            timeout=LOAD_TIMEOUT,
        )

    async def unload(self, name: str) -> None:
        await self._request("POST", "/api/generate", json={"model": name, "keep_alive": 0})

    async def use_only(self, name: str, keep: tuple[str, ...] = ()) -> list[str]:
        """Make `name` the only model in memory, apart from the models in `keep`.

        Returns the names of the models that were unloaded.
        """
        wanted = full_name(name)
        spared = {wanted, *(full_name(k) for k in keep)}

        # One switch at a time, so two requests can't load two models at once.
        async with self._switch_lock:
            unloaded = []
            for loaded in await self.loaded_models():
                if full_name(loaded) in spared:
                    continue
                log.info("Unloading %s to make room for %s", loaded, wanted)
                await self.unload(loaded)
                unloaded.append(loaded)
            await self.load(wanted)
            return unloaded
