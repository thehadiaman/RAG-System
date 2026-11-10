"""In-memory stand-in for the Ollama HTTP API, so tests don't need real models."""

import json
import math
import re
import zlib

import httpx2

from app.llm.ollama import OllamaClient, full_name

DEFAULT_MODELS = {
    "llama3.2:3b": ["completion", "tools"],
    "qwen2.5vl:3b": ["completion", "vision"],
    "nomic-embed-text:latest": ["embedding"],
}


def fake_vector(text: str, dims: int = 16) -> list[float]:
    """Bag-of-words vector: texts sharing words point the same way. Task prefixes are ignored."""
    if text.startswith("search_"):
        text = text.split(": ", 1)[-1]
    vector = [0.0] * dims
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[zlib.crc32(word.encode()) % dims] += 1.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


FIGURE_REPLY = """Type: chart
Title: Revenue by year
Description: A bar chart of revenue from 2018 to 2025.
Data:
| Year | Revenue (AED m) |
|---|---|
| 2018 | 120 |
| 2025 | 270 |"""

SCANNED_REPLY = "# Scanned page\n\nTotal revenue was AED 5 million."


def default_chat_reply(body: dict) -> str:
    prompt = body["messages"][-1]["content"]
    return SCANNED_REPLY if "scanned PDF" in prompt else FIGURE_REPLY


class FakeOllama:
    def __init__(self, models: dict[str, list[str]] | None = None, loaded=(), reachable=True):
        self.models = dict(DEFAULT_MODELS if models is None else models)
        self.loaded = list(loaded)
        self.reachable = reachable
        self.calls: list[tuple[str, str, dict]] = []
        self.chat_reply = default_chat_reply
        self.chat_failures = 0  # the next this-many chat calls fail

    def client(self, **kwargs) -> OllamaClient:
        return OllamaClient("http://ollama.test", transport=httpx2.MockTransport(self.handle), **kwargs)

    def calls_to(self, path: str) -> list[dict]:
        return [body for _, p, body in self.calls if p == path]

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if not self.reachable:
            raise httpx2.ConnectError("Connection refused", request=request)

        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.method, request.url.path, body))
        route = (request.method, request.url.path)

        if route == ("GET", "/api/version"):
            return httpx2.Response(200, json={"version": "0.30.10"})

        if route == ("GET", "/api/tags"):
            return httpx2.Response(200, json={"models": [self._tag(name) for name in self.models]})

        if route == ("GET", "/api/ps"):
            return httpx2.Response(200, json={"models": [{"name": n, "model": n} for n in self.loaded]})

        if route == ("POST", "/api/show"):
            name = full_name(body["model"])
            if name not in self.models:
                return self._not_found(name)
            return httpx2.Response(200, json={"capabilities": self.models[name]})

        if route == ("POST", "/api/embed"):
            name = full_name(body["model"])
            if name not in self.models:
                return self._not_found(name)
            if name not in self.loaded:
                self.loaded.append(name)
            return httpx2.Response(200, json={"model": name, "embeddings": [fake_vector(t) for t in body["input"]]})

        if route == ("POST", "/api/chat"):
            name = full_name(body["model"])
            if name not in self.models:
                return self._not_found(name)
            if self.chat_failures > 0:
                self.chat_failures -= 1
                return httpx2.Response(500, json={"error": "model runner crashed"})
            if name not in self.loaded:
                self.loaded.append(name)
            message = {"role": "assistant", "content": self.chat_reply(body)}
            return httpx2.Response(200, json={"model": name, "message": message, "done": True})

        if route == ("POST", "/api/generate"):
            name = full_name(body["model"])
            if name not in self.models:
                return self._not_found(name)
            if body.get("keep_alive") == 0:
                if name in self.loaded:
                    self.loaded.remove(name)
                return httpx2.Response(200, json={"model": name, "done": True, "done_reason": "unload"})
            if name not in self.loaded:
                self.loaded.append(name)
            return httpx2.Response(200, json={"model": name, "done": True, "done_reason": "load"})

        return httpx2.Response(404, json={"error": "not found"})

    @staticmethod
    def _tag(name: str) -> dict:
        return {
            "name": name,
            "model": name,
            "size": 2_000_000_000,
            "digest": f"sha256-{name}",
            "details": {"family": "test", "parameter_size": "3B", "quantization_level": "Q4_K_M"},
        }

    @staticmethod
    def _not_found(name: str) -> httpx2.Response:
        return httpx2.Response(404, json={"error": f'model "{name}" not found, try pulling it first'})
