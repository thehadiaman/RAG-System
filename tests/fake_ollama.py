"""In-memory stand-in for the Ollama HTTP API, so tests don't need real models."""

import json

import httpx2

from app.llm.ollama import OllamaClient

DEFAULT_MODELS = {
    "llama3.2:3b": ["completion", "tools"],
    "qwen2.5vl:3b": ["completion", "vision"],
    "nomic-embed-text:latest": ["embedding"],
}


class FakeOllama:
    def __init__(self, models: dict[str, list[str]] | None = None, loaded=(), reachable=True):
        self.models = dict(DEFAULT_MODELS if models is None else models)
        self.loaded = list(loaded)
        self.reachable = reachable
        self.calls: list[tuple[str, str, dict]] = []

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
            name = body["model"]
            if name not in self.models:
                return self._not_found(name)
            return httpx2.Response(200, json={"capabilities": self.models[name]})

        if route == ("POST", "/api/generate"):
            name = body["model"]
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
        return httpx2.Response(404, json={"error": f"model '{name}' not found"})
