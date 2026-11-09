"""Turns text into vectors with an Ollama embedding model (nomic-embed-text by default).

The embedder runs on the CPU (num_gpu 0) and stays loaded (keep_alive -1), so it
never takes GPU memory from the chat or vision model.
"""

from __future__ import annotations

from itertools import batched

from app.llm.ollama import OllamaClient, OllamaError

CPU_ONLY = {"num_gpu": 0}
STAY_LOADED = -1

# nomic-embed-text needs a task prefix on every input to give good results.
NOMIC_DOCUMENT_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "


class EmbedderError(Exception):
    pass


class Embedder:
    def __init__(self, ollama: OllamaClient, model: str, batch_size: int = 32) -> None:
        self.ollama = ollama
        self.model = model
        self.batch_size = batch_size
        uses_prefixes = "nomic" in model
        self._document_prefix = NOMIC_DOCUMENT_PREFIX if uses_prefixes else ""
        self._query_prefix = NOMIC_QUERY_PREFIX if uses_prefixes else ""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed([self._document_prefix + text for text in texts])

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([self._query_prefix + text]))[0]

    async def problem(self) -> str | None:
        """None if the embedder is ready, otherwise why it isn't."""
        try:
            installed = await self.ollama.has_model(self.model)
        except OllamaError as exc:
            return str(exc)
        return None if installed else self._not_installed()

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in batched(inputs, self.batch_size):
            try:
                vectors.extend(
                    await self.ollama.embed(self.model, list(batch), keep_alive=STAY_LOADED, options=CPU_ONLY)
                )
            except OllamaError as exc:
                if "not found" in str(exc):
                    raise EmbedderError(self._not_installed()) from exc
                raise EmbedderError(f"Embedding failed: {exc}") from exc
        return vectors

    def _not_installed(self) -> str:
        return f"Embedding model {self.model} isn't installed. Run: ollama pull {self.model}"
