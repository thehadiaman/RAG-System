"""A stand-in for the search model, so most tests don't load the real one."""

import math
import re
import zlib

from app.llm.embedder import DOCUMENT_PREFIX, QUERY_PREFIX, EmbedderError


def fake_vector(text: str, dims: int = 16) -> list[float]:
    """Bag-of-words vector: texts sharing words point the same way. Task prefixes are ignored."""
    for prefix in (DOCUMENT_PREFIX, QUERY_PREFIX):
        text = text.removeprefix(prefix)
    vector = [0.0] * dims
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[zlib.crc32(word.encode()) % dims] += 1.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class FakeEmbedder:
    def __init__(self, model: str = "fake-search-model", batch_size: int = 4, problem: str | None = None):
        self.model = model
        self.label = f"{model} (test)"
        self.batch_size = batch_size
        self.broken = problem  # when set, every call fails with this
        self.inputs: list[list[str]] = []  # the texts of each call, with their prefixes

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([DOCUMENT_PREFIX + text for text in texts])

    async def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0]

    async def problem(self) -> str | None:
        return self.broken

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        if self.broken:
            raise EmbedderError(self.broken)
        self.inputs.append(inputs)
        return [fake_vector(text) for text in inputs]
