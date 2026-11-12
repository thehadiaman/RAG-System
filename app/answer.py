"""Answers a question from the user's documents, with numbered citations.

1. Search picks the best chunks; they become numbered sources, trimmed to fit.
2. If the chosen model can read images, the images of the top chart sources go
   along too, so the model can check numbers itself.
3. The model is told to answer only from the sources and cite them like [2].
4. The answer streams back as events: sources first, then text, then which
   sources were cited.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from app.ingest.pdf import image_for_vision
from app.llm.bedrock import QWEN_LABEL, BedrockClient, BedrockError
from app.llm.messages import ChatMessage, ImageInput
from app.llm.ollama import OllamaClient, OllamaError, full_name
from app.search import Searcher

log = logging.getLogger(__name__)

AWS_QWEN_ID = "aws:qwen2.5-vl"
MAX_CHARS_PER_SOURCE = 3000
IMAGE_MEGAPIXELS = 1.0
NOT_FOUND = "I couldn't find this in your documents."
NOTHING_INDEXED = (
    "I couldn't find anything in your documents to answer this. Upload a PDF, or wait until it is ready."
)

SYSTEM_PROMPT = f"""You answer questions about the user's documents, using only the numbered sources you are given.

Rules:
- Use only facts stated in the sources. If they don't contain the answer, reply exactly: "{NOT_FOUND}"
- After each fact, cite its source number in square brackets, like [2]. For several sources write [1][3].
- Copy numbers, units, currencies and periods exactly as the sources write them. Don't round or convert them.
- If sources disagree, say so and cite both.
- Start with the direct answer, then add short supporting details. Keep it brief.
- Answer in the same language as the question."""

CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


class AnswerError(Exception):
    """The question can't be answered with this model (shown to the user before anything streams)."""


@dataclass(frozen=True)
class ModelChoice:
    id: str
    provider: str  # "ollama" or "aws"
    name: str
    vision: bool


@dataclass
class Source:
    number: int
    hit: dict  # the chunk row, plus "filename"
    text: str  # possibly shortened
    image: ImageInput | None = None

    @property
    def label(self) -> str:
        chunk = self.hit
        where = f"{chunk['filename']}, page {chunk['page_no']}, {chunk['kind']}"
        return f"{where}, section: {chunk['section']}" if chunk["section"] else where


@dataclass
class Prepared:
    question: str
    model: ModelChoice
    sources: list[Source]
    search_mode: str
    notice: str | None = None


@dataclass
class Event:
    type: str  # "sources", "status", "token", "done" or "error"
    data: dict = field(default_factory=dict)


# ---- Plain functions (easy to test) ----


def select_sources(hits: list[dict], max_sources: int, max_chars: int) -> list[Source]:
    """The best hits, in order, shortened so that together they stay within `max_chars`."""
    sources: list[Source] = []
    used = 0
    for hit in hits[:max_sources]:
        room = max_chars - used
        if sources and room < 200:
            break
        text = hit["text"]
        limit = min(MAX_CHARS_PER_SOURCE, max(room, 200))
        if len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0] + " …"
        sources.append(Source(len(sources) + 1, hit, text))
        used += len(text)
    return sources


def build_prompt(question: str, sources: list[Source]) -> str:
    parts = ["Sources:"]
    for source in sources:
        parts.append(f"[{source.number}] {source.label}\n{source.text}")
    with_images = [s.number for s in sources if s.image]
    if with_images:
        listed = ", ".join(f"image {i} is source [{n}]" for i, n in enumerate(with_images, start=1))
        parts.append(f"The images attached are the figures of some sources: {listed}.")
    parts.append(f"Question: {question}")
    parts.append("Answer from the sources above, citing them like [1].")
    return "\n\n".join(parts)


def cited_numbers(answer: str, source_count: int) -> list[int]:
    """Source numbers the answer cites, in order of first use. Numbers that don't exist are ignored."""
    cited: list[int] = []
    for match in CITATION.finditer(answer):
        for number in re.split(r"\s*[,;]\s*", match.group(1)):
            n = int(number)
            if 1 <= n <= source_count and n not in cited:
                cited.append(n)
    return cited


# ---- Answering ----


class Answerer:
    def __init__(
        self,
        searcher: Searcher,
        ollama: OllamaClient,
        bedrock: BedrockClient,
        data_dir: Path,
        num_ctx: int = 8192,
        max_sources: int = 8,
        max_source_chars: int = 12000,
        max_images: int = 2,
        max_answer_tokens: int = 1024,
    ) -> None:
        self.searcher = searcher
        self.ollama = ollama
        self.bedrock = bedrock
        self.data_dir = data_dir
        self.num_ctx = num_ctx
        self.max_sources = max_sources
        self.max_source_chars = max_source_chars
        self.max_images = max_images
        self.max_answer_tokens = max_answer_tokens

    async def resolve_model(self, model_id: str) -> ModelChoice:
        if model_id == AWS_QWEN_ID:
            problem = self.bedrock.qwen_problem()
            if problem:
                raise AnswerError(f"{QWEN_LABEL} can't be used: {problem}")
            return ModelChoice(model_id, "aws", QWEN_LABEL, vision=True)

        if model_id.startswith("ollama:"):
            name = model_id.removeprefix("ollama:")
            try:
                installed = {full_name(m.name): m for m in await self.ollama.list_models()}
            except OllamaError as exc:
                raise AnswerError(str(exc)) from exc
            model = installed.get(full_name(name))
            if model is None:
                raise AnswerError(f"{name} isn't installed in Ollama. Run: ollama pull {name}")
            if not model.can_chat:
                raise AnswerError(f"{name} can't chat (it is an embedding model). Pick another model.")
            return ModelChoice(model_id, "ollama", model.name, vision=model.has_vision)

        raise AnswerError(f"Unknown model {model_id!r}. Pick one from /api/models.")

    async def prepare(self, question: str, model_id: str, document_ids: list[str] | None = None) -> Prepared:
        model = await self.resolve_model(model_id)
        result = await self.searcher.search(question, document_ids, limit=self.max_sources)
        sources = select_sources([hit.chunk for hit in result.hits], self.max_sources, self.max_source_chars)
        if model.vision and self.max_images > 0:
            await self._attach_images(sources)
        return Prepared(question, model, sources, result.mode, result.notice)

    async def _attach_images(self, sources: list[Source]) -> None:
        attached = 0
        for source in sources:
            path = source.hit.get("image_path")
            if source.hit["kind"] != "figure" or not path:
                continue
            try:
                data = await asyncio.to_thread(image_for_vision, self.data_dir / path, IMAGE_MEGAPIXELS)
            except Exception as exc:  # noqa: BLE001 - a missing image just means no image
                log.warning("Couldn't load figure image %s: %s", path, exc)
                continue
            source.image = ImageInput(data, "png")
            attached += 1
            if attached == self.max_images:
                break

    async def stream(self, prepared: Prepared) -> AsyncIterator[Event]:
        """Events for one answer. Never raises: problems arrive as an "error" event."""
        started = time.perf_counter()
        yield Event("sources", {"mode": prepared.search_mode, "notice": prepared.notice, "sources": prepared.sources})

        if not prepared.sources:
            yield Event("token", {"text": NOTHING_INDEXED})
            yield Event("done", self._done(prepared, NOTHING_INDEXED, started, None))
            return

        images = tuple(s.image for s in prepared.sources if s.image)
        messages = [ChatMessage("user", build_prompt(prepared.question, prepared.sources), images)]
        answer: list[str] = []
        first_token: float | None = None
        try:
            async for event in self._generate(prepared.model, messages):
                if event.type == "token":
                    if first_token is None:
                        first_token = time.perf_counter() - started
                    answer.append(event.data["text"])
                yield event
        except (OllamaError, BedrockError) as exc:
            log.warning("Answer failed with %s: %s", prepared.model.id, exc)
            yield Event("error", {"message": str(exc), "partial_answer": "".join(answer)})
            return
        yield Event("done", self._done(prepared, "".join(answer), started, first_token))

    async def _generate(self, model: ModelChoice, messages: list[ChatMessage]) -> AsyncIterator[Event]:
        if model.provider == "aws":
            async for piece in self.bedrock.qwen_stream(
                messages, system=SYSTEM_PROMPT, max_tokens=self.max_answer_tokens, temperature=0.1
            ):
                yield Event("token", {"text": piece})
            return

        options = {"num_ctx": self.num_ctx, "temperature": 0.1, "num_predict": self.max_answer_tokens}
        if self.ollama.gpu.locked():
            yield Event("status", {"message": "Waiting for the GPU: a PDF is being read by the vision model."})
        async with self.ollama.gpu:
            if full_name(model.name) not in {full_name(n) for n in await self.ollama.loaded_models()}:
                yield Event("status", {"message": f"Loading {model.name}. This can take up to a minute."})
            await self.ollama.use_only(model.name, options=options)
            yield Event("status", {"message": "Writing the answer."})
            async for piece in self.ollama.chat_stream(model.name, messages, system=SYSTEM_PROMPT, options=options):
                yield Event("token", {"text": piece})

    def _done(self, prepared: Prepared, answer: str, started: float, first_token: float | None) -> dict:
        return {
            "answer": answer,
            "cited": cited_numbers(answer, len(prepared.sources)),
            "model": prepared.model.id,
            "seconds": round(time.perf_counter() - started, 2),
            "first_token_seconds": round(first_token, 2) if first_token is not None else None,
        }
