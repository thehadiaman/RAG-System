"""Turns figures and scanned pages into text with a vision model.

Two backends, chosen per upload:
- "local": Qwen2.5-VL in Ollama. One image at a time; the GPU is cleared of
  other models first (the embedder on CPU stays).
- "aws": Amazon Nova Lite on Bedrock. Several images in parallel.
"none" skips this step, so only the PDF's own text is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from app.llm.bedrock import BedrockClient, BedrockError
from app.llm.messages import ChatMessage, ImageInput
from app.llm.ollama import OllamaClient, OllamaError

Extractor = Literal["local", "aws", "none"]
EXTRACTORS: tuple[str, ...] = ("local", "aws", "none")

FIGURE_KINDS = ("chart", "table", "diagram", "photo", "decorative")
KIND_ALIASES = {
    "graph": "chart",
    "plot": "chart",
    "infographic": "diagram",
    "flowchart": "diagram",
    "map": "diagram",
    "image": "photo",
    "picture": "photo",
    "logo": "decorative",
    "icon": "decorative",
}

SYSTEM_PROMPT = (
    "You read figures and pages from business and financial documents and write them out as plain text "
    "for a search index. Be exact. Never invent numbers, labels or years that you cannot see."
)

FIGURE_PROMPT = """This image is a figure cropped from page {page_no} of a PDF.

Text the PDF itself has near and inside this figure. It is exact: when it matches what you see, use it for numbers, labels and years.
Page heading: {heading}
Caption: {caption}
Text inside the figure: {text_inside}

Reply in exactly this format:

Type: <one of: chart, table, diagram, photo, decorative>
Title: <the figure's title, or a short title of your own>
Description: <what the figure shows, in 1-3 sentences>
Data:
<For charts and tables: markdown tables holding EVERY number printed in the figure, with units and periods. Include values above bars, percentages inside bars, growth labels such as "+24% y/y", and rows of numbers under a chart. If the figure has several charts or panels, write one table per panel with the panel's title on the line above it. Otherwise write: none>

Only report what is printed in the figure. Do not add conclusions, comparisons or calculations of your own.
Use "decorative" for logos, icons, backgrounds and stock photos that carry no information, and keep the rest short for those."""

SCANNED_PAGE_PROMPT = """This image is page {page_no} of a scanned PDF.

Transcribe all of its text in reading order as markdown: headings with #, lists with -, and tables as markdown tables.
If the page has charts or pictures, describe each one in a short paragraph starting with "[Figure]" and list the values you can read.
Write only the transcription, nothing else."""

TYPE_LINE = re.compile(r"^\W*type\W*:\W*([a-z]+)", re.I | re.M)
CODE_FENCE = re.compile(r"^```[a-z]*\n(.*?)\n```$", re.S)


class VisionError(Exception):
    pass


def figure_prompt(page_no: int, heading: str, caption: str, text_inside: str) -> str:
    return FIGURE_PROMPT.format(
        page_no=page_no,
        heading=heading or "(none)",
        caption=caption or "(none)",
        text_inside=text_inside or "(none)",
    )


def scanned_page_prompt(page_no: int) -> str:
    return SCANNED_PAGE_PROMPT.format(page_no=page_no)


def clean_reply(text: str) -> str:
    text = text.strip()
    match = CODE_FENCE.match(text)
    return match.group(1).strip() if match else text


def figure_kind(description: str) -> str:
    """The figure type from the model's "Type:" line, or "other" if it's missing or unknown."""
    match = TYPE_LINE.search(description)
    if not match:
        return "other"
    word = match.group(1).lower()
    word = KIND_ALIASES.get(word, word)
    return word if word in FIGURE_KINDS else "other"


def page_heading(markdown: str) -> str:
    """The first heading on a page, else its first short line."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("#"):
            return line.lstrip("#").strip()[:200]
    return lines[0][:200] if lines and len(lines[0]) <= 200 else ""


# ---- Backends ----


class Describer(Protocol):
    label: str  # stored with each description, e.g. "qwen2.5vl:latest"
    concurrency: int

    async def prepare(self) -> None: ...

    async def describe(self, image: ImageInput, prompt: str) -> str: ...


@dataclass
class LocalDescriber:
    ollama: OllamaClient
    model: str
    num_ctx: int
    concurrency: int = 1

    @property
    def label(self) -> str:
        return self.model

    @property
    def _options(self) -> dict:
        return {"num_ctx": self.num_ctx, "temperature": 0}

    async def prepare(self) -> None:
        try:
            async with self.ollama.gpu:
                await self.ollama.use_only(self.model, options=self._options)
        except OllamaError as exc:
            raise VisionError(f"Couldn't load {self.model}: {exc}") from exc

    async def describe(self, image: ImageInput, prompt: str) -> str:
        # The GPU is held for one figure at a time, so a chat question can get in between figures.
        try:
            async with self.ollama.gpu:
                await self.ollama.use_only(self.model, options=self._options)
                reply = await self.ollama.chat(
                    self.model, [ChatMessage("user", prompt, (image,))], system=SYSTEM_PROMPT, options=self._options
                )
        except OllamaError as exc:
            raise VisionError(str(exc)) from exc
        return clean_reply(reply)


@dataclass
class AwsDescriber:
    bedrock: BedrockClient
    concurrency: int = 4
    max_tokens: int = 2000

    @property
    def label(self) -> str:
        return self.bedrock.nova_model_id

    async def prepare(self) -> None:
        pass

    async def describe(self, image: ImageInput, prompt: str) -> str:
        try:
            reply = await self.bedrock.nova(
                [ChatMessage("user", prompt, (image,))], system=SYSTEM_PROMPT, max_tokens=self.max_tokens, temperature=0
            )
        except BedrockError as exc:
            raise VisionError(str(exc)) from exc
        return clean_reply(reply)


# ---- Choosing a backend ----


class VisionService:
    def __init__(
        self,
        ollama: OllamaClient,
        bedrock: BedrockClient,
        local_model: str,
        num_ctx: int = 8192,
        max_megapixels: float = 1.0,
        aws_concurrency: int = 4,
    ) -> None:
        self.ollama = ollama
        self.bedrock = bedrock
        self.local_model = local_model
        self.num_ctx = num_ctx
        self.max_megapixels = max_megapixels
        self.aws_concurrency = aws_concurrency

    def label(self, extractor: str) -> str:
        return {
            "local": f"Qwen2.5-VL on this computer ({self.local_model})",
            "aws": "Amazon Nova Lite (AWS)",
            "none": "Text only (charts and scanned pages are not read)",
        }[extractor]

    async def problem(self, extractor: str) -> str | None:
        """None if this extractor can be used right now, otherwise why not."""
        if extractor == "none":
            return None
        if extractor == "aws":
            return self.bedrock.nova_problem()
        if extractor == "local":
            try:
                installed = {m.name: m for m in await self.ollama.list_models()}
            except OllamaError as exc:
                return str(exc)
            model = installed.get(self.local_model) or installed.get(f"{self.local_model}:latest")
            if model is None:
                return f"{self.local_model} isn't installed. Run: ollama pull {self.local_model}"
            if not model.has_vision:
                return f"{self.local_model} can't read images. Set VISION_LOCAL_MODEL to a vision model."
            return None
        return f"Unknown extractor {extractor!r}. Use one of: {', '.join(EXTRACTORS)}."

    def describer(self, extractor: str) -> Describer:
        if extractor == "local":
            return LocalDescriber(self.ollama, self.local_model, self.num_ctx)
        if extractor == "aws":
            return AwsDescriber(self.bedrock, concurrency=self.aws_concurrency)
        raise VisionError(f"No vision model for extractor {extractor!r}")
