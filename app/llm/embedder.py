"""Turns text into vectors for meaning search, inside the app: no Ollama needed.

The model is nomic-embed-text v1.5 (Apache-2.0) in its quantised ONNX form, run on the CPU by
onnxruntime. Its files are built into the Docker image (scripts/fetch_search_model.py), so search
works offline.

Texts are embedded one at a time. On a CPU that is the fastest way, because nothing is padded. It
keeps memory low: a batch is padded to its longest text, and memory grows with the square of the
length. And a text's vector doesn't depend on which other texts happen to share its batch.
On the test decks: about 20 chunks a second with 4 threads, under 1 GB at the peak.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_ID = "nomic-embed-text-v1.5-q"  # stored with each document's index; also names the vector collection
MODEL_LABEL = "nomic-embed-text v1.5 (built in)"
MODEL_FILE = "onnx/model_quantized.onnx"
TOKENIZER_FILE = "tokenizer.json"
MAX_TOKENS = 2048  # longer texts are cut; the same window Ollama gave this model

# nomic-embed-text needs a task prefix on every input to give good results.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbedderError(Exception):
    pass


class Embedder(Protocol):
    model: str  # a stable id of the model that makes the vectors
    label: str  # shown to people
    batch_size: int  # how many texts are embedded between progress updates

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def problem(self) -> str | None: ...


def default_threads() -> int:
    # More threads than this gained little in measurements and leave less CPU for everything else.
    return min(4, os.cpu_count() or 1)


class OnnxEmbedder:
    def __init__(self, model_dir: Path, threads: int = 0, batch_size: int = 32) -> None:
        self.model_dir = model_dir
        self.threads = threads or default_threads()
        self.batch_size = batch_size
        self.model = MODEL_ID
        self.label = MODEL_LABEL
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._inputs: set[str] = set()
        self._load_lock = threading.Lock()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed([DOCUMENT_PREFIX + text for text in texts])

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([QUERY_PREFIX + text]))[0]

    async def problem(self) -> str | None:
        """None if the model is ready, otherwise why it isn't. Loads it on the first call."""
        try:
            await asyncio.to_thread(self._load)
        except EmbedderError as exc:
            return str(exc)
        return None

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        # In a thread: onnxruntime releases the GIL, so the server keeps answering meanwhile.
        try:
            return await asyncio.to_thread(self._embed_all, inputs)
        except EmbedderError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as a search problem, not a crash
            raise EmbedderError(f"The search model failed: {exc}") from exc

    def _embed_all(self, inputs: list[str]) -> list[list[float]]:
        self._load()
        return [self._vector(text) for text in inputs]

    def _vector(self, text: str) -> list[float]:
        ids = np.array([self._tokenizer.encode(text).ids], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        tokens = self._session.run(None, feed)[0][0]
        vector = tokens.mean(axis=0)  # mean of the token vectors, as the model was trained
        return (vector / np.linalg.norm(vector)).tolist()

    def _load(self) -> None:
        with self._load_lock:
            if self._session is not None:
                return
            model, tokenizer_file = self.model_dir / MODEL_FILE, self.model_dir / TOKENIZER_FILE
            if not (model.is_file() and tokenizer_file.is_file()):
                raise EmbedderError(
                    f"The search model's files are missing from {self.model_dir}. "
                    "Rebuild the app: docker compose up -d --build"
                )
            try:
                tokenizer = Tokenizer.from_file(str(tokenizer_file))
                tokenizer.enable_truncation(max_length=MAX_TOKENS)
                tokenizer.no_padding()
                options = ort.SessionOptions()
                options.intra_op_num_threads = self.threads
                options.inter_op_num_threads = 1
                # Give memory back after a long text, instead of keeping it for the next one.
                options.enable_cpu_mem_arena = False
                options.log_severity_level = 3  # errors only
                session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
            except Exception as exc:  # noqa: BLE001 - a broken file is reported, not a crash
                raise EmbedderError(f"Couldn't load the search model from {self.model_dir}: {exc}") from exc
            self._inputs = {i.name for i in session.get_inputs()}
            self._tokenizer = tokenizer
            self._session = session
