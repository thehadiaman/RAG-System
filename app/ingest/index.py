"""The indexing step: cut a document into chunks, embed them, and store them for search."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.ingest.chunker import chunk_document, embedding_text
from app.llm.embedder import Embedder
from app.store import chunks as chunk_store
from app.store import documents
from app.store.db import Database
from app.store.documents import DocumentGone
from app.store.vectors import VectorStore


async def index_document(
    db: Database,
    vectors: VectorStore,
    embedder: Embedder,
    doc_id: str,
    on_progress: Callable[[int, int], None],
) -> int:
    """Replaces any earlier chunks of the document. Returns the number of chunks."""
    doc = documents.get(db, doc_id)
    if doc is None:
        raise DocumentGone(doc_id)
    chunks = chunk_document(documents.pages_text(db, doc_id), documents.figures(db, doc_id))

    # Embed first: if the embedding model is unavailable, the old index stays untouched.
    embeddings: list[list[float]] = []
    step = max(1, embedder.batch_size)
    for start in range(0, len(chunks), step):
        batch = chunks[start : start + step]
        embeddings += await embedder.embed_documents([embedding_text(doc["filename"], c) for c in batch])
        on_progress(len(embeddings), len(chunks))

    ids = chunk_store.replace(db, doc_id, chunks)
    await asyncio.to_thread(vectors.delete_document, doc_id)
    await asyncio.to_thread(
        vectors.upsert,
        ids,
        embeddings,
        [{"document_id": doc_id, "page_no": c.page_no, "kind": c.kind} for c in chunks],
    )
    return len(chunks)
