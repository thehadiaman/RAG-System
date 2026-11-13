"""Chunk vectors in ChromaDB, running inside the app with its files under data/chroma.

Only ids, vectors and a little metadata live here. The chunk text itself is in
SQLite, so there's one source of truth for content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

COLLECTION = "chunks"


def collection_for(model: str) -> str:
    """Each search model gets its own collection: vectors from two models can't be compared."""
    return f"{COLLECTION}-{model}"


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    distance: float  # cosine distance: 0 = same meaning, 2 = opposite
    metadata: dict


class VectorStore:
    def __init__(self, path: Path, collection: str = COLLECTION) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # We always pass our own vectors, so Chroma never needs its own embedding model.
        self._collection = self._client.get_or_create_collection(
            collection,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        self.collection = collection

    def remove_other_collections(self) -> list[str]:
        """Deletes vectors made by earlier search models. Returns the names of the collections removed."""
        removed = []
        for collection in self._client.list_collections():
            if collection.name != self.collection:
                self._client.delete_collection(collection.name)
                removed.append(collection.name)
        return removed

    def count(self) -> int:
        return self._collection.count()

    def upsert(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        if ids:
            self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(
        self,
        embedding: list[float],
        limit: int = 10,
        document_ids: list[str] | None = None,
    ) -> list[VectorHit]:
        """Closest chunks first. `document_ids` limits the search to those documents."""
        if self.count() == 0:
            return []
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=where,
            include=["distances", "metadatas"],
        )
        return [
            VectorHit(chunk_id=chunk_id, distance=distance, metadata=metadata or {})
            for chunk_id, distance, metadata in zip(
                result["ids"][0], result["distances"][0], result["metadatas"][0], strict=True
            )
        ]

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
