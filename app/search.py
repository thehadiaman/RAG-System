"""Hybrid search: meaning search and keyword search, merged into one ranking.

- Meaning search embeds the question and finds the closest chunk vectors in
  ChromaDB. It finds answers worded differently from the question.
- Keyword search runs the question's words against SQLite's FTS5 index. It
  finds exact terms and numbers ("12.5%", "4Q'25", "EBITDA") that meaning
  search can miss.
- Reciprocal Rank Fusion merges the two lists: each chunk scores
  1 / (60 + rank) in every list it appears in. A chunk found by both rises to
  the top; a chunk found by only one still shows up.

If the embedding model is unavailable, search carries on with keywords only.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from app.llm.embedder import Embedder, EmbedderError
from app.store.db import Database
from app.store.vectors import VectorStore

log = logging.getLogger(__name__)

RRF_K = 60  # the usual constant; it stops the very top ranks from dominating
CANDIDATES = 30  # how many results each method contributes before merging
MAX_KEYWORDS = 24

# Words that carry no meaning for finding an answer.
STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be been before being below between both but by
    can could did do does doing down during each few for from further had has have having he her here hers him
    his how i if in into is it its itself just me more most my no nor not now of off on once only or other our
    ours out over own same she should so some such than that the their theirs them then there these they this
    those through to too under until up very was we were what when where which while who whom why will with
    would you your yours tell show give find please much many
    """.split()
)
WORD = re.compile(r"[^\s\"'()\[\]{}<>,;:?!*^+~=|\\/]+(?:'[^\s\"()\[\]{}<>,;:?!*^+~=|\\/]+)?")


@dataclass
class Hit:
    chunk: dict  # the chunk row, plus "filename"
    score: float = 0.0
    meaning_rank: int | None = None
    keyword_rank: int | None = None
    distance: float | None = None  # cosine distance from meaning search

    @property
    def found_by(self) -> list[str]:
        return [name for name, rank in (("meaning", self.meaning_rank), ("keyword", self.keyword_rank)) if rank]


@dataclass
class SearchResult:
    hits: list[Hit]
    mode: str  # "hybrid", or "keyword_only" when the embedder is unavailable
    notice: str | None = None
    keywords: list[str] = field(default_factory=list)


# ---- Keyword query ----


def keywords(question: str) -> list[str]:
    """The question's meaningful words, in order, without repeats."""
    words = []
    for match in WORD.finditer(question):
        word = match.group(0).strip(".-'’")
        if not any(ch.isalnum() for ch in word):
            continue
        if word.lower() in STOPWORDS:
            continue
        if word.lower() not in (w.lower() for w in words):
            words.append(word)
    return words[:MAX_KEYWORDS]


def fts_query(words: list[str]) -> str:
    """An FTS5 query matching any of the words. Each word is quoted, so characters
    like "." or "'" inside it ("12.5", "4Q'25") can't break the query syntax."""
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in words)


# ---- Merging ----


def fuse(meaning: list[str], keyword: list[str], k: int = RRF_K) -> list[tuple[str, float, int | None, int | None]]:
    """Reciprocal Rank Fusion of two ranked id lists.

    Returns (id, score, meaning_rank, keyword_rank), best first. Ranks start at 1.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, list[int | None]] = {}
    for ids, slot in ((meaning, 0), (keyword, 1)):
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(chunk_id, [None, None])[slot] = rank
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], _best_rank(ranks[chunk_id])))
    return [(chunk_id, scores[chunk_id], *ranks[chunk_id]) for chunk_id in ordered]


def _best_rank(ranks: list[int | None]) -> int:
    return min(rank for rank in ranks if rank is not None)


# ---- Searcher ----


class Searcher:
    def __init__(self, db: Database, vectors: VectorStore, embedder: Embedder, candidates: int = CANDIDATES) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.candidates = candidates

    async def search(self, question: str, document_ids: list[str] | None = None, limit: int = 8) -> SearchResult:
        words = keywords(question)
        keyword_ids = self.keyword_search(words, document_ids)

        mode, notice = "hybrid", None
        distances: dict[str, float] = {}
        try:
            distances = await self.meaning_search(question, document_ids)
        except EmbedderError as exc:
            log.warning("Meaning search unavailable, using keywords only: %s", exc)
            mode = "keyword_only"
            notice = f"Searched by keywords only, because meaning search is unavailable: {exc}"

        fused = fuse(list(distances), keyword_ids)
        rows = self._chunks([chunk_id for chunk_id, *_ in fused])
        hits = []
        for chunk_id, score, meaning_rank, keyword_rank in fused:
            row = rows.get(chunk_id)
            if row is None:  # a vector left over from a chunk that no longer exists
                continue
            hits.append(Hit(row, score, meaning_rank, keyword_rank, distances.get(chunk_id)))
            if len(hits) == limit:
                break
        return SearchResult(hits, mode, notice, words)

    async def meaning_search(self, question: str, document_ids: list[str] | None) -> dict[str, float]:
        """Chunk id -> cosine distance, closest first."""
        embedding = await self.embedder.embed_query(question)
        found = await asyncio.to_thread(self.vectors.query, embedding, self.candidates, document_ids)
        return {hit.chunk_id: hit.distance for hit in found}

    def keyword_search(self, words: list[str], document_ids: list[str] | None) -> list[str]:
        """Chunk ids, best keyword match first (FTS5's BM25 ranking)."""
        if not words:
            return []
        query = """SELECT c.id FROM chunks_fts
                   JOIN chunks c ON c.seq = chunks_fts.rowid
                   WHERE chunks_fts MATCH ?"""
        params: list = [fts_query(words)]
        if document_ids:
            query += f" AND c.document_id IN ({','.join('?' * len(document_ids))})"
            params += document_ids
        # Matches in the chunk text count twice as much as matches in its section name.
        query += " ORDER BY bm25(chunks_fts, 1.0, 0.5) LIMIT ?"
        params.append(self.candidates)
        with self.db.session() as conn:
            return [row["id"] for row in conn.execute(query, params).fetchall()]

    def _chunks(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        with self.db.session() as conn:
            rows = conn.execute(
                f"""SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE c.id IN ({','.join('?' * len(ids))})""",
                ids,
            ).fetchall()
        return {row["id"]: dict(row) for row in rows}
