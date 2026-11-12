from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from app.api.deps import get_db, get_searcher
from app.search import Searcher
from app.store import documents
from app.store.db import Database

router = APIRouter(tags=["search"])

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
DocumentId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]


class SearchRequest(BaseModel):
    question: Question
    document_ids: list[DocumentId] | None = Field(
        default=None, description="Only search these documents. Leave out to search all of them."
    )
    limit: int = Field(default=8, ge=1, le=30)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_no: int
    kind: str  # text, table or figure
    section: str
    text: str
    score: float  # fused score; higher is better
    found_by: list[str]  # "meaning", "keyword" or both
    meaning_rank: int | None
    keyword_rank: int | None
    distance: float | None  # cosine distance from the question (meaning search only)
    page_image_url: str
    figure_image_url: str | None


class SearchResponse(BaseModel):
    question: str
    mode: str  # "hybrid", or "keyword_only" when meaning search is unavailable
    notice: str | None
    keywords: list[str]  # the words keyword search used
    hits: list[SearchHit]


@router.post("/search")
async def search(
    request: SearchRequest,
    searcher: Searcher = Depends(get_searcher),
    db: Database = Depends(get_db),
) -> SearchResponse:
    """Find the chunks that best answer a question, using meaning and keyword search together."""
    if request.document_ids:
        unknown = [doc_id for doc_id in request.document_ids if documents.get(db, doc_id) is None]
        if unknown:
            raise HTTPException(404, f"Unknown document id(s): {', '.join(unknown)}")

    result = await searcher.search(request.question, request.document_ids, request.limit)
    hits = []
    for hit in result.hits:
        chunk = hit.chunk
        doc_id = chunk["document_id"]
        hits.append(
            SearchHit(
                chunk_id=chunk["id"],
                document_id=doc_id,
                filename=chunk["filename"],
                page_no=chunk["page_no"],
                kind=chunk["kind"],
                section=chunk["section"],
                text=chunk["text"],
                score=round(hit.score, 6),
                found_by=hit.found_by,
                meaning_rank=hit.meaning_rank,
                keyword_rank=hit.keyword_rank,
                distance=round(hit.distance, 4) if hit.distance is not None else None,
                page_image_url=f"/api/documents/{doc_id}/pages/{chunk['page_no']}/image",
                figure_image_url=(
                    f"/api/documents/{doc_id}/figures/{chunk['figure_id']}/image" if chunk["figure_id"] else None
                ),
            )
        )
    return SearchResponse(
        question=request.question, mode=result.mode, notice=result.notice, keywords=result.keywords, hits=hits
    )
