import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from app.answer import Answerer, AnswerError, Event, Source
from app.api.deps import get_answerer, get_db
from app.store import documents
from app.store.db import Database

router = APIRouter(tags=["chat"])

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
DocumentId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]


class ChatRequest(BaseModel):
    question: Question
    model: str = Field(description='A model id from /api/models, e.g. "ollama:qwen2.5vl:latest" or "aws:qwen2.5-vl".')
    document_ids: list[DocumentId] | None = Field(
        default=None, description="Only use these documents. Leave out to use all of them."
    )


def source_json(source: Source) -> dict:
    chunk = source.hit
    doc_id = chunk["document_id"]
    return {
        "number": source.number,
        "chunk_id": chunk["id"],
        "document_id": doc_id,
        "filename": chunk["filename"],
        "page_no": chunk["page_no"],
        "kind": chunk["kind"],
        "section": chunk["section"],
        "text": source.text,
        "image_sent_to_model": source.image is not None,
        "page_image_url": f"/api/documents/{doc_id}/pages/{chunk['page_no']}/image",
        "figure_image_url": (
            f"/api/documents/{doc_id}/figures/{chunk['figure_id']}/image" if chunk["figure_id"] else None
        ),
    }


def sse(event: Event) -> str:
    data = dict(event.data)
    if event.type == "sources":
        data["sources"] = [source_json(source) for source in data["sources"]]
    return f"event: {event.type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/chat",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Server-Sent Events"}},
)
async def chat(
    request: ChatRequest,
    answerer: Answerer = Depends(get_answerer),
    db: Database = Depends(get_db),
) -> StreamingResponse:
    """Answer a one-off question from your documents (nothing is saved), streamed as Server-Sent Events.

    Events, in order: `sources` (the numbered sources the model reads), `status`
    (optional progress notes), `token` (pieces of the answer), then `done`
    (the full answer and which sources it cited) or `error`. For saved
    conversations with follow-up questions, use /api/chats.
    """
    if request.document_ids:
        unknown = [doc_id for doc_id in request.document_ids if documents.get(db, doc_id) is None]
        if unknown:
            raise HTTPException(404, f"Unknown document id(s): {', '.join(unknown)}")
    try:
        model = await answerer.resolve_model(request.model)
    except AnswerError as exc:
        raise HTTPException(400, str(exc)) from exc

    async def events() -> AsyncIterator[str]:
        async for event in answerer.answer(request.question, model, request.document_ids):
            yield sse(event)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
