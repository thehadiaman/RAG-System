import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from app.answer import HISTORY_TURNS, Answerer, AnswerError, Event, ModelChoice, cited_numbers
from app.api.chat import Question, source_json, sse
from app.api.deps import get_answerer, get_db
from app.store import chats, documents
from app.store.chats import COMPLETE, ERROR, NEW_CHAT_TITLE, STOPPED, ChatGone
from app.store.db import Database

router = APIRouter(prefix="/chats", tags=["chats"])

ChatId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]
DocumentId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class ChatCreate(BaseModel):
    title: Title | None = None
    model: str | None = Field(default=None, description="A model id from /api/models.")
    document_ids: list[DocumentId] | None = Field(default=None, description="Leave out to use all documents.")


class ChatUpdate(BaseModel):
    """Only the fields you send are changed. Send "document_ids": null to use all documents again."""

    title: Title | None = None
    model: str | None = None
    document_ids: list[DocumentId] | None = None


class MessageRequest(BaseModel):
    question: Question
    model: str | None = Field(default=None, description="Leave out to use the chat's model. A new one is remembered.")


class ChatOut(BaseModel):
    id: str
    title: str
    model: str | None
    document_ids: list[str] | None
    message_count: int
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    model: str | None
    sources: list[dict] | None
    cited: list[int] | None
    search_query: str | None
    seconds: float | None
    status: str  # complete, stopped or error
    error: str | None
    created_at: str


class ChatDetail(ChatOut):
    messages: list[MessageOut]


def _chat_out(chat: dict) -> ChatOut:
    return ChatOut(
        **{key: chat[key] for key in ("id", "title", "document_ids", "message_count", "created_at", "updated_at")},
        model=chat["model_id"],
    )


def _message_out(message: dict) -> MessageOut:
    return MessageOut(
        **{
            key: message[key]
            for key in ("id", "role", "content", "sources", "cited", "search_query", "seconds", "status", "error")
        },
        model=message["model_id"],
        created_at=message["created_at"],
    )


def _get_or_404(db: Database, chat_id: str) -> dict:
    chat = chats.get(db, chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found.")
    return chat


async def _check_model(answerer: Answerer, model_id: str) -> ModelChoice:
    try:
        return await answerer.resolve_model(model_id)
    except AnswerError as exc:
        raise HTTPException(400, str(exc)) from exc


def _check_documents(db: Database, document_ids: list[str] | None) -> None:
    unknown = [doc_id for doc_id in document_ids or [] if documents.get(db, doc_id) is None]
    if unknown:
        raise HTTPException(404, f"Unknown document id(s): {', '.join(unknown)}")


# ---- Chats ----


@router.post("", status_code=201)
async def create_chat(
    body: ChatCreate, db: Database = Depends(get_db), answerer: Answerer = Depends(get_answerer)
) -> ChatOut:
    if body.model:
        await _check_model(answerer, body.model)
    _check_documents(db, body.document_ids)
    return _chat_out(chats.create(db, body.title, body.model, body.document_ids))


@router.get("")
async def list_chats(db: Database = Depends(get_db)) -> list[ChatOut]:
    """All chats, most recently used first."""
    return [_chat_out(chat) for chat in chats.list_all(db)]


@router.get("/{chat_id}")
async def get_chat(chat_id: ChatId, db: Database = Depends(get_db)) -> ChatDetail:
    chat = _get_or_404(db, chat_id)
    return ChatDetail(
        **_chat_out(chat).model_dump(), messages=[_message_out(m) for m in chats.messages(db, chat_id)]
    )


@router.patch("/{chat_id}")
async def update_chat(
    chat_id: ChatId,
    body: ChatUpdate,
    db: Database = Depends(get_db),
    answerer: Answerer = Depends(get_answerer),
) -> ChatOut:
    _get_or_404(db, chat_id)
    changes: dict = {}
    if "title" in body.model_fields_set and body.title:
        changes["title"] = body.title
    if "model" in body.model_fields_set and body.model:
        await _check_model(answerer, body.model)
        changes["model_id"] = body.model
    if "document_ids" in body.model_fields_set:
        _check_documents(db, body.document_ids)
        changes["document_ids"] = body.document_ids
    chats.update(db, chat_id, **changes)
    return _chat_out(chats.get(db, chat_id))


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(chat_id: ChatId, db: Database = Depends(get_db)) -> Response:
    if not chats.delete(db, chat_id):
        raise HTTPException(404, "Chat not found.")
    return Response(status_code=204)


# ---- Asking ----


async def answer_and_save(
    answerer: Answerer,
    db: Database,
    chat_id: str,
    question: str,
    model: ModelChoice,
    document_ids: list[str] | None,
    history: list,
) -> AsyncIterator[str]:
    """Stream the answer, and save it when it ends: complete, failed, or cut off part way."""
    started = time.perf_counter()
    sources: list[dict] | None = None
    search_query = question
    pieces: list[str] = []
    saved = False

    def save(status: str, content: str, cited: list[int], seconds: float, error: str | None = None) -> int | None:
        nonlocal saved
        saved = True
        try:
            return chats.add_message(
                db,
                chat_id,
                "assistant",
                content,
                model_id=model.id,
                sources=sources,
                cited=cited,
                search_query=search_query,
                seconds=round(seconds, 2),
                status=status,
                error=error,
            )
        except ChatGone:
            return None

    try:
        async for event in answerer.answer(question, model, document_ids, history):
            if event.type == "sources":
                sources = [source_json(source) for source in event.data["sources"]]
                search_query = event.data["search_query"]
            elif event.type == "token":
                pieces.append(event.data["text"])
            elif event.type == "done":
                data = event.data
                message_id = save(COMPLETE, data["answer"], data["cited"], data["seconds"])
                event = Event("done", {**data, "message_id": message_id})
            elif event.type == "error":
                partial = event.data["partial_answer"]
                message_id = save(
                    ERROR,
                    partial,
                    cited_numbers(partial, len(sources or [])),
                    time.perf_counter() - started,
                    event.data["message"],
                )
                event = Event("error", {**event.data, "message_id": message_id})
            yield sse(event)
    finally:
        if not saved:  # stopped by the user, or the connection closed
            partial = "".join(pieces)
            save(STOPPED, partial, cited_numbers(partial, len(sources or [])), time.perf_counter() - started)


@router.post(
    "/{chat_id}/messages",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Server-Sent Events"}},
)
async def ask(
    chat_id: ChatId,
    body: MessageRequest,
    db: Database = Depends(get_db),
    answerer: Answerer = Depends(get_answerer),
) -> StreamingResponse:
    """Ask a question in a chat. Same events as POST /api/chat, and both sides are saved.

    Follow-up questions are understood using the chat's recent messages; the
    `sources` event's `search_query` shows the complete question that was searched.
    """
    chat = _get_or_404(db, chat_id)
    model_id = body.model or chat["model_id"]
    if not model_id:
        raise HTTPException(400, "This chat has no model yet. Send one of the ids from /api/models as `model`.")
    model = await _check_model(answerer, model_id)

    # Documents deleted since the chat was limited to them are left out.
    document_ids = None
    if chat["document_ids"]:
        document_ids = [doc_id for doc_id in chat["document_ids"] if documents.get(db, doc_id)] or None

    history = chats.recent_turns(db, chat_id, HISTORY_TURNS)
    changes: dict = {}
    if chat["message_count"] == 0 and chat["title"] == NEW_CHAT_TITLE:
        changes["title"] = chats.title_from(body.question)
    if model_id != chat["model_id"]:
        changes["model_id"] = model_id
    chats.update(db, chat_id, **changes)
    chats.add_message(db, chat_id, "user", body.question)

    return StreamingResponse(
        answer_and_save(answerer, db, chat_id, body.question, model, document_ids, history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
