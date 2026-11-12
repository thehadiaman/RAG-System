"""Saved chats and their messages."""

from __future__ import annotations

import json
import re

from app.answer import Turn
from app.store.db import NOW, Database
from app.store.documents import new_id

NEW_CHAT_TITLE = "New chat"
TITLE_CHARS = 60

COMPLETE = "complete"
STOPPED = "stopped"  # the user stopped the answer, or the connection closed
ERROR = "error"


class ChatGone(Exception):
    """The chat was deleted."""


def title_from(question: str) -> str:
    """A chat title made from its first question."""
    text = re.sub(r"\s+", " ", question).strip()
    if len(text) <= TITLE_CHARS:
        return text
    return text[:TITLE_CHARS].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def _chat(row) -> dict:
    chat = dict(row)
    chat["document_ids"] = json.loads(chat["document_ids"]) if chat["document_ids"] else None
    return chat


def _message(row) -> dict:
    message = dict(row)
    for key in ("sources", "cited", "number_check"):
        message[key] = json.loads(message[key]) if message[key] else None
    return message


# ---- Chats ----


def create(db: Database, title: str | None, model_id: str | None, document_ids: list[str] | None) -> dict:
    chat_id = new_id()
    with db.session() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, model_id, document_ids) VALUES (?, ?, ?, ?)",
            (chat_id, title or NEW_CHAT_TITLE, model_id, json.dumps(document_ids) if document_ids else None),
        )
    return get(db, chat_id)


def get(db: Database, chat_id: str) -> dict | None:
    with db.session() as conn:
        row = conn.execute(
            """SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS message_count
               FROM chats c WHERE c.id = ?""",
            (chat_id,),
        ).fetchone()
    return _chat(row) if row else None


def list_all(db: Database) -> list[dict]:
    with db.session() as conn:
        rows = conn.execute(
            """SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS message_count
               FROM chats c ORDER BY c.updated_at DESC, c.created_at DESC"""
        ).fetchall()
    return [_chat(row) for row in rows]


def update(db: Database, chat_id: str, **fields) -> None:
    """Change any of: title, model_id, document_ids (None means all documents)."""
    if "document_ids" in fields:
        fields["document_ids"] = json.dumps(fields["document_ids"]) if fields["document_ids"] else None
    if not fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with db.session() as conn:
        cursor = conn.execute(
            f"UPDATE chats SET {assignments}, updated_at = {NOW} WHERE id = ?", (*fields.values(), chat_id)
        )
        if cursor.rowcount == 0:
            raise ChatGone(chat_id)


def delete(db: Database, chat_id: str) -> bool:
    with db.session() as conn:
        return conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount > 0


# ---- Messages ----


def add_message(
    db: Database,
    chat_id: str,
    role: str,
    content: str,
    *,
    model_id: str | None = None,
    sources: list[dict] | None = None,
    cited: list[int] | None = None,
    search_query: str | None = None,
    seconds: float | None = None,
    status: str = COMPLETE,
    error: str | None = None,
    number_check: dict | None = None,
) -> int:
    with db.session() as conn:
        if not conn.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone():
            raise ChatGone(chat_id)
        cursor = conn.execute(
            """INSERT INTO messages
                   (chat_id, role, content, model_id, sources, cited, search_query, seconds, status, error,
                    number_check)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chat_id,
                role,
                content,
                model_id,
                json.dumps(sources, ensure_ascii=False) if sources is not None else None,
                json.dumps(cited) if cited is not None else None,
                search_query,
                seconds,
                status,
                error,
                json.dumps(number_check) if number_check is not None else None,
            ),
        )
        conn.execute(f"UPDATE chats SET updated_at = {NOW} WHERE id = ?", (chat_id,))
        return cursor.lastrowid


def messages(db: Database, chat_id: str) -> list[dict]:
    with db.session() as conn:
        rows = conn.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)).fetchall()
    return [_message(row) for row in rows]


def recent_turns(db: Database, chat_id: str, limit: int) -> list[Turn]:
    """The last completed question-and-answer pairs, oldest first.

    Each question is taken in its complete form (the rewritten one that was
    searched), so a chain of follow-ups keeps its context.
    """
    turns: list[Turn] = []
    question: str | None = None
    for message in messages(db, chat_id):
        if message["role"] == "user":
            question = message["content"]
        elif question is not None:
            if message["status"] == COMPLETE and message["content"]:
                turns.append(Turn(message["search_query"] or question, message["content"]))
            question = None
    return turns[-limit:]
