import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import __version__
from app.api.deps import get_db, get_embedder, get_vectors
from app.llm.embedder import Embedder
from app.store.db import Database
from app.store.vectors import VectorStore

log = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


class Component(BaseModel):
    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    components: dict[str, Component]


async def _probe(name: str, probe) -> Component:
    try:
        return Component(ok=True, detail=await probe())
    except Exception as exc:  # noqa: BLE001 - a health check reports any failure
        log.warning("%s check failed: %s", name, exc)
        return Component(ok=False, detail=str(exc))


@router.get("/health")
async def health(
    db: Database = Depends(get_db),
    vectors: VectorStore = Depends(get_vectors),
    embedder: Embedder = Depends(get_embedder),
) -> HealthResponse:
    """Always answers 200 while the server runs; `status` says whether every part is ready."""

    async def database() -> str:
        return f"schema version {await asyncio.to_thread(db.ping)}"

    async def vector_store() -> str:
        return f"{await asyncio.to_thread(vectors.count)} chunks"

    problem = await embedder.problem()
    components = {
        "database": await _probe("database", database),
        "vectors": await _probe("vectors", vector_store),
        "embedder": Component(ok=problem is None, detail=problem or embedder.model),
    }
    status = "ok" if all(c.ok for c in components.values()) else "degraded"
    return HealthResponse(status=status, version=__version__, components=components)
