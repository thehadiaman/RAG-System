from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_vision
from app.config import Settings, get_settings
from app.ingest.vision import EXTRACTORS, VisionService

router = APIRouter(tags=["documents"])


class ExtractorOut(BaseModel):
    id: str
    label: str
    available: bool
    problem: str | None


class ExtractorsResponse(BaseModel):
    default: str
    extractors: list[ExtractorOut]


@router.get("/extractors")
async def list_extractors(
    vision: VisionService = Depends(get_vision),
    settings: Settings = Depends(get_settings),
) -> ExtractorsResponse:
    """Who can read charts and scanned pages right now (for the upload screen)."""
    items = []
    for extractor in EXTRACTORS:
        problem = await vision.problem(extractor)
        items.append(
            ExtractorOut(id=extractor, label=vision.label(extractor), available=problem is None, problem=problem)
        )
    return ExtractorsResponse(default=settings.default_extractor, extractors=items)
