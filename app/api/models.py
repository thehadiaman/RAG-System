import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_ollama
from app.llm.ollama import OllamaClient, OllamaError, full_name

log = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


class ChatModel(BaseModel):
    id: str  # what the UI sends back when asking a question, e.g. "ollama:llama3.2:3b"
    provider: str
    name: str
    label: str
    vision: bool  # can read images
    loaded: bool  # currently in memory
    parameter_size: str = ""
    quantization: str = ""
    size_gb: float | None = None


class ProviderStatus(BaseModel):
    reachable: bool
    version: str | None = None
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[ChatModel]
    providers: dict[str, ProviderStatus]


@router.get("/models")
async def list_models(ollama: OllamaClient = Depends(get_ollama)) -> ModelsResponse:
    """Models that can be picked for chat. Embedding-only models are left out."""
    try:
        version = await ollama.version()
        installed = await ollama.list_models()
        loaded = {full_name(name) for name in await ollama.loaded_models()}
    except OllamaError as exc:
        log.warning("Ollama unavailable: %s", exc)
        return ModelsResponse(
            models=[],
            providers={"ollama": ProviderStatus(reachable=False, error=str(exc))},
        )

    models = [
        ChatModel(
            id=f"ollama:{m.name}",
            provider="ollama",
            name=m.name,
            label=m.name,
            vision=m.has_vision,
            loaded=full_name(m.name) in loaded,
            parameter_size=m.parameter_size,
            quantization=m.quantization,
            size_gb=round(m.size_bytes / 1e9, 1),
        )
        for m in installed
        if m.can_chat
    ]
    return ModelsResponse(
        models=models,
        providers={"ollama": ProviderStatus(reachable=True, version=version)},
    )
