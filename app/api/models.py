import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_bedrock, get_ollama
from app.llm.bedrock import QWEN_LABEL, BedrockClient
from app.llm.ollama import OllamaClient, OllamaError, full_name

log = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

AWS_QWEN_ID = "aws:qwen2.5-vl"


class ChatModel(BaseModel):
    id: str  # what the UI sends back when asking a question, e.g. "ollama:llama3.2:3b"
    provider: str  # "ollama" or "aws"
    name: str
    label: str
    vision: bool  # can read images
    loaded: bool  # currently in memory (always False for AWS)
    parameter_size: str = ""
    quantization: str = ""
    size_gb: float | None = None


class ProviderStatus(BaseModel):
    available: bool
    version: str | None = None
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[ChatModel]
    providers: dict[str, ProviderStatus]


async def _ollama_models(ollama: OllamaClient) -> tuple[list[ChatModel], ProviderStatus]:
    try:
        version = await ollama.version()
        installed = await ollama.list_models()
        loaded = {full_name(name) for name in await ollama.loaded_models()}
    except OllamaError as exc:
        log.warning("Ollama unavailable: %s", exc)
        return [], ProviderStatus(available=False, error=str(exc))

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
    return models, ProviderStatus(available=True, version=version)


def _aws_models(bedrock: BedrockClient) -> tuple[list[ChatModel], ProviderStatus]:
    # Only checks settings; no AWS call, so the list stays fast.
    problem = bedrock.qwen_problem()
    if problem:
        return [], ProviderStatus(available=False, error=problem)
    model = ChatModel(
        id=AWS_QWEN_ID,
        provider="aws",
        name="qwen2.5-vl",
        label=QWEN_LABEL,
        vision=True,
        loaded=False,
    )
    return [model], ProviderStatus(available=True)


@router.get("/models")
async def list_models(
    ollama: OllamaClient = Depends(get_ollama),
    bedrock: BedrockClient = Depends(get_bedrock),
) -> ModelsResponse:
    """Models that can be picked for chat. Embedding-only models are left out."""
    ollama_models, ollama_status = await _ollama_models(ollama)
    aws_models, aws_status = _aws_models(bedrock)
    return ModelsResponse(
        models=ollama_models + aws_models,
        providers={"ollama": ollama_status, "aws": aws_status},
    )
