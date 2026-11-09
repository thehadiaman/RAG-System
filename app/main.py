from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import aws, health, models
from app.config import Settings, get_settings
from app.llm.bedrock import BedrockClient
from app.llm.embedder import Embedder
from app.llm.ollama import OllamaClient
from app.store.db import Database
from app.store.vectors import VectorStore

STATIC_DIR = Path(__file__).parent / "static"


def make_bedrock(settings: Settings) -> BedrockClient:
    return BedrockClient(
        region=settings.aws_region,
        nova_model_id=settings.bedrock_nova_lite_model_id,
        qwen_model_arn=settings.bedrock_qwen_vl_model_arn,
        access_key_id=settings.secret("aws_access_key_id"),
        secret_access_key=settings.secret("aws_secret_access_key"),
        session_token=settings.secret("aws_session_token"),
        cold_start_wait=settings.bedrock_cold_start_wait,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    app.state.db = Database(settings.data_dir / "app.db")
    app.state.db.migrate()
    app.state.vectors = VectorStore(settings.data_dir / "chroma")

    app.state.ollama = OllamaClient(
        settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive,
        keep_loaded=(settings.ollama_embed_model,),
    )
    app.state.embedder = Embedder(app.state.ollama, settings.ollama_embed_model, settings.embed_batch_size)
    app.state.bedrock = make_bedrock(settings)
    yield
    await app.state.ollama.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="PDF RAG", version=__version__, lifespan=lifespan)
    app.include_router(health.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(aws.router, prefix="/api")

    # The web UI is plain static files served by this same server.
    # Mounted last so every /api route takes priority.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


app = create_app()
