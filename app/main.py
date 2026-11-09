from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import health, models
from app.config import get_settings
from app.llm.ollama import OllamaClient

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app.state.ollama = OllamaClient(settings.ollama_base_url, keep_alive=settings.ollama_keep_alive)
    yield
    await app.state.ollama.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="PDF RAG", version=__version__, lifespan=lifespan)
    app.include_router(health.router, prefix="/api")
    app.include_router(models.router, prefix="/api")

    # The web UI is plain static files served by this same server.
    # Mounted last so every /api route takes priority.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


app = create_app()
