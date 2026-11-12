import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.answer import Answerer
from app.api import aws, chat, chats, documents, extractors, health, models, search
from app.config import Settings, get_settings
from app.ingest.jobs import IngestQueue
from app.ingest.vision import VisionService
from app.llm.bedrock import BedrockClient
from app.llm.embedder import Embedder
from app.llm.ollama import OllamaClient
from app.search import Searcher
from app.store import documents as document_store
from app.store.db import Database
from app.store.vectors import VectorStore

STATIC_DIR = Path(__file__).parent / "static"

log = logging.getLogger("app")


class UIFiles(StaticFiles):
    """The web page's files. Browsers check for a newer copy on every load (a quick "not modified"
    when nothing changed), so an updated app shows up without a hard refresh."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def make_ollama(settings: Settings) -> OllamaClient:
    return OllamaClient(
        settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive,
        keep_loaded=(settings.ollama_embed_model,),
    )


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

    app.state.ollama = make_ollama(settings)
    app.state.embedder = Embedder(app.state.ollama, settings.ollama_embed_model, settings.embed_batch_size)
    app.state.bedrock = make_bedrock(settings)
    app.state.searcher = Searcher(app.state.db, app.state.vectors, app.state.embedder)
    app.state.answerer = Answerer(
        app.state.searcher,
        app.state.ollama,
        app.state.bedrock,
        settings.data_dir,
        num_ctx=settings.chat_num_ctx,
        max_sources=settings.chat_max_sources,
        max_source_chars=settings.chat_max_source_chars,
        max_images=settings.chat_max_images,
        max_answer_tokens=settings.chat_max_answer_tokens,
    )
    app.state.vision = VisionService(
        app.state.ollama,
        app.state.bedrock,
        local_model=settings.vision_local_model,
        num_ctx=settings.vision_num_ctx,
        max_megapixels=settings.vision_max_megapixels,
        aws_concurrency=settings.vision_aws_concurrency,
    )

    app.state.ingest = IngestQueue(
        app.state.db,
        settings.data_dir,
        settings.page_preview_dpi,
        settings.figure_dpi,
        vision=app.state.vision,
        embedder=app.state.embedder,
        vectors=app.state.vectors,
    )
    app.state.ingest.start()
    # Logged in the background, so a slow Ollama can't hold up startup.
    summary = asyncio.create_task(log_startup_summary(app, settings), name="startup-summary")
    yield
    summary.cancel()
    with suppress(asyncio.CancelledError):
        await summary
    await app.state.ingest.stop()
    await app.state.ollama.aclose()


async def log_startup_summary(app: FastAPI, settings: Settings) -> None:
    """A few lines in the log saying what is ready, and what to fix if something isn't."""
    try:
        state = app.state
        documents = len(await asyncio.to_thread(document_store.list_all, state.db))
        chunks = await asyncio.to_thread(state.vectors.count)
        log.info("PDF RAG %s on http://%s:%s", __version__, settings.app_host, settings.app_port)
        log.info("Data in %s: schema v%d, %d documents, %d search chunks",
                 settings.data_dir, state.db.ping(), documents, chunks)
        problem = await state.embedder.problem()
        log.info("Embedding model %s: %s", settings.ollama_embed_model, problem or "ready")
        for extractor in ("local", "aws"):
            problem = await state.vision.problem(extractor)
            log.info("Charts read by %s: %s", state.vision.label(extractor), problem or "ready")
        problem = state.bedrock.qwen_problem()
        log.info("Qwen2.5-VL on AWS for answers: %s", problem or "ready")
    except Exception:  # noqa: BLE001 - a summary must never take the app down
        log.exception("Couldn't write the startup summary")


async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unexpected error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on the server. Details are in the app log: docker compose logs app"},
    )


def create_app() -> FastAPI:
    app = FastAPI(title="PDF RAG", version=__version__, lifespan=lifespan)
    app.add_exception_handler(Exception, unexpected_error)
    app.include_router(health.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(aws.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(extractors.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(chats.router, prefix="/api")

    # The web UI is plain static files served by this same server.
    # Mounted last so every /api route takes priority.
    app.mount("/", UIFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


app = create_app()
