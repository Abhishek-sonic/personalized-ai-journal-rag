"""FastAPI app factory: logging, routers, startup validation, static frontend."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers import auth_routes, chat, entries
from app.services.embeddings import get_embedder, validate_embedding_dim
from app.services.llm import get_llm

log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


def configure_logging(level: str) -> None:
    # Readable single-line format: the [RAG]/[INGEST] logs are shown in the demo.
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # replace uvicorn's handler so our format wins
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Log the active configuration (never the keys) so the demo shows which provider is live.
    log.info("[STARTUP] llm_provider=%s base_url=%s model=%s", settings.llm_provider, settings.llm_base_url, settings.llm_model)
    log.info("[STARTUP] embedding_provider=%s model=%s dim=%d", settings.embedding_provider, settings.embedding_model, settings.embedding_dim)
    get_llm()
    # Fail fast if the embedding model's size disagrees with vector(N) in schema.sql;
    # otherwise every insert would fail later with an opaque Postgres error.
    validate_embedding_dim(get_embedder(), settings.embedding_dim)
    log.info("[STARTUP] ready")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="AI Journal", version="1.0.0", lifespan=lifespan)
    app.include_router(auth_routes.router)
    app.include_router(entries.router)
    app.include_router(chat.router)

    @app.get("/api/health", tags=["health"])
    def health():
        # Deliberately contains no secrets - safe to expose.
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
        }

    # Mounted last so /api/* routes take precedence. html=True serves index.html at "/".
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
