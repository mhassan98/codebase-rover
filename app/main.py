from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import router
from app.catalog import apply_config_file, seed
from app.config import get_settings
from app.db import init_db, session_scope
from app.indexing.service import recover_stale_jobs
from app.logging_config import configure_logging

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    from app.indexing.limits import EMBEDDING_DIM

    if settings.embedding_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"EMBEDDING_DIM is {settings.embedding_dim}, but this schema stores {EMBEDDING_DIM}-dimensional vectors."
        )
    init_db()
    with session_scope() as session:
        seed(session)
        apply_config_file(session, settings)
    recover_stale_jobs()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Codebase Rover", version=__version__, lifespan=lifespan)
    app.include_router(router, prefix="/api")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
