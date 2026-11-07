"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app import __version__
from app.api.routes import analyses, health
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.storage import ensure_dirs

logger = get_logger(__name__)

DESCRIPTION = """
Asynchronous REST microservice that detects whether an identity document is
**authentic** or **manipulated** and localizes the altered regions with a mask.

Submit an image to `POST /analyses`; it is queued to a Celery worker for CPU
inference. Poll `GET /analyses/{id}` for the verdict, probabilities and mask.

This is a **research demo** — do not upload real identity documents. Stored
images and masks are deleted automatically after the retention window.
"""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    ensure_dirs()
    settings = get_settings()
    logger.info("DocVerify API %s starting (env=%s)", __version__, settings.environment)
    yield
    logger.info("DocVerify API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocVerify API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(analyses.router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
