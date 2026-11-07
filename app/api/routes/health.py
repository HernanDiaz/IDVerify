"""Liveness/readiness endpoint: API, database, Redis and model status."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.schemas.health import ComponentHealth, HealthResponse

router = APIRouter(tags=["health"])


async def _check_database(session: AsyncSession) -> ComponentHealth:
    try:
        await session.execute(text("SELECT 1"))
        return ComponentHealth(status="ok")
    except Exception as exc:
        return ComponentHealth(status="error", detail=str(exc))


async def _check_redis(settings: Settings) -> ComponentHealth:
    import redis.asyncio as aioredis

    client = aioredis.Redis.from_url(settings.redis_url)
    try:
        await client.ping()
        return ComponentHealth(status="ok")
    except Exception as exc:
        return ComponentHealth(status="error", detail=str(exc))
    finally:
        await client.aclose()


async def _check_model(settings: Settings) -> ComponentHealth:
    if settings.use_mock_model:
        return ComponentHealth(status="ok", detail="mock model")

    # The worker publishes readiness to Redis after loading the model.
    import redis.asyncio as aioredis

    client = aioredis.Redis.from_url(settings.redis_url)
    try:
        ready = await client.get(settings.model_ready_key)
    except Exception as exc:
        return ComponentHealth(status="error", detail=f"redis unavailable: {exc}")
    finally:
        await client.aclose()

    if ready:
        return ComponentHealth(status="ok", detail="worker loaded the model")

    weights_present = Path(settings.model_weights_path).is_file()
    detail = (
        "weights present, worker not warmed yet"
        if weights_present
        else "worker not ready and weights file missing"
    )
    return ComponentHealth(status="error", detail=detail)


@router.get("/health", response_model=HealthResponse, summary="Service health")
async def health(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    components = {
        "database": await _check_database(session),
        "redis": await _check_redis(settings),
        "model": await _check_model(settings),
    }
    healthy = all(c.status == "ok" for c in components.values())
    payload = HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        components=components,
    )
    return JSONResponse(
        status_code=200 if healthy else 503,
        content=payload.model_dump(),
    )
