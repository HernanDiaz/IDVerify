"""Celery application: Redis broker/backend, Beat schedule and model warm-up."""

from __future__ import annotations

from typing import Any

from celery import Celery
from celery.signals import worker_process_init

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

celery_app = Celery(
    "docverify",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    beat_schedule={
        "cleanup-expired-analyses": {
            "task": "app.workers.tasks.cleanup_expired",
            "schedule": float(settings.cleanup_interval_minutes * 60),
        },
    },
)


def _publish_model_ready(ready: bool) -> None:
    """Record worker model readiness in Redis so /health can report it."""
    import redis

    try:
        client = redis.Redis.from_url(settings.redis_url)
        if ready:
            client.set(settings.model_ready_key, "1")
        else:
            client.delete(settings.model_ready_key)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Could not publish model readiness: %s", exc)


@worker_process_init.connect
def _warm_model(**_: Any) -> None:
    """Load the model once per worker process (not per task)."""
    from app.services.inference import get_predictor

    try:
        get_predictor()
        _publish_model_ready(True)
        logger.info("Worker process initialized; model loaded and ready")
    except Exception as exc:
        logger.exception("Model warm-up failed: %s", exc)
        _publish_model_ready(False)
