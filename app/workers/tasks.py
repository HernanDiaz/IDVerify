"""Celery tasks: run one analysis, and the periodic retention sweep."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.sync_session import session_scope
from app.models.analysis import Analysis, AnalysisStatus
from app.services import storage
from app.services.inference import get_predictor
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="app.workers.tasks.run_analysis", bind=True, max_retries=2)
def run_analysis(self: Any, analysis_id: str) -> dict[str, Any]:
    """Load the stored image, run inference, persist verdict + mask."""
    aid = uuid.UUID(analysis_id)

    with session_scope() as session:
        analysis = session.get(Analysis, aid)
        if analysis is None:
            logger.warning("run_analysis: %s not found", analysis_id)
            return {"status": "missing"}
        analysis.status = AnalysisStatus.RUNNING
        analysis.started_at = datetime.now(timezone.utc)
        image_path = analysis.image_path

    try:
        if not image_path or not Path(image_path).exists():
            raise FileNotFoundError(f"stored image not found: {image_path}")

        with Image.open(image_path) as img:
            img.load()
            result = get_predictor().predict(img)

        mask_path, coverage = storage.save_mask(aid, result.mask)
        verdict = "manipulated" if result.p_manipulated >= 0.5 else "authentic"

        with session_scope() as session:
            analysis = session.get(Analysis, aid)
            if analysis is None:  # pragma: no cover - deleted mid-flight
                return {"status": "missing"}
            analysis.status = AnalysisStatus.DONE
            analysis.verdict = verdict
            analysis.p_manipulated = result.p_manipulated
            analysis.p_authentic = 1.0 - result.p_manipulated
            analysis.pred_coverage = coverage
            analysis.inference_ms = result.inference_ms
            analysis.mask_path = mask_path
            analysis.finished_at = datetime.now(timezone.utc)

        logger.info("Analysis %s -> %s (%.1f ms)", analysis_id, verdict, result.inference_ms)
        return {"status": "done", "verdict": verdict}

    except Exception as exc:
        logger.exception("Analysis %s failed", analysis_id)
        with session_scope() as session:
            analysis = session.get(Analysis, aid)
            if analysis is not None:
                analysis.status = AnalysisStatus.FAILED
                analysis.error = str(exc)[:512]
                analysis.finished_at = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(exc)}


@celery_app.task(name="app.workers.tasks.cleanup_expired")
def cleanup_expired() -> dict[str, int]:
    """Delete images and masks older than the retention window (privacy)."""
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.retention_hours)
    removed = 0

    with session_scope() as session:
        rows = (
            session.execute(select(Analysis).where(Analysis.created_at < cutoff))
            .scalars()
            .all()
        )
        for analysis in rows:
            if analysis.image_path or analysis.mask_path:
                storage.delete_file(analysis.image_path)
                storage.delete_file(analysis.mask_path)
                analysis.image_path = None
                analysis.mask_path = None
                removed += 1

    if removed:
        logger.info("Retention sweep cleared files for %d analyses", removed)
    return {"removed": removed}
