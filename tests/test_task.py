"""Celery task behaviour: retention sweep and failure handling."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from app.db.sync_session import session_scope
from app.models.analysis import Analysis, AnalysisStatus
from app.services import storage
from app.workers.tasks import cleanup_expired, run_analysis


def test_cleanup_removes_expired_files() -> None:
    analysis_id = uuid.uuid4()
    image_path = storage.save_upload(analysis_id, "image/png", b"\x89PNG\r\n\x1a\n binary")
    mask_path, _ = storage.save_mask(analysis_id, np.zeros((8, 8), dtype=np.float32))

    with session_scope() as session:
        session.add(
            Analysis(
                id=analysis_id,
                status=AnalysisStatus.DONE,
                content_type="image/png",
                image_path=image_path,
                mask_path=mask_path,
                created_at=datetime.now(UTC) - timedelta(hours=48),
            )
        )

    result = cleanup_expired()

    assert result["removed"] >= 1
    assert not Path(image_path).exists()
    assert not Path(mask_path).exists()
    with session_scope() as session:
        refreshed = session.get(Analysis, analysis_id)
        assert refreshed is not None
        assert refreshed.image_path is None
        assert refreshed.mask_path is None


def test_run_analysis_marks_failed_when_image_missing() -> None:
    analysis_id = uuid.uuid4()
    with session_scope() as session:
        session.add(
            Analysis(
                id=analysis_id,
                status=AnalysisStatus.PENDING,
                content_type="image/png",
                image_path="/nonexistent/missing.png",
            )
        )

    output = run_analysis.run(str(analysis_id))

    assert output["status"] == "failed"
    with session_scope() as session:
        refreshed = session.get(Analysis, analysis_id)
        assert refreshed is not None
        assert refreshed.status == AnalysisStatus.FAILED
        assert refreshed.error
