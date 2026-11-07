"""Response schemas for the analyses endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.analysis import Analysis, AnalysisStatus


class AnalysisCreated(BaseModel):
    """Returned by ``POST /analyses`` right after enqueueing."""

    id: uuid.UUID
    status: AnalysisStatus = Field(examples=[AnalysisStatus.PENDING])


class Probabilities(BaseModel):
    manipulated: float = Field(ge=0.0, le=1.0)
    authentic: float = Field(ge=0.0, le=1.0)


class AnalysisRead(BaseModel):
    """Full view of a single analysis."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: AnalysisStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    content_type: str
    original_filename: str | None = None

    # Populated when status == done
    verdict: str | None = None
    probabilities: Probabilities | None = None
    pred_coverage: float | None = Field(
        default=None, description="Predicted altered area, percent of the document."
    )
    inference_ms: float | None = None
    mask_url: str | None = None
    mask_base64: str | None = Field(
        default=None, description="PNG mask as base64, only when explicitly requested."
    )

    # Populated when status == failed
    error: str | None = None

    @classmethod
    def from_model(cls, obj: Analysis, *, mask_base64: str | None = None) -> AnalysisRead:
        probabilities = None
        if obj.p_manipulated is not None and obj.p_authentic is not None:
            probabilities = Probabilities(
                manipulated=obj.p_manipulated, authentic=obj.p_authentic
            )
        mask_url = (
            f"/analyses/{obj.id}/mask"
            if obj.status == AnalysisStatus.DONE and obj.mask_path
            else None
        )
        return cls(
            id=obj.id,
            status=obj.status,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            started_at=obj.started_at,
            finished_at=obj.finished_at,
            content_type=obj.content_type,
            original_filename=obj.original_filename,
            verdict=obj.verdict,
            probabilities=probabilities,
            pred_coverage=obj.pred_coverage,
            inference_ms=obj.inference_ms,
            mask_url=mask_url,
            mask_base64=mask_base64,
            error=obj.error,
        )


class AnalysisSummary(BaseModel):
    """Compact view used in the paginated listing."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: AnalysisStatus
    created_at: datetime
    verdict: str | None = None


class AnalysisPage(BaseModel):
    """A page of analyses."""

    items: list[AnalysisSummary]
    total: int
    limit: int
    offset: int
