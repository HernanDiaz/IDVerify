"""Analyses endpoints: submit, retrieve, list and fetch the mask."""

from __future__ import annotations

import base64
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Pagination, ValidatedImage, pagination, validated_image
from app.db.session import get_session
from app.models.analysis import AnalysisStatus
from app.schemas.analysis import (
    AnalysisCreated,
    AnalysisPage,
    AnalysisRead,
    AnalysisSummary,
)
from app.services import analyses as analyses_service
from app.workers.tasks import run_analysis

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post(
    "",
    response_model=AnalysisCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a document image for analysis",
)
async def submit_analysis(
    image: ValidatedImage = Depends(validated_image),
    session: AsyncSession = Depends(get_session),
) -> AnalysisCreated:
    analysis = await analyses_service.create_analysis(
        session,
        content_type=image.content_type,
        original_filename=image.filename,
        data=image.data,
    )
    run_analysis.delay(str(analysis.id))
    return AnalysisCreated(id=analysis.id, status=analysis.status)


@router.get("", response_model=AnalysisPage, summary="List analyses (paginated)")
async def list_analyses(
    page: Pagination = Depends(pagination),
    session: AsyncSession = Depends(get_session),
) -> AnalysisPage:
    rows, total = await analyses_service.list_analyses(
        session, limit=page.limit, offset=page.offset
    )
    return AnalysisPage(
        items=[AnalysisSummary.model_validate(row) for row in rows],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{analysis_id}", response_model=AnalysisRead, summary="Get one analysis")
async def get_analysis(
    analysis_id: uuid.UUID,
    include_mask: bool = Query(False, description="Embed the mask as base64 when done."),
    session: AsyncSession = Depends(get_session),
) -> AnalysisRead:
    analysis = await analyses_service.get_analysis(session, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")

    mask_base64: str | None = None
    if include_mask and analysis.status == AnalysisStatus.DONE and analysis.mask_path:
        mask_file = Path(analysis.mask_path)
        if mask_file.exists():
            mask_base64 = base64.b64encode(mask_file.read_bytes()).decode("ascii")

    return AnalysisRead.from_model(analysis, mask_base64=mask_base64)


@router.get(
    "/{analysis_id}/mask",
    summary="Download the predicted mask (PNG)",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_mask(
    analysis_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    analysis = await analyses_service.get_analysis(session, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    if (
        analysis.status != AnalysisStatus.DONE
        or not analysis.mask_path
        or not Path(analysis.mask_path).exists()
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mask not available (analysis not finished or already purged).",
        )
    return FileResponse(
        analysis.mask_path, media_type="image/png", filename=f"{analysis_id}_mask.png"
    )
