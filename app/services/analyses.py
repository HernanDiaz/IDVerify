"""Async data-access helpers for analyses (used by the API layer)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import Analysis, AnalysisStatus
from app.services import storage


async def create_analysis(
    session: AsyncSession,
    *,
    content_type: str,
    original_filename: str | None,
    data: bytes,
) -> Analysis:
    """Insert a pending analysis and persist the uploaded image to disk.

    The row is flushed first to obtain its id, which names the stored file.
    """
    analysis = Analysis(
        status=AnalysisStatus.PENDING,
        content_type=content_type,
        original_filename=original_filename,
    )
    session.add(analysis)
    await session.flush()  # assign primary key
    analysis.image_path = storage.save_upload(analysis.id, content_type, data)
    await session.commit()
    await session.refresh(analysis)
    return analysis


async def get_analysis(session: AsyncSession, analysis_id: uuid.UUID) -> Analysis | None:
    return await session.get(Analysis, analysis_id)


async def list_analyses(
    session: AsyncSession, *, limit: int, offset: int
) -> tuple[Sequence[Analysis], int]:
    total = await session.scalar(select(func.count()).select_from(Analysis))
    result = await session.execute(
        select(Analysis).order_by(Analysis.created_at.desc()).limit(limit).offset(offset)
    )
    return result.scalars().all(), int(total or 0)
