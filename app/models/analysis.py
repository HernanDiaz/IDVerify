"""The ``analyses`` table: one row per submitted document analysis.

Note: the original uploaded image is never stored in the database. Only a
filesystem path to the temporary file is kept (and it is deleted by the
retention job), together with the model's outputs.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AnalysisStatus(str, enum.Enum):
    """Lifecycle of an analysis."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[AnalysisStatus] = mapped_column(
        # Store the enum *values* ("pending", …), not the member names
        # ("PENDING"). This matches the labels the Alembic migration creates for
        # the native Postgres enum and the lowercase status the API exposes;
        # without it SQLAlchemy binds the name and Postgres rejects the insert.
        Enum(
            AnalysisStatus,
            name="analysis_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=AnalysisStatus.PENDING,
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Request metadata
    content_type: Mapped[str] = mapped_column(String(32), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255))

    # Filesystem pointers (never image bytes in the DB)
    image_path: Mapped[str | None] = mapped_column(String(512))
    mask_path: Mapped[str | None] = mapped_column(String(512))

    # Model outputs (populated when status == DONE)
    verdict: Mapped[str | None] = mapped_column(String(16))
    p_manipulated: Mapped[float | None] = mapped_column(Float)
    p_authentic: Mapped[float | None] = mapped_column(Float)
    pred_coverage: Mapped[float | None] = mapped_column(Float)
    inference_ms: Mapped[float | None] = mapped_column(Float)

    # Failure detail (populated when status == FAILED)
    error: Mapped[str | None] = mapped_column(String(512))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Analysis id={self.id} status={self.status.value}>"
