"""initial analyses table

Revision ID: 0001
Revises:
Create Date: 2026-11-07
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "done", "failed", name="analysis_status"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content_type", sa.String(length=32), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("image_path", sa.String(length=512), nullable=True),
        sa.Column("mask_path", sa.String(length=512), nullable=True),
        sa.Column("verdict", sa.String(length=16), nullable=True),
        sa.Column("p_manipulated", sa.Float(), nullable=True),
        sa.Column("p_authentic", sa.Float(), nullable=True),
        sa.Column("pred_coverage", sa.Float(), nullable=True),
        sa.Column("inference_ms", sa.Float(), nullable=True),
        sa.Column("error", sa.String(length=512), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_analyses"),
    )
    op.create_index("ix_analyses_status", "analyses", ["status"])
    op.create_index("ix_analyses_created_at", "analyses", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_analyses_created_at", table_name="analyses")
    op.drop_index("ix_analyses_status", table_name="analyses")
    op.drop_table("analyses")
    sa.Enum(name="analysis_status").drop(op.get_bind(), checkfirst=True)
