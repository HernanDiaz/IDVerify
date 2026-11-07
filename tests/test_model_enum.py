"""Guard against the status enum name/value mismatch.

The Alembic migration creates the native Postgres enum with the *values*
("pending", ...) as labels. SQLAlchemy must serialize the column with those same
values, not the member names ("PENDING", ...), or Postgres rejects every insert
with ``invalid input value for enum``. SQLite has no native enum, so this only
surfaces on Postgres — hence this cheap, DB-free guard.
"""

from __future__ import annotations

from app.models.analysis import Analysis, AnalysisStatus

EXPECTED_LABELS = ["pending", "running", "done", "failed"]


def test_status_enum_serializes_lowercase_values() -> None:
    # The labels SQLAlchemy will emit in DDL and bind on insert.
    assert Analysis.__table__.c.status.type.enums == EXPECTED_LABELS


def test_status_enum_labels_are_member_values_not_names() -> None:
    assert [m.value for m in AnalysisStatus] == EXPECTED_LABELS
    assert [m.name for m in AnalysisStatus] == ["PENDING", "RUNNING", "DONE", "FAILED"]
    # The DB labels must be the values, never the names.
    assert Analysis.__table__.c.status.type.enums == [m.value for m in AnalysisStatus]
