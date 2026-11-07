"""Synchronous engine and session factory for the Celery worker.

Celery tasks run in a synchronous context. Using a plain blocking session here
avoids the pitfalls of driving an asyncio engine from inside a worker process
(event-loop/connection affinity), while sharing the exact same ORM models.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

_settings = get_settings()

sync_engine = create_engine(
    _settings.sync_database_url,
    pool_pre_ping=True,
    future=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False,
    class_=Session,
)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of worker operations."""
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
