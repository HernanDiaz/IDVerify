"""Test configuration.

Environment variables are set *before* importing the application so the module
level engines point at a throwaway SQLite file shared by the async API and the
synchronous Celery worker. The fast suite therefore needs no external services:
SQLite + mock model + Celery eager mode.
"""

from __future__ import annotations

import io
import os
import pathlib
import tempfile
from collections.abc import AsyncIterator, Iterator

import pytest

_TMP = pathlib.Path(tempfile.mkdtemp(prefix="docverify_test_"))
_DB_FILE = (_TMP / "test.db").as_posix()

# Defaults for a zero-service local run (SQLite + mock model). CI may inject
# DATABASE_URL/REDIS_URL to run the same suite against Postgres and Redis, so
# only fill in what the environment does not already provide.
for _key, _value in {
    "DATABASE_URL": f"sqlite+aiosqlite:///{_DB_FILE}",
    "USE_MOCK_MODEL": "true",
    "UPLOAD_DIR": str(_TMP / "uploads"),
    "MASK_DIR": str(_TMP / "masks"),
    "REDIS_URL": "redis://localhost:6379/15",
    "RETENTION_HOURS": "24",
    "INPUT_SIZE": "224",
}.items():
    os.environ.setdefault(_key, _value)

# Imported after the environment is configured.
from PIL import Image  # noqa: E402
from sqlalchemy import event  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool  # noqa: E402

import app.db.session as db_session  # noqa: E402
import app.models  # noqa: E402, F401  (register tables on the metadata)
from app.db.base import Base  # noqa: E402
from app.db.sync_session import sync_engine  # noqa: E402
from app.workers.celery_app import celery_app  # noqa: E402

# Rebuild the async engine with NullPool and rebind it on the module. pytest
# runs each async test on its own event loop; asyncpg refuses to reuse a pooled
# connection created on a now-closed loop ("attached to a different loop").
# NullPool opens and closes a connection within a single test's loop, so none is
# ever carried across loops. get_session() resolves AsyncSessionLocal from the
# module at call time, so rebinding here is enough. (SQLite tolerated the shared
# pool locally, but Postgres in CI does not.)
async_engine = create_async_engine(
    os.environ["DATABASE_URL"],
    poolclass=NullPool,
    future=True,
)
db_session.engine = async_engine
db_session.AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


def _apply_sqlite_pragmas(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()


# PRAGMAs only make sense on SQLite; skip them when CI points at Postgres.
if sync_engine.dialect.name == "sqlite":
    event.listen(sync_engine, "connect", _apply_sqlite_pragmas)
    event.listen(async_engine.sync_engine, "connect", _apply_sqlite_pragmas)

# Run Celery tasks inline, in-process, and surface their exceptions.
celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (96, 96), color: int = 200) -> bytes:
    """Return encoded bytes of a solid RGB image."""
    buf = io.BytesIO()
    Image.new("RGB", size, (color, color, color)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    return make_image_bytes("PNG")


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_image_bytes("JPEG")


# --------------------------------------------------------------------------- #
# Database lifecycle
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session", autouse=True)
def _schema() -> Iterator[None]:
    Base.metadata.create_all(sync_engine)
    yield
    Base.metadata.drop_all(sync_engine)


@pytest.fixture(autouse=True)
def _clean_tables() -> Iterator[None]:
    from app.models.analysis import Analysis

    with sync_engine.begin() as conn:
        conn.execute(Analysis.__table__.delete())
    yield


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:  # noqa: F821
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
