"""Application configuration, loaded from environment variables (and an optional .env)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings for the API and the Celery worker.

    Values are read from environment variables (case-insensitive) or a local
    ``.env`` file. See ``.env.example`` for the full list.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # ``model_*`` fields would otherwise clash with Pydantic's protected namespace.
        protected_namespaces=(),
    )

    # --- API ---
    app_name: str = "DocVerify API"
    environment: str = "development"

    # --- PostgreSQL ---
    # Async URL used by the API; the sync URL for Alembic/worker is derived from it.
    database_url: str = "postgresql+asyncpg://docverify:docverify@localhost:5432/docverify"

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # --- Model ---
    model_weights_path: str = "/models/model_multitask_seed47.pt"
    use_mock_model: bool = False
    input_size: int = 224
    torch_num_threads: int = 1
    model_ready_key: str = "docverify:worker:model_ready"

    # --- Storage (filesystem, never the database) ---
    upload_dir: str = "/data/uploads"
    mask_dir: str = "/data/masks"

    # --- Upload validation ---
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MiB
    allowed_content_types: tuple[str, ...] = ("image/jpeg", "image/png")

    # --- Privacy / retention ---
    retention_hours: int = 24
    cleanup_interval_minutes: int = 60

    @property
    def sync_database_url(self) -> str:
        """Synchronous (psycopg) URL for Alembic migrations and the Celery worker."""
        return self.database_url.replace("+asyncpg", "+psycopg")

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
