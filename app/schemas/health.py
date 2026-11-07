"""Health-check schemas."""

from __future__ import annotations

from pydantic import BaseModel


class ComponentHealth(BaseModel):
    status: str  # "ok" | "error"
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    version: str
    components: dict[str, ComponentHealth]
