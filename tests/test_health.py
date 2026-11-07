"""Health endpoint."""

from __future__ import annotations

from httpx import AsyncClient


async def test_health_reports_all_components(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code in (200, 503)
    body = response.json()
    assert set(body["components"]) == {"database", "redis", "model"}
    # Database and (mock) model must be healthy regardless of Redis availability.
    assert body["components"]["database"]["status"] == "ok"
    assert body["components"]["model"]["status"] == "ok"
    assert body["version"]
