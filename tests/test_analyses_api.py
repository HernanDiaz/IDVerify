"""End-to-end API flow with the mock model and Celery eager mode."""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient


async def _submit(client: AsyncClient, png_bytes: bytes) -> str:
    response = await client.post(
        "/analyses", files={"file": ("doc.png", png_bytes, "image/png")}
    )
    assert response.status_code == 201
    return response.json()["id"]


async def test_submit_returns_pending_id(client: AsyncClient, png_bytes: bytes) -> None:
    response = await client.post(
        "/analyses", files={"file": ("doc.png", png_bytes, "image/png")}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    uuid.UUID(body["id"])  # parseable


async def test_result_is_available_after_processing(
    client: AsyncClient, png_bytes: bytes
) -> None:
    analysis_id = await _submit(client, png_bytes)
    response = await client.get(f"/analyses/{analysis_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["verdict"] in ("authentic", "manipulated")
    probs = body["probabilities"]
    assert probs["manipulated"] + probs["authentic"] == pytest.approx(1.0, abs=1e-6)
    assert body["mask_url"] == f"/analyses/{analysis_id}/mask"
    assert body["inference_ms"] is not None


async def test_mask_download(client: AsyncClient, png_bytes: bytes) -> None:
    analysis_id = await _submit(client, png_bytes)
    response = await client.get(f"/analyses/{analysis_id}/mask")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


async def test_include_mask_returns_base64(client: AsyncClient, png_bytes: bytes) -> None:
    analysis_id = await _submit(client, png_bytes)
    response = await client.get(f"/analyses/{analysis_id}", params={"include_mask": "true"})
    assert response.status_code == 200
    assert response.json()["mask_base64"]


async def test_list_is_paginated(client: AsyncClient, png_bytes: bytes) -> None:
    for _ in range(3):
        await _submit(client, png_bytes)
    response = await client.get("/analyses", params={"limit": 2, "offset": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 0


async def test_unknown_id_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/analyses/{uuid.uuid4()}")
    assert response.status_code == 404
