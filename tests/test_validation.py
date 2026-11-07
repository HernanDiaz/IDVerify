"""Upload validation: content type, magic bytes, emptiness and size limits."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.core.config import get_settings


async def test_rejects_non_image_content_type(client: AsyncClient) -> None:
    response = await client.post(
        "/analyses", files={"file": ("doc.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 422


async def test_rejects_declared_png_with_wrong_bytes(client: AsyncClient) -> None:
    response = await client.post(
        "/analyses", files={"file": ("x.png", b"definitely not a png", "image/png")}
    )
    assert response.status_code == 422


async def test_rejects_empty_file(client: AsyncClient) -> None:
    response = await client.post("/analyses", files={"file": ("x.png", b"", "image/png")})
    assert response.status_code == 422


async def test_rejects_oversized_file(
    client: AsyncClient, png_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "max_upload_bytes", 10)
    response = await client.post(
        "/analyses", files={"file": ("x.png", png_bytes, "image/png")}
    )
    assert response.status_code == 413
