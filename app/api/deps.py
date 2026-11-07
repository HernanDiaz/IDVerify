"""Shared FastAPI dependencies: upload validation and pagination."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, File, HTTPException, Query, UploadFile, status

from app.core.config import Settings, get_settings

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass(slots=True)
class ValidatedImage:
    content_type: str
    filename: str | None
    data: bytes


def _magic_matches(data: bytes, content_type: str) -> bool:
    if content_type == "image/jpeg":
        return data[:3] == _JPEG_MAGIC
    if content_type == "image/png":
        return data[:8] == _PNG_MAGIC
    return False


async def validated_image(
    file: UploadFile = File(..., description="Document image (JPEG or PNG)."),
    settings: Settings = Depends(get_settings),
) -> ValidatedImage:
    """Validate the uploaded file's type, size and magic bytes.

    Raises 422 for an unsupported/invalid image and 413 when it is too large.
    """
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Allowed: {', '.join(settings.allowed_content_types)}."
            ),
        )

    # Read at most max+1 bytes so an oversized upload is rejected without
    # buffering the whole payload.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the maximum size of {settings.max_upload_bytes} bytes.",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file."
        )
    if not _magic_matches(data, file.content_type):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File content is not a valid JPEG or PNG image.",
        )

    return ValidatedImage(content_type=file.content_type, filename=file.filename, data=data)


@dataclass(slots=True)
class Pagination:
    limit: int
    offset: int


def pagination(
    limit: int = Query(20, ge=1, le=100, description="Page size (max 100)."),
    offset: int = Query(0, ge=0, description="Number of items to skip."),
) -> Pagination:
    return Pagination(limit=limit, offset=offset)
