"""Filesystem storage for uploaded images and generated masks.

Images and masks live on disk (not in the database) and are removed by the
retention job. See :mod:`app.workers.tasks`.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import get_settings

_EXT_BY_CONTENT_TYPE = {"image/jpeg": "jpg", "image/png": "png"}


def ensure_dirs() -> None:
    settings = get_settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.mask_dir).mkdir(parents=True, exist_ok=True)


def save_upload(analysis_id: uuid.UUID, content_type: str, data: bytes) -> str:
    """Persist the uploaded image bytes; return the filesystem path."""
    ensure_dirs()
    ext = _EXT_BY_CONTENT_TYPE.get(content_type, "bin")
    path = Path(get_settings().upload_dir) / f"{analysis_id}.{ext}"
    path.write_bytes(data)
    return str(path)


def save_mask(analysis_id: uuid.UUID, mask: np.ndarray, threshold: float = 0.5) -> tuple[str, float]:
    """Binarize the mask, save it as a grayscale PNG, and return (path, coverage%)."""
    ensure_dirs()
    binary = mask > threshold
    image = Image.fromarray((binary * 255).astype(np.uint8), mode="L")
    path = Path(get_settings().mask_dir) / f"{analysis_id}.png"
    image.save(path, format="PNG")
    coverage_pct = float(binary.mean() * 100.0)
    return str(path), coverage_pct


def delete_file(path: str | None) -> None:
    """Best-effort deletion of a stored file."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
