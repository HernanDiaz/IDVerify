"""Preprocessing (torch-free)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.services.inference import preprocess


def test_preprocess_shape_dtype_and_range() -> None:
    image = Image.new("RGB", (640, 480), (128, 64, 200))
    arr = preprocess(image, 224)
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32
    assert float(arr.min()) >= 0.0
    assert float(arr.max()) <= 1.0


def test_preprocess_is_normalized() -> None:
    image = Image.new("RGB", (10, 10), (255, 255, 255))
    arr = preprocess(image, 224)
    assert np.allclose(arr, 1.0)
