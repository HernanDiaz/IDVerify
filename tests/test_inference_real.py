"""Real-model inference (slow): requires the trained weights.

Run with ``pytest -m slow`` and MODEL_WEIGHTS_PATH pointing at the checkpoint.
Skipped automatically when the weights (or torch) are unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.slow


def test_real_model_inference() -> None:
    weights = os.environ.get("MODEL_WEIGHTS_PATH", "/models/model_multitask_seed47.pt")
    if not Path(weights).is_file():
        pytest.skip(f"weights not found at {weights}")
    pytest.importorskip("torch")

    from app.services.inference import TorchPredictor

    predictor = TorchPredictor(weights, input_size=224, num_threads=1)
    image = Image.new("RGB", (512, 320), (180, 170, 160))
    result = predictor.predict(image)

    assert 0.0 <= result.p_manipulated <= 1.0
    assert result.mask.shape == (224, 224)
    assert result.inference_ms >= 0.0
