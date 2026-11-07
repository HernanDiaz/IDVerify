"""Inference service: preprocessing, the model wrapper, and a process-wide singleton.

The real model is loaded exactly once per process (see :func:`get_predictor`).
A deterministic :class:`MockPredictor` is used when ``USE_MOCK_MODEL`` is set,
which keeps the fast test suite free of PyTorch weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from PIL import Image

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


@dataclass(slots=True)
class PredictionResult:
    """Raw model outputs for one image."""

    p_manipulated: float
    mask: np.ndarray  # float32, HxW in [0, 1], at model-input resolution
    inference_ms: float


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """Resize to ``size`` x ``size``, scale to [0, 1], return CHW float32.

    Pure NumPy/PIL so it is testable without importing PyTorch.
    """
    rgb = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)).copy()  # (3, size, size)


class Predictor(Protocol):
    """Anything that turns a PIL image into a :class:`PredictionResult`."""

    def predict(self, image: Image.Image) -> PredictionResult: ...


class TorchPredictor:
    """Loads the trained checkpoint once and runs CPU inference."""

    def __init__(self, weights_path: str, input_size: int, num_threads: int) -> None:
        import torch

        from app.services.model_arch import build_model

        torch.set_num_threads(max(1, num_threads))
        self._torch = torch
        self.input_size = input_size

        logger.info("Loading model weights from %s", weights_path)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model = build_model(checkpoint["params"], torch.device("cpu"))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        logger.info("Model ready (threads=%d)", torch.get_num_threads())

    def predict(self, image: Image.Image) -> PredictionResult:
        torch = self._torch
        chw = preprocess(image, self.input_size)
        x = torch.from_numpy(chw).unsqueeze(0)
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model(x)
            p_manipulated = float(torch.sigmoid(out["cls"]).item())
            mask = torch.sigmoid(out["mask"][0, 0]).cpu().numpy().astype(np.float32)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return PredictionResult(p_manipulated=p_manipulated, mask=mask, inference_ms=elapsed_ms)


class MockPredictor:
    """Deterministic fake predictor for fast tests (no PyTorch, no weights)."""

    def __init__(self, input_size: int) -> None:
        self.input_size = input_size

    def predict(self, image: Image.Image) -> PredictionResult:
        chw = preprocess(image, self.input_size)
        # Darker-than-mid images are flagged as manipulated, purely for determinism.
        p_manipulated = 0.9 if float(chw.mean()) < 0.5 else 0.1
        mask = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        if p_manipulated >= 0.5:
            s = self.input_size
            mask[s // 4 : s // 2, s // 4 : (3 * s) // 4] = 1.0
        return PredictionResult(p_manipulated=p_manipulated, mask=mask, inference_ms=0.0)


_predictor: Predictor | None = None


def build_predictor(settings: Settings) -> Predictor:
    """Create a predictor according to settings (mock or real)."""
    if settings.use_mock_model:
        logger.info("Using MockPredictor (USE_MOCK_MODEL=true)")
        return MockPredictor(settings.input_size)
    return TorchPredictor(
        settings.model_weights_path, settings.input_size, settings.torch_num_threads
    )


def get_predictor(settings: Settings | None = None) -> Predictor:
    """Return the process-wide predictor, loading it on first use."""
    global _predictor
    if _predictor is None:
        _predictor = build_predictor(settings or get_settings())
    return _predictor


def reset_predictor() -> None:
    """Drop the cached predictor (used by tests)."""
    global _predictor
    _predictor = None
