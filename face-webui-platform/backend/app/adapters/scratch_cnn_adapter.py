"""
ScratchCNN ModelAdapter — wraps the from-scratch TinyCNN detector
from realtime-face-detection-dl/src/ and exposes it through the
ModelManager interface.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import get_settings
from app.services.model_manager import ModelAdapter

logger = logging.getLogger(__name__)
settings = get_settings()

# Path to ML engine src/ so we can import ScratchCNNDetectorWrapper
_ML_SRC = settings.ml_engine_root / "src"


def _ensure_ml_src_on_path() -> None:
    src = str(_ML_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


class ScratchCNNAdapter(ModelAdapter):
    """
    Adapter wrapping ScratchCNNDetectorWrapper from the ML engine src/.
    Converts (x, y, w, h, confidence) detections → {"x1","y1","x2","y2","confidence","class"}.
    """

    # ── identity ──────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return "scratch_cnn"

    @property
    def framework(self) -> str:
        return "PyTorch"

    # ── lifecycle ─────────────────────────────────────────────────────────
    def __init__(
        self,
        weights_path: Optional[Path] = None,
        confidence_threshold: float = 0.75,
        window_stride: int = 32,
    ) -> None:
        self._weights_path = weights_path or (settings.models_dir / "scratch_cnn.pth")
        self._confidence_threshold = confidence_threshold
        self._window_stride = window_stride
        self._detector = None

    def load(self) -> None:
        _ensure_ml_src_on_path()
        try:
            import torch
            from detector import ScratchCNNDetectorWrapper  # type: ignore[import-untyped]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._detector = ScratchCNNDetectorWrapper(
                device=device,
                model_path=self._weights_path,
                confidence_threshold=self._confidence_threshold,
                window_stride=self._window_stride,
            )
            logger.info(
                "ScratchCNNAdapter loaded from %s on %s",
                self._weights_path,
                device,
            )
        except Exception as exc:
            logger.error("ScratchCNNAdapter.load() failed: %s", exc)
            raise RuntimeError(f"Cannot load scratch_cnn: {exc}") from exc

    def unload(self) -> None:
        self._detector = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("ScratchCNNAdapter unloaded")

    def is_loaded(self) -> bool:
        return self._detector is not None

    # ── inference ─────────────────────────────────────────────────────────
    def predict(self, frame: Any) -> Tuple[List[Dict], float]:
        if self._detector is None:
            raise RuntimeError("scratch_cnn is not loaded — call load() first")

        t0 = time.perf_counter()
        raw: List = self._detector.detect(frame)  # (x, y, w, h, conf)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        for det in raw:
            x, y, w, h, conf = det
            detections.append(
                {
                    "x1": float(x),
                    "y1": float(y),
                    "x2": float(x + w),
                    "y2": float(y + h),
                    "confidence": float(conf),
                    "class": "face",
                }
            )

        return detections, elapsed_ms
