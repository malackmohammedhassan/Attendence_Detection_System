"""
MTCNN ModelAdapter — wraps the facenet-pytorch MTCNNDetector from
realtime-face-detection-dl/src/ and exposes it through the
ModelManager interface.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.services.model_manager import ModelAdapter

logger = logging.getLogger(__name__)
settings = get_settings()

_ML_SRC = settings.ml_engine_root / "src"


def _ensure_ml_src_on_path() -> None:
    src = str(_ML_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


class MTCNNAdapter(ModelAdapter):
    """
    Adapter wrapping MTCNNDetector from the ML engine src/.
    Converts (x, y, w, h, confidence) detections → {"x1","y1","x2","y2","confidence","class"}.
    """

    @property
    def name(self) -> str:
        return "mtcnn"

    @property
    def framework(self) -> str:
        return "PyTorch / facenet-pytorch"

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        self._confidence_threshold = confidence_threshold
        self._detector = None

    def load(self) -> None:
        _ensure_ml_src_on_path()
        try:
            import torch
            from detector import MTCNNDetector  # type: ignore[import-untyped]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._detector = MTCNNDetector(
                device=device,
                confidence_threshold=self._confidence_threshold,
            )
            logger.info("MTCNNAdapter loaded on %s", device)
        except Exception as exc:
            logger.error("MTCNNAdapter.load() failed: %s", exc)
            raise RuntimeError(f"Cannot load mtcnn: {exc}") from exc

    def unload(self) -> None:
        self._detector = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("MTCNNAdapter unloaded")

    def is_loaded(self) -> bool:
        return self._detector is not None

    def predict(self, frame: Any) -> Tuple[List[Dict], float]:
        if self._detector is None:
            raise RuntimeError("mtcnn is not loaded — call load() first")

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
