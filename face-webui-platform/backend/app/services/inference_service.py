"""
Inference Service.

Accepts raw frames (bytes or numpy arrays), runs the active model,
measures precise latency, and returns structured detection results.
All heavy work stays off the event loop via asyncio.run_in_executor.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import get_settings
from app.services.model_manager import model_manager
from app.utils.metrics_collector import metrics_collector
from app.utils.performance_tracker import performance_tracker

logger = logging.getLogger(__name__)
settings = get_settings()

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "face"

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_dict(self) -> Dict:
        return {
            **asdict(self),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "area": round(self.area, 2),
        }


@dataclass
class InferenceResult:
    model_name: str
    detections: List[BoundingBox]
    latency_ms: float
    frame_width: int
    frame_height: int
    confidence_threshold: float
    raw_detection_count: int    # before threshold filtering
    timestamp: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "detections": [b.as_dict() for b in self.detections],
            "latency_ms": round(self.latency_ms, 3),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "detection_count": len(self.detections),
            "raw_detection_count": self.raw_detection_count,
            "confidence_threshold": self.confidence_threshold,
            "timestamp": self.timestamp,
        }


class InferenceService:
    """
    Stateless inference wrapper.

    Decodes input data → preprocesses → delegates to ModelManager →
    post-processes results → records metrics.
    """

    def __init__(self) -> None:
        self._conf_threshold = settings.inference_confidence_threshold
        self._target_w = settings.inference_input_width
        self._target_h = settings.inference_input_height

    # ------------------------------------------------------------------ #
    #  Public async API
    # ------------------------------------------------------------------ #

    async def infer_from_bytes(
        self,
        image_bytes: bytes,
        confidence_threshold: Optional[float] = None,
    ) -> InferenceResult:
        """Decode JPEG/PNG bytes and run inference (non-blocking)."""
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(
            _executor, self._decode_image, image_bytes
        )
        return await self._run_inference(frame, confidence_threshold)

    async def infer_from_array(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> InferenceResult:
        """Run inference on a pre-decoded numpy array."""
        return await self._run_inference(frame, confidence_threshold)

    async def infer_batch(
        self,
        frames: List[np.ndarray],
        confidence_threshold: Optional[float] = None,
    ) -> List[InferenceResult]:
        """Process multiple frames concurrently up to max_batch_size."""
        batch = frames[: settings.inference_max_batch_size]
        tasks = [self._run_inference(f, confidence_threshold) for f in batch]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------ #
    #  Core inference pipeline
    # ------------------------------------------------------------------ #

    async def _run_inference(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float],
    ) -> InferenceResult:
        threshold = confidence_threshold or self._conf_threshold
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            _executor,
            self._sync_infer,
            frame,
            threshold,
        )
        return result

    def _sync_infer(
        self, frame: np.ndarray, threshold: float
    ) -> InferenceResult:
        """Synchronous inference — runs in thread pool."""
        h, w = frame.shape[:2]
        active_name = model_manager.get_active_name()
        if active_name is None:
            raise RuntimeError("No active model. Set an active model before inference.")

        t0 = time.perf_counter()
        raw_detections, model_latency = model_manager.run_active(frame)
        wall_latency_ms = (time.perf_counter() - t0) * 1000

        # Parse and filter detections
        boxes: List[BoundingBox] = []
        for det in raw_detections:
            conf = float(det.get("confidence", 0.0))
            if conf >= threshold:
                boxes.append(
                    BoundingBox(
                        x1=float(det.get("x1", 0)),
                        y1=float(det.get("y1", 0)),
                        x2=float(det.get("x2", 0)),
                        y2=float(det.get("y2", 0)),
                        confidence=conf,
                        label=str(det.get("class", "face")),
                    )
                )

        result = InferenceResult(
            model_name=active_name,
            detections=boxes,
            latency_ms=model_latency or wall_latency_ms,
            frame_width=w,
            frame_height=h,
            confidence_threshold=threshold,
            raw_detection_count=len(raw_detections),
            timestamp=time.time(),
        )

        # Side-effects: metrics + FPS counter
        mean_conf = (
            sum(b.confidence for b in boxes) / len(boxes) if boxes else 0.0
        )
        metrics_collector.record_inference(
            model_name=active_name,
            latency_ms=result.latency_ms,
            confidence=mean_conf,
            num_detections=len(boxes),
            image_width=w,
            image_height=h,
        )
        performance_tracker.record_frame()

        return result

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        """Decode image bytes to a numpy BGR array using OpenCV."""
        try:
            import cv2
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("cv2.imdecode returned None — invalid image data")
            return frame
        except ImportError:
            # Fallback: PIL → numpy
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return np.array(img)[:, :, ::-1].copy()  # RGB → BGR

    def set_confidence_threshold(self, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self._conf_threshold = threshold


# Module-level singleton
inference_service = InferenceService()
