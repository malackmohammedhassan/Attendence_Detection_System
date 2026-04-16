"""
Metrics collector — aggregates and stores inference & training metrics
for querying by the dashboard routes.

Designed to be used as a singleton; thread-safe.
"""

from __future__ import annotations

import statistics
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class InferenceRecord:
    """One inference pass result."""
    model_name: str
    latency_ms: float
    confidence: float
    num_detections: int
    image_width: int
    image_height: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class TrainingRecord:
    """One epoch result."""
    model_name: str
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    lr: float
    duration_sec: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class AggregateInferenceStats:
    model_name: str
    sample_count: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    mean_confidence: float
    mean_detections: float
    fps_estimate: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """
    In-memory store for inference and training metrics.

    Maintains rolling buffers per model.  All public methods are thread-safe.
    """

    def __init__(self, max_inference_records: int = 1000) -> None:
        self._max_records = max_inference_records
        self._lock = threading.RLock()

        # model_name -> deque of InferenceRecord
        self._inference: Dict[str, Deque[InferenceRecord]] = defaultdict(
            lambda: deque(maxlen=self._max_records)
        )
        # model_name -> list of TrainingRecord  (keep all epochs)
        self._training: Dict[str, List[TrainingRecord]] = defaultdict(list)

        # lightweight counters
        self._total_inferences: int = 0
        self._error_count: int = 0

    # ------------------------------------------------------------------ #
    #  Ingestion
    # ------------------------------------------------------------------ #

    def record_inference(
        self,
        model_name: str,
        latency_ms: float,
        confidence: float,
        num_detections: int,
        image_width: int = 0,
        image_height: int = 0,
    ) -> None:
        rec = InferenceRecord(
            model_name=model_name,
            latency_ms=latency_ms,
            confidence=confidence,
            num_detections=num_detections,
            image_width=image_width,
            image_height=image_height,
        )
        with self._lock:
            self._inference[model_name].append(rec)
            self._total_inferences += 1

    def record_training_epoch(
        self,
        model_name: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        duration_sec: float,
    ) -> None:
        rec = TrainingRecord(
            model_name=model_name,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            lr=lr,
            duration_sec=duration_sec,
        )
        with self._lock:
            self._training[model_name].append(rec)

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    # ------------------------------------------------------------------ #
    #  Queries — inference
    # ------------------------------------------------------------------ #

    def get_inference_records(
        self,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if model_name:
                records = list(self._inference.get(model_name, []))
            else:
                records = [
                    r for dq in self._inference.values() for r in dq
                ]
                records.sort(key=lambda r: r.timestamp, reverse=True)
        return [r.as_dict() for r in records[:limit]]

    def get_inference_stats(
        self, model_name: str
    ) -> Optional[AggregateInferenceStats]:
        with self._lock:
            records = list(self._inference.get(model_name, []))
        if not records:
            return None

        latencies = sorted(r.latency_ms for r in records)
        n = len(latencies)

        def percentile(data: List[float], pct: float) -> float:
            idx = int(len(data) * pct / 100)
            return data[min(idx, len(data) - 1)]

        mean_lat = statistics.mean(latencies)
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        return AggregateInferenceStats(
            model_name=model_name,
            sample_count=n,
            mean_latency_ms=round(mean_lat, 3),
            p50_latency_ms=round(percentile(latencies, 50), 3),
            p95_latency_ms=round(percentile(latencies, 95), 3),
            p99_latency_ms=round(percentile(latencies, 99), 3),
            max_latency_ms=round(max(latencies), 3),
            min_latency_ms=round(min(latencies), 3),
            mean_confidence=round(statistics.mean(r.confidence for r in records), 4),
            mean_detections=round(statistics.mean(r.num_detections for r in records), 2),
            fps_estimate=round(fps, 2),
        )

    def get_all_inference_stats(self) -> Dict[str, Any]:
        with self._lock:
            model_names = list(self._inference.keys())
        return {
            name: (s.as_dict() if (s := self.get_inference_stats(name)) else None)
            for name in model_names
        }

    # ------------------------------------------------------------------ #
    #  Queries — training
    # ------------------------------------------------------------------ #

    def get_training_history(
        self,
        model_name: str,
        last_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._training.get(model_name, []))
        if last_n:
            records = records[-last_n:]
        return [r.as_dict() for r in records]

    def get_best_epoch(self, model_name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            records = list(self._training.get(model_name, []))
        if not records:
            return None
        best = min(records, key=lambda r: r.val_loss)
        return best.as_dict()

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_inferences": self._total_inferences,
                "error_count": self._error_count,
                "models_tracked": list(self._inference.keys()),
                "training_runs": {
                    name: len(epochs)
                    for name, epochs in self._training.items()
                },
            }

    def reset(self, model_name: Optional[str] = None) -> None:
        """Clear records for one model or all models."""
        with self._lock:
            if model_name:
                self._inference[model_name].clear()
                self._training.pop(model_name, None)
            else:
                self._inference.clear()
                self._training.clear()
                self._total_inferences = 0
                self._error_count = 0


# Module-level singleton
metrics_collector = MetricsCollector()
