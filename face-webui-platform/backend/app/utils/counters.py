"""
Thread-safe observability counters for the dashboard backend.

Tracks:
  - Total frames received / dropped via the live WebSocket
  - Current and peak concurrent WS connections
  - Total benchmark runs initiated
  - Total uncaught server errors

Usage
─────
    from app.utils.counters import counters

    counters.inc_frames_received()
    counters.inc_ws_connected()
    snap = counters.as_dict()
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict


class MetricsCounters:
    """
    All fields are integers guarded by a single reentrant lock.

    Methods are deliberately fine-grained so callers don't hold the
    lock longer than a single field update.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at: float = time.time()

        # Frame pipeline
        self._frames_received:   int = 0
        self._frames_dropped:    int = 0
        self._infer_q_overflows: int = 0  # times the infer ring-buffer evicted a frame

        # WebSocket connections
        self._ws_current:        int = 0
        self._ws_peak:           int = 0
        self._ws_total_accepted: int = 0

        # Benchmark runs
        self._benchmark_runs:    int = 0

        # Errors — broken down by subsystem for targeted debugging
        self._ws_errors:         int = 0   # WebSocket handshake / send failures
        self._inference_errors:  int = 0   # per-frame model inference failures
        self._training_errors:   int = 0   # training job exceptions
        self._benchmark_errors:  int = 0   # benchmark engine exceptions

    # ------------------------------------------------------------------ #
    #  Increment helpers
    # ------------------------------------------------------------------ #

    def inc_frames_received(self, n: int = 1) -> None:
        with self._lock:
            self._frames_received += n

    def inc_frames_dropped(self, n: int = 1) -> None:
        with self._lock:
            self._frames_dropped += n

    def inc_inference_queue_overflow(self, n: int = 1) -> None:
        with self._lock:
            self._infer_q_overflows += n

    def inc_ws_connected(self) -> None:
        with self._lock:
            self._ws_current += 1
            self._ws_total_accepted += 1
            if self._ws_current > self._ws_peak:
                self._ws_peak = self._ws_current

    def dec_ws_connected(self) -> None:
        with self._lock:
            if self._ws_current > 0:
                self._ws_current -= 1

    def inc_benchmark_runs(self) -> None:
        with self._lock:
            self._benchmark_runs += 1

    def inc_errors(self) -> None:
        """Legacy catch-all — increments inference_errors for back-compat."""
        with self._lock:
            self._inference_errors += 1

    def inc_ws_errors(self, n: int = 1) -> None:
        with self._lock:
            self._ws_errors += n

    def inc_inference_errors(self, n: int = 1) -> None:
        with self._lock:
            self._inference_errors += n

    def inc_training_errors(self, n: int = 1) -> None:
        with self._lock:
            self._training_errors += n

    def inc_benchmark_errors(self, n: int = 1) -> None:
        with self._lock:
            self._benchmark_errors += n

    # ------------------------------------------------------------------ #
    #  Snapshot
    # ------------------------------------------------------------------ #

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            uptime_sec = time.time() - self._started_at
            return {
                "uptime_sec":                    round(uptime_sec, 1),
                "total_frames_received":         self._frames_received,
                "total_frames_dropped":          self._frames_dropped,
                "inference_queue_overflows":      self._infer_q_overflows,
                "frame_drop_rate":               (
                    round(self._frames_dropped / self._frames_received, 4)
                    if self._frames_received
                    else 0.0
                ),
                "ws_connections_current": self._ws_current,
                "ws_connections_peak":    self._ws_peak,
                "ws_connections_total":   self._ws_total_accepted,
                "total_benchmark_runs":   self._benchmark_runs,
                # Error counters — per-subsystem
                "errors": {
                    "ws":        self._ws_errors,
                    "inference": self._inference_errors,
                    "training":  self._training_errors,
                    "benchmark": self._benchmark_errors,
                    "total":     self._ws_errors + self._inference_errors + self._training_errors + self._benchmark_errors,
                },
            }

    def reset(self) -> None:
        """Reset all counters (useful for tests)."""
        with self._lock:
            self._frames_received   = 0
            self._frames_dropped    = 0
            self._infer_q_overflows = 0
            self._ws_current        = 0
            self._ws_peak           = 0
            self._ws_total_accepted = 0
            self._benchmark_runs    = 0
            self._ws_errors         = 0
            self._inference_errors  = 0
            self._training_errors   = 0
            self._benchmark_errors  = 0
            self._started_at        = time.time()


# Module-level singleton
counters = MetricsCounters()
