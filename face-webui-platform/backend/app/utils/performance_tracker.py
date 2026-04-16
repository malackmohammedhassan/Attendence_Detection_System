"""
Real-time performance tracker using psutil.

Tracks CPU, memory, and rolling FPS in a background daemon thread.
Thread-safe via a reentrant lock.  Consumers call snapshot() to get
the latest readings without triggering any I/O themselves.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    timestamp: datetime
    cpu_percent: float          # overall CPU %
    cpu_per_core: List[float]   # per-core CPU %
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    swap_used_mb: float
    swap_total_mb: float
    process_cpu_percent: float  # this process only
    process_memory_mb: float    # RSS of this process
    fps: float                  # rolling average FPS
    frame_count: int            # cumulative frames processed
    uptime_sec: float

    def as_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu": {
                "overall_percent": self.cpu_percent,
                "per_core": self.cpu_per_core,
                "process_percent": self.process_cpu_percent,
            },
            "memory": {
                "used_mb": self.memory_used_mb,
                "total_mb": self.memory_total_mb,
                "percent": self.memory_percent,
                "swap_used_mb": self.swap_used_mb,
                "swap_total_mb": self.swap_total_mb,
                "process_rss_mb": self.process_memory_mb,
            },
            "inference": {
                "fps": round(self.fps, 2),
                "total_frames": self.frame_count,
            },
            "uptime_sec": round(self.uptime_sec, 1),
        }


class PerformanceTracker:
    """
    Background thread that polls system metrics at a configurable interval.

    Usage:
        tracker = PerformanceTracker(sample_interval=1.0, window=60)
        tracker.start()
        snap = tracker.snapshot()
        tracker.stop()
    """

    def __init__(
        self,
        sample_interval: float = 1.0,
        window: int = 60,
    ) -> None:
        self._interval = sample_interval
        self._window = window

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._snapshots: Deque[PerformanceSnapshot] = deque(maxlen=window)
        self._latest: Optional[PerformanceSnapshot] = None

        # FPS tracking
        self._frame_timestamps: Deque[float] = deque(maxlen=120)
        self._total_frames: int = 0

        self._start_time: float = time.monotonic()
        self._process = psutil.Process()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="perf-tracker",
            daemon=True,
        )
        self._thread.start()
        logger.info("PerformanceTracker started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval * 2)
        logger.info("PerformanceTracker stopped")

    def record_frame(self) -> None:
        """Call this each time an inference frame is processed."""
        now = time.monotonic()
        with self._lock:
            self._frame_timestamps.append(now)
            self._total_frames += 1

    def snapshot(self) -> Optional[PerformanceSnapshot]:
        """Return the most recently collected snapshot (thread-safe)."""
        with self._lock:
            return self._latest

    def history(self, n: Optional[int] = None) -> List[PerformanceSnapshot]:
        """Return up to `n` recent snapshots (newest first)."""
        with self._lock:
            snaps = list(self._snapshots)
        snaps.reverse()
        return snaps[:n] if n else snaps

    def history_as_dicts(self, n: Optional[int] = None) -> List[Dict]:
        return [s.as_dict() for s in self.history(n)]

    def rolling_fps(self) -> float:
        """Compute FPS from the recent frame timestamp ring buffer."""
        with self._lock:
            frames = list(self._frame_timestamps)
        if len(frames) < 2:
            return 0.0
        elapsed = frames[-1] - frames[0]
        if elapsed <= 0:
            return 0.0
        return (len(frames) - 1) / elapsed

    # ------------------------------------------------------------------ #
    #  Internal polling loop
    # ------------------------------------------------------------------ #

    def _poll_loop(self) -> None:
        # Warm up psutil's CPU measurement (first call is always 0.0)
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        while self._running:
            try:
                snap = self._collect()
                with self._lock:
                    self._snapshots.append(snap)
                    self._latest = snap
            except Exception as exc:
                logger.warning("PerformanceTracker poll error: %s", exc)
            time.sleep(self._interval)

    def _collect(self) -> PerformanceSnapshot:
        cpu_pct = psutil.cpu_percent(interval=None)
        cpu_cores = psutil.cpu_percent(interval=None, percpu=True)

        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        proc_cpu = self._process.cpu_percent(interval=None)
        proc_mem = self._process.memory_info().rss / 1024 / 1024

        fps = self.rolling_fps()
        with self._lock:
            total_frames = self._total_frames

        return PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_pct,
            cpu_per_core=cpu_cores,
            memory_used_mb=mem.used / 1024 / 1024,
            memory_total_mb=mem.total / 1024 / 1024,
            memory_percent=mem.percent,
            swap_used_mb=swap.used / 1024 / 1024,
            swap_total_mb=swap.total / 1024 / 1024,
            process_cpu_percent=proc_cpu,
            process_memory_mb=proc_mem,
            fps=fps,
            frame_count=total_frames,
            uptime_sec=time.monotonic() - self._start_time,
        )


# Module-level singleton
performance_tracker = PerformanceTracker()
