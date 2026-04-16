"""
Training Service — subprocess-based.

Manages training job lifecycle: submit, run (via real subprocess), monitor, cancel.

Architecture:
  - Each job launches the ML training script as a subprocess (subprocess.Popen).
  - A background thread reads stdout/stderr line-by-line (non-blocking per-line).
  - Each line is:
      1. Stored in a per-job ring-buffer (latest MAX_JOB_LOGS lines).
      2. Parsed for epoch metrics via regex → EpochResult.
      3. Broadcast to WS channel "train-logs" via _log_callback.
      4. Epoch metrics additionally broadcast via _progress_callback.
  - Cancellation: sets cancel event + SIGTERM → SIGKILL on timeout.
  - Safe to call cancel() from any thread/coroutine.
  - Prevents multiple concurrent jobs (enforced by callers via get_active_job()).

Real training script location (relative to ml_engine_root):
    scripts/train_scratch_cnn.py

Log line format emitted by the script:
    "INFO Epoch [  1/50] | Train Loss: 0.5431, Acc: 0.7234 | Val Loss: 0.6123, Acc: 0.6987"
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from app.utils.metrics_collector import metrics_collector

logger = logging.getLogger(__name__)

# ── Regex to parse epoch log lines ───────────────────────────────────────────
# Matches training script output:
#   "INFO Epoch [  1/50] | Train Loss: 0.5431, Acc: 0.7234 | Val Loss: 0.6123, Acc: 0.6987"
_EPOCH_RE = re.compile(
    r"Epoch\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]"
    r".*?Train Loss:\s*([\d.]+),\s*Acc:\s*([\d.]+)"
    r".*?Val Loss:\s*([\d.]+),\s*Acc:\s*([\d.]+)",
    re.IGNORECASE,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_JOB_LOGS = 2000       # max log lines kept in per-job ring buffer
KILL_TIMEOUT_SEC = 5      # seconds to wait after SIGTERM before SIGKILL


# ── Data models ───────────────────────────────────────────────────────────────

class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    model_name: str
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    checkpoint_every_n_epochs: int = 1
    data_dir: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    lr: float
    duration_sec: float

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingJob:
    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    current_epoch: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    epoch_results: List[EpochResult] = field(default_factory=list)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Internal — not exposed via as_dict()
    _cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    _process: Optional[subprocess.Popen] = field(default=None, repr=False)  # type: ignore[type-arg]
    _log_buffer: Deque[str] = field(
        default_factory=lambda: deque(maxlen=MAX_JOB_LOGS), repr=False
    )

    # ── Cancellation ─────────────────────────────────────────────────────────

    def request_cancel(self) -> None:
        """Set the cancel flag and terminate the subprocess if running."""
        self._cancel_event.set()
        proc = self._process
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass

    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    # ── Logs ─────────────────────────────────────────────────────────────────

    def get_logs(self) -> List[str]:
        return list(self._log_buffer)

    # ── Serialisation ────────────────────────────────────────────────────────

    def as_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "model_name": self.config.model_name,
            "status": self.status.value,
            "current_epoch": self.current_epoch,
            "total_epochs": self.config.epochs,
            "progress_pct": (
                round(self.current_epoch / self.config.epochs * 100, 1)
                if self.config.epochs > 0
                else 0
            ),
            "best_val_loss": (
                self.best_val_loss if self.best_val_loss != float("inf") else None
            ),
            "best_epoch": self.best_epoch,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat(),
            "epoch_history": [r.as_dict() for r in self.epoch_results],
        }


# ── Callback types ────────────────────────────────────────────────────────────

# called from the async event loop; both functions are sync (use ensure_future inside)
ProgressCallback = Callable[[TrainingJob, Optional[EpochResult]], None]
LogCallback = Callable[[TrainingJob, str], None]


# ── Service ───────────────────────────────────────────────────────────────────

class TrainingService:
    """
    Manages training jobs.  Each job:
      1. Spawns the real ML training script via subprocess.Popen.
      2. Reads stdout line-by-line in the training thread.
      3. Parses epoch metrics; broadcasts progress + raw log lines via callbacks.
      4. Supports cancellation via SIGTERM → SIGKILL.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.RLock()
        self._progress_callback: Optional[ProgressCallback] = None
        self._log_callback: Optional[LogCallback] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_progress_callback(self, cb: ProgressCallback) -> None:
        self._progress_callback = cb

    def set_log_callback(self, cb: LogCallback) -> None:
        self._log_callback = cb

    # ── Job management ────────────────────────────────────────────────────────

    def submit(self, config: TrainingConfig) -> TrainingJob:
        job_id = str(uuid.uuid4())
        job = TrainingJob(job_id=job_id, config=config)
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(
            target=self._train_thread,
            args=(job,),
            name=f"train-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        logger.info(
            "Training job submitted | id=%s | model=%s | epochs=%d",
            job_id,
            config.model_name,
            config.epochs,
        )
        return job

    def cancel(self, job_id: str) -> bool:
        """Cancel a specific job. Returns True if cancellation was initiated."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job and job.status == TrainingStatus.RUNNING:
            job.request_cancel()
            logger.info("Cancel requested for job %s", job_id)
            return True
        return False

    def cancel_active(self) -> Optional[str]:
        """Cancel whatever job is currently running. Returns the job_id or None."""
        with self._lock:
            for job in self._jobs.values():
                if job.status == TrainingStatus.RUNNING:
                    job.request_cancel()
                    logger.info("Active job %s cancelled via stop endpoint", job.job_id)
                    return job.job_id
        return None

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_job_logs(self, job_id: str) -> List[str]:
        with self._lock:
            job = self._jobs.get(job_id)
        return job.get_logs() if job else []

    def list_jobs(self, status_filter: Optional[TrainingStatus] = None) -> List[Dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.as_dict() for j in jobs]

    def get_active_job(self) -> Optional[TrainingJob]:
        with self._lock:
            for job in self._jobs.values():
                if job.status in (TrainingStatus.RUNNING, TrainingStatus.PENDING):
                    return job
        return None

    # ── Training thread ───────────────────────────────────────────────────────

    def _train_thread(self, job: TrainingJob) -> None:
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.utcnow()
        self._emit_progress(job, None)

        try:
            self._run_subprocess(job)
        except Exception as exc:
            job.status = TrainingStatus.FAILED
            job.error_message = str(exc)
            logger.exception("Job %s failed: %s", job.job_id, exc)
        finally:
            # Only set terminal status if not already set by _run_subprocess
            if job.status == TrainingStatus.RUNNING:
                job.status = (
                    TrainingStatus.CANCELLED
                    if job.is_cancel_requested()
                    else TrainingStatus.COMPLETED
                )
            job.completed_at = datetime.utcnow()
            job._process = None
            self._emit_progress(job, None)
            logger.info(
                "Job %s finished | status=%s | epoch=%d",
                job.job_id,
                job.status.value,
                job.current_epoch,
            )

    def _run_subprocess(self, job: TrainingJob) -> None:
        """Launch the training script and stream its output."""
        # Late import to avoid circular dependency at module load
        from app.config import get_settings  # noqa: PLC0415

        settings = get_settings()
        config = job.config

        script_path = settings.ml_engine_root / "scripts" / "train_scratch_cnn.py"
        if not script_path.exists():
            raise FileNotFoundError(
                f"Training script not found at {script_path}. "
                "Check ml_engine_root in config."
            )

        data_dir = Path(config.data_dir) if config.data_dir else settings.data_dir
        output_dir = settings.models_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            sys.executable,
            str(script_path),
            "--epochs",        str(config.epochs),
            "--batch_size",    str(config.batch_size),
            "--lr",            str(config.learning_rate),
            "--weight_decay",  str(config.weight_decay),
            "--data_dir",      str(data_dir),
            "--output_dir",    str(output_dir),
        ]

        logger.info("Spawning training subprocess: %s", " ".join(cmd))

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr → stdout for unified stream
            text=True,
            bufsize=1,                  # line-buffered
            encoding="utf-8",
            errors="replace",
            cwd=str(settings.ml_engine_root),
        )
        job._process = process

        # ── Non-blocking line reader ──────────────────────────────────────────
        # stdout iteration blocks per-line (that IS the non-blocking approach
        # for buffered text streams — each readline() returns as soon as a
        # full line is available without busy-waiting).
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            self._handle_output_line(job, line)
            if job.is_cancel_requested():
                break

        # ── Termination handling ──────────────────────────────────────────────
        if job.is_cancel_requested():
            _terminate_process(process)
            job.status = TrainingStatus.CANCELLED
        else:
            exit_code = process.wait()
            if exit_code != 0:
                raise RuntimeError(
                    f"Training script exited with non-zero code {exit_code}"
                )
            job.status = TrainingStatus.COMPLETED

    # ── Per-line processing ───────────────────────────────────────────────────

    def _handle_output_line(self, job: TrainingJob, line: str) -> None:
        """
        Store line, optionally parse epoch metrics, schedule WS broadcasts.
        Called only from the training thread.
        """
        job._log_buffer.append(line)

        # Try to parse structured epoch metrics
        m = _EPOCH_RE.search(line)
        if m:
            epoch      = int(m.group(1))
            total      = int(m.group(2))
            train_loss = float(m.group(3))
            train_acc  = float(m.group(4))
            val_loss   = float(m.group(5))
            val_acc    = float(m.group(6))

            # Sync config.epochs with what the script is actually doing
            job.config.epochs = total
            job.current_epoch = epoch

            epoch_result = EpochResult(
                epoch=epoch,
                train_loss=round(train_loss, 6),
                val_loss=round(val_loss, 6),
                train_acc=round(train_acc, 4),
                val_acc=round(val_acc, 4),
                lr=round(job.config.learning_rate * (0.95 ** epoch), 8),
                duration_sec=0.0,
            )
            job.epoch_results.append(epoch_result)

            if val_loss < job.best_val_loss:
                job.best_val_loss = val_loss
                job.best_epoch = epoch

            try:
                metrics_collector.record_training_epoch(
                    model_name=job.config.model_name,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    lr=epoch_result.lr,
                    duration_sec=0.0,
                )
            except Exception:
                pass

            self._emit_progress(job, epoch_result)

        # Always broadcast the raw line to log subscribers
        self._emit_log_line(job, line)

    # ── Async emission helpers ────────────────────────────────────────────────

    def _emit_progress(self, job: TrainingJob, epoch_result: Optional[EpochResult]) -> None:
        if self._progress_callback and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_emit_progress(job, epoch_result), self._loop
            )

    def _emit_log_line(self, job: TrainingJob, line: str) -> None:
        if self._log_callback and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_emit_log(job, line), self._loop
            )

    async def _async_emit_progress(
        self, job: TrainingJob, epoch_result: Optional[EpochResult]
    ) -> None:
        try:
            if self._progress_callback:
                self._progress_callback(job, epoch_result)
        except Exception as exc:
            logger.debug("Progress callback error: %s", exc)

    async def _async_emit_log(self, job: TrainingJob, line: str) -> None:
        try:
            if self._log_callback:
                self._log_callback(job, line)
        except Exception as exc:
            logger.debug("Log callback error: %s", exc)


# ── Subprocess termination helper ─────────────────────────────────────────────

def _terminate_process(process: subprocess.Popen) -> None:  # type: ignore[type-arg]
    """SIGTERM → wait → SIGKILL if still alive."""
    try:
        process.terminate()
        try:
            process.wait(timeout=KILL_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            logger.warning("Process did not exit after SIGTERM; sending SIGKILL")
            process.kill()
            process.wait()
    except OSError as exc:
        logger.debug("terminate_process OSError: %s", exc)


# ── Module-level singleton ────────────────────────────────────────────────────

training_service = TrainingService()
