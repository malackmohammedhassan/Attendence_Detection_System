"""
Training routes.

Existing endpoints (unchanged):
  POST   /train/start          — Submit a new training job (full config)
  POST   /train/{job_id}/cancel — Cancel a running job by ID
  GET    /train/{job_id}        — Get job state + epoch history
  GET    /train/               — List all jobs (optional ?status=... filter)
  GET    /train/active          — Currently running job (if any)
  WS     /train/ws/{job_id}     — Real-time progress stream for a specific job

New endpoints (Phase 4):
  POST   /train                — Convenience start (same body as /start)
  POST   /train/stop           — Stop whichever job is currently running
  GET    /train/status         — Quick training status summary (no job_id needed)
  WS     /train/ws/logs        — Stream raw stdout/stderr lines for the active job
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, field_validator

from app.services.model_manager import model_manager
from app.services.training_service import (
    EpochResult,
    TrainingConfig,
    TrainingJob,
    TrainingStatus,
    training_service,
)
from app.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["training"])

# WS channel names
_CHANNEL_TRAINING   = "training"
_CHANNEL_TRAIN_LOGS = "train-logs"


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TrainingStartRequest(BaseModel):
    model_name: str = Field(..., description="Registered model identifier")
    epochs: int = Field(default=10, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0, le=1.0)
    optimizer: str = Field(default="adam")
    weight_decay: float = Field(default=1e-4, ge=0)
    early_stopping_patience: int = Field(default=5, ge=0, le=50)
    checkpoint_every_n_epochs: int = Field(default=1, ge=1)
    data_dir: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("optimizer")
    @classmethod
    def _validate_optimizer(cls, v: str) -> str:
        allowed = {"adam", "sgd", "adamw", "rmsprop"}
        if v.lower() not in allowed:
            raise ValueError(f"optimizer must be one of {allowed}")
        return v.lower()

    @field_validator("model_name")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        available = model_manager.available_names()
        if available and v not in available:
            raise ValueError(f"Model '{v}' not registered. Available: {available}")
        return v


class TrainingResponse(BaseModel):
    job_id: str
    model_name: str
    status: str
    message: str


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_config(request: TrainingStartRequest) -> TrainingConfig:
    return TrainingConfig(
        model_name=request.model_name,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        optimizer=request.optimizer,
        weight_decay=request.weight_decay,
        early_stopping_patience=request.early_stopping_patience,
        checkpoint_every_n_epochs=request.checkpoint_every_n_epochs,
        data_dir=request.data_dir,
        extra=request.extra,
    )


async def _submit_job(request: TrainingStartRequest) -> TrainingJob:
    """Shared logic for both POST /train and POST /train/start."""
    active = training_service.get_active_job()
    if active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A training job is already running: {active.job_id}",
        )
    loop = asyncio.get_event_loop()
    training_service.set_event_loop(loop)
    training_service.set_progress_callback(_broadcast_progress)
    training_service.set_log_callback(_broadcast_log_line)

    return training_service.submit(_make_config(request))


# ── Phase 4: POST /train  (convenience start, no /start suffix) ──────────────
@router.post(
    "",
    response_model=TrainingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start training job",
)
async def start_training_root(request: TrainingStartRequest) -> TrainingResponse:
    """Start a training job.  Alias for POST /train/start."""
    job = await _submit_job(request)
    return TrainingResponse(
        job_id=job.job_id,
        model_name=job.config.model_name,
        status=job.status.value,
        message=f"Training job {job.job_id} submitted.",
    )


# ── Phase 4: POST /train/stop ─────────────────────────────────────────────────
@router.post("/stop", status_code=status.HTTP_200_OK, summary="Stop active training job")
async def stop_training() -> Dict[str, Any]:
    """Cancel whichever job is currently running.  No-op if nothing is running."""
    job_id = training_service.cancel_active()
    if job_id is None:
        return {"message": "No active training job.", "job_id": None}
    return {"message": f"Cancellation requested for {job_id}.", "job_id": job_id}


# ── Phase 4: GET /train/status ────────────────────────────────────────────────
@router.get("/status", response_model=Dict[str, Any], summary="Training status summary")
async def training_status_summary() -> Dict[str, Any]:
    """Quick status object — no job_id required."""
    active = training_service.get_active_job()
    total_jobs = len(training_service.list_jobs())
    return {
        "active": active.as_dict() if active else None,
        "is_running": active is not None and active.status == TrainingStatus.RUNNING,
        "total_jobs": total_jobs,
    }


# ── POST /train/start  (original full endpoint) ───────────────────────────────
@router.post(
    "/start",
    response_model=TrainingResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_training(request: TrainingStartRequest) -> TrainingResponse:
    """Submit a training job.  Returns immediately with a job_id."""
    job = await _submit_job(request)
    return TrainingResponse(
        job_id=job.job_id,
        model_name=job.config.model_name,
        status=job.status.value,
        message=f"Training job {job.job_id} submitted successfully.",
    )


# ── POST /train/{job_id}/cancel ───────────────────────────────────────────────
@router.post("/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_training(job_id: str) -> Dict[str, Any]:
    """Request cancellation of a running training job."""
    cancelled = training_service.cancel(job_id)
    if not cancelled:
        job = training_service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' is not running (status={job.status.value}).",
        )
    return {"job_id": job_id, "message": "Cancellation requested."}


# ── GET /train/active ─────────────────────────────────────────────────────────
@router.get("/active", response_model=Optional[Dict[str, Any]])
async def get_active_job() -> Optional[Dict[str, Any]]:
    job = training_service.get_active_job()
    return job.as_dict() if job else None


# ── GET /train/ ───────────────────────────────────────────────────────────────
@router.get("/", response_model=List[Dict[str, Any]])
async def list_jobs(
    status_filter: Optional[str] = Query(default=None, alias="status"),
) -> List[Dict[str, Any]]:
    ts_filter: Optional[TrainingStatus] = None
    if status_filter:
        try:
            ts_filter = TrainingStatus(status_filter)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status '{status_filter}'.")
    return training_service.list_jobs(status_filter=ts_filter)


# ── GET /train/{job_id} ───────────────────────────────────────────────────────
@router.get("/{job_id}", response_model=Dict[str, Any])
async def get_job(job_id: str) -> Dict[str, Any]:
    job = training_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.as_dict()


# ── WS /train/ws/logs  — Phase 4 ─────────────────────────────────────────────
@router.websocket("/ws/logs")
async def training_logs_ws(websocket: WebSocket) -> None:
    """
    Stream raw stdout/stderr lines from the running training process.

    On connect:
      • Immediately sends all buffered lines as { type: 'train_log_batch', job_id, lines }.
      • Stays open receiving { type: 'train_log', job_id, line, ts } messages
        broadcast by _broadcast_log_line() in real-time.
    """
    conn = await ws_manager.connect(websocket, channel=_CHANNEL_TRAIN_LOGS)

    # Replay buffered lines for the active job
    active = training_service.get_active_job()
    if active is None:
        all_jobs = training_service.list_jobs()
        job_id_hint: Optional[str] = all_jobs[0]["job_id"] if all_jobs else None
    else:
        job_id_hint = active.job_id

    if job_id_hint:
        existing = training_service.get_job_logs(job_id_hint)
        if existing:
            try:
                await conn.send_json(
                    {"type": "train_log_batch", "job_id": job_id_hint, "lines": existing}
                )
            except Exception:
                await ws_manager.disconnect(conn)
                return

    try:
        while True:
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(conn)


# ── WS /train/ws/{job_id}  — existing ────────────────────────────────────────
@router.websocket("/ws/{job_id}")
async def training_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream real-time epoch progress for a specific job."""
    conn = await ws_manager.connect(websocket, channel=f"train-{job_id}")
    job = training_service.get_job(job_id)

    if job:
        await conn.send_json({"type": "job_state", **job.as_dict()})
    else:
        await conn.send_json({"type": "error", "message": f"Job '{job_id}' not found."})
        await ws_manager.disconnect(conn)
        return

    try:
        while True:
            await asyncio.sleep(1.0)
            j = training_service.get_job(job_id)
            if j and j.status not in (TrainingStatus.RUNNING, TrainingStatus.PENDING):
                await conn.send_json({"type": "job_complete", **j.as_dict()})
                break
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(conn)


# ── Broadcast callbacks ───────────────────────────────────────────────────────

def _broadcast_progress(job: TrainingJob, epoch_result: Optional[EpochResult]) -> None:
    """Sync callback — called from the async event loop via run_coroutine_threadsafe."""
    payload: Dict[str, Any] = {
        "type": "epoch" if epoch_result else "status_change",
        "job_id": job.job_id,
        "model_name": job.config.model_name,
        "status": job.status.value,
        "current_epoch": job.current_epoch,
        "total_epochs": job.config.epochs,
        "progress_pct": round(
            job.current_epoch / job.config.epochs * 100, 1
        ) if job.config.epochs else 0,
    }
    if epoch_result:
        payload["epoch_result"] = epoch_result.as_dict()

    asyncio.ensure_future(ws_manager.broadcast(payload, channel=f"train-{job.job_id}"))
    asyncio.ensure_future(ws_manager.broadcast(payload, channel=_CHANNEL_TRAINING))


def _broadcast_log_line(job: TrainingJob, line: str) -> None:
    """Sync callback — called from the async event loop via run_coroutine_threadsafe."""
    import time as _time  # noqa: PLC0415

    payload: Dict[str, Any] = {
        "type": "train_log",
        "job_id": job.job_id,
        "line": line,
        "ts": _time.time(),
    }
    asyncio.ensure_future(ws_manager.broadcast(payload, channel=_CHANNEL_TRAIN_LOGS))
