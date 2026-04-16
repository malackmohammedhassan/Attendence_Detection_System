"""
Metrics routes.

GET  /metrics/system            — Live CPU, memory, FPS snapshot
GET  /metrics/system/history    — Rolling performance history
GET  /metrics/inference         — Summary stats across all models
GET  /metrics/inference/{model} — Per-model inference stats
GET  /metrics/inference/{model}/history — Raw inference records
GET  /metrics/training/{model}  — Training epoch history
GET  /metrics/training/{model}/best — Best epoch for a model
GET  /metrics/summary           — Combined dashboard summary
POST /metrics/reset             — Clear metrics (dev / test use)
WS   /metrics/live              — Push metrics over WebSocket at interval
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from app.utils.metrics_collector import metrics_collector
from app.utils.performance_tracker import performance_tracker
from app.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


# ────────────────────────────────────────────── System / performance ─ #

@router.get("/system", response_model=Dict[str, Any])
async def get_system_metrics() -> Dict[str, Any]:
    """Return the latest real-time system performance snapshot."""
    snap = performance_tracker.snapshot()
    if snap is None:
        return {"message": "No metrics collected yet. Tracker may still be initializing."}
    return snap.as_dict()


@router.get("/system/history", response_model=List[Dict[str, Any]])
async def get_system_history(
    n: int = Query(default=60, ge=1, le=3600, description="Number of recent samples"),
) -> List[Dict[str, Any]]:
    """Return recent performance snapshots (newest first)."""
    return performance_tracker.history_as_dicts(n=n)


# ────────────────────────────────────────────── Inference metrics ─── #

@router.get("/inference", response_model=Dict[str, Any])
async def get_all_inference_stats() -> Dict[str, Any]:
    """Aggregate inference stats for every tracked model."""
    return metrics_collector.get_all_inference_stats()


@router.get("/inference/{model_name}", response_model=Optional[Dict[str, Any]])
async def get_inference_stats(model_name: str) -> Optional[Dict[str, Any]]:
    """Latency percentiles, FPS, and accuracy stats for one model."""
    stats = metrics_collector.get_inference_stats(model_name)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No inference data for model '{model_name}'.",
        )
    return stats.as_dict()


@router.get(
    "/inference/{model_name}/history",
    response_model=List[Dict[str, Any]],
)
async def get_inference_history(
    model_name: str,
    limit: int = Query(default=100, ge=1, le=1000),
) -> List[Dict[str, Any]]:
    """Return individual inference records for a model."""
    return metrics_collector.get_inference_records(model_name=model_name, limit=limit)


# ────────────────────────────────────────────── Training metrics ─── #

@router.get(
    "/training/{model_name}",
    response_model=List[Dict[str, Any]],
)
async def get_training_history(
    model_name: str,
    last_n: Optional[int] = Query(default=None, ge=1, description="Limit to last N epochs"),
) -> List[Dict[str, Any]]:
    """Return epoch-by-epoch training history for a model."""
    history = metrics_collector.get_training_history(model_name, last_n=last_n)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training history for model '{model_name}'.",
        )
    return history


@router.get("/training/{model_name}/best", response_model=Optional[Dict[str, Any]])
async def get_best_epoch(model_name: str) -> Optional[Dict[str, Any]]:
    """Return the epoch with the lowest validation loss."""
    best = metrics_collector.get_best_epoch(model_name)
    if best is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training data for model '{model_name}'.",
        )
    return best


# ────────────────────────────────────────────── Summary ───────────── #

@router.get("/summary", response_model=Dict[str, Any])
async def get_metrics_summary() -> Dict[str, Any]:
    """Combined high-level summary for the dashboard overview panel."""
    snap = performance_tracker.snapshot()
    return {
        "system": snap.as_dict() if snap else None,
        "collector": metrics_collector.summary(),
        "inference": metrics_collector.get_all_inference_stats(),
    }


# ────────────────────────────────────────────── Reset (dev) ───────── #

class ResetRequest(BaseModel):
    model_name: Optional[str] = Field(
        default=None,
        description="If set, reset only this model. If null, reset all.",
    )
    confirm: bool = Field(
        ...,
        description="Must be true to proceed.",
    )


@router.post("/reset", status_code=status.HTTP_200_OK)
async def reset_metrics(request: ResetRequest) -> Dict[str, str]:
    """Clear collected metrics. Requires explicit confirmation flag."""
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Set confirm=true to reset metrics.",
        )
    metrics_collector.reset(model_name=request.model_name)
    scope = request.model_name or "all models"
    return {"message": f"Metrics reset for {scope}."}


# ────────────────────────────────────────────── WebSocket live feed ─ #

@router.websocket("/live")
async def metrics_live_ws(
    websocket: WebSocket,
    interval: float = 1.0,
) -> None:
    """
    Push live metrics to connected client at the given interval (seconds).
    Client can send {"interval": 2.0} to change the push rate dynamically.
    """
    conn = await ws_manager.connect(websocket, channel="metrics")
    await conn.send_json({"type": "connected", "interval_sec": interval})

    try:
        while True:
            snap = performance_tracker.snapshot()
            payload = {
                "type": "metrics_tick",
                "system": snap.as_dict() if snap else None,
                "collector_summary": metrics_collector.summary(),
            }
            await conn.send_json(payload)

            # Non-blocking wait — check for incoming control messages
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=interval)
                if "interval" in data:
                    interval = max(0.1, float(data["interval"]))
                    await conn.send_json({"type": "ack", "interval_sec": interval})
            except asyncio.TimeoutError:
                pass  # Normal — no message received this tick

    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(conn)
