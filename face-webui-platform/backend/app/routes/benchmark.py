"""
Benchmark routes.

POST  /benchmark                       — Start a full 200-frame evaluation (fire-and-forget)
POST  /benchmark/run            — Run a latency benchmark for a single model
POST  /benchmark/compare        — Compare multiple models in one request
GET   /benchmark/results        — List all historical benchmark results
GET   /benchmark/results/{id}   — Get a specific benchmark result
GET   /benchmark/results/model/{name} — All results for one model
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.services.benchmark_engine import BenchmarkStatus, benchmark_engine, ComparativeResult
from app.services.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


# ────────────────────────────────────────────── Pydantic schemas ──── #

class BenchmarkRunRequest(BaseModel):
    model_name: str = Field(..., description="Model to benchmark")
    warmup_runs: Optional[int] = Field(default=None, ge=1, le=100)
    measure_runs: Optional[int] = Field(default=None, ge=5, le=1000)
    frame_width: int = Field(default=640, ge=32, le=4096)
    frame_height: int = Field(default=480, ge=32, le=4096)


class BenchmarkCompareRequest(BaseModel):
    model_names: List[str] = Field(..., min_length=2, description="Models to compare")
    warmup_runs: Optional[int] = Field(default=None, ge=1, le=100)
    measure_runs: Optional[int] = Field(default=None, ge=5, le=1000)
    frame_width: int = Field(default=640, ge=32, le=4096)
    frame_height: int = Field(default=480, ge=32, le=4096)


class FullBenchmarkRequest(BaseModel):
    model_name: str = Field(..., description="Model to benchmark")
    num_frames: int = Field(default=200, ge=10, le=2000, description="Number of evaluation frames")
    frame_width: int = Field(default=640, ge=32, le=4096)
    frame_height: int = Field(default=480, ge=32, le=4096)
    run_tag: Optional[str] = Field(default=None, max_length=100, description="Short label e.g. 'baseline'")
    run_notes: Optional[str] = Field(default=None, max_length=1000, description="Freeform notes for this run")


class ComparativeBenchmarkRequest(BaseModel):
    model_names: List[str] = Field(..., min_length=2, max_length=8, description="At least 2 models to compare")
    num_frames: int = Field(default=200, ge=10, le=2000)
    frame_width: int = Field(default=640, ge=32, le=4096)
    frame_height: int = Field(default=480, ge=32, le=4096)
    run_tag: Optional[str] = Field(default=None, max_length=100)
    run_notes: Optional[str] = Field(default=None, max_length=1000)


# ────────────────────────────────────────────── Full evaluation ───── #

@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=Dict[str, Any])
async def start_full_benchmark(request: FullBenchmarkRequest) -> Dict[str, Any]:
    """
    Start a full evaluation benchmark (fire-and-forget).
    Returns run_id immediately.  Poll GET /benchmark/results/{run_id} for progress/results.
    """
    _require_model(request.model_name)
    run_id = await benchmark_engine.start_full(
        model_name=request.model_name,
        num_frames=request.num_frames,
        frame_width=request.frame_width,
        frame_height=request.frame_height,
        run_tag=request.run_tag,
        run_notes=request.run_notes,
    )
    return {
        "run_id": run_id,
        "status": "running",
        "message": f"Full benchmark started for '{request.model_name}' ({request.num_frames} frames). Poll /benchmark/results/{run_id} for progress.",
    }


# ────────────────────────────────────────────── Run / compare ─────── #

@router.post("/run", status_code=status.HTTP_200_OK, response_model=Dict[str, Any])
async def run_benchmark(request: BenchmarkRunRequest) -> Dict[str, Any]:
    """
    Run a latency benchmark for the specified model.
    Returns when the benchmark completes (synchronous from client's perspective).
    """
    _require_model(request.model_name)
    try:
        result = await benchmark_engine.run(
            model_name=request.model_name,
            warmup_runs=request.warmup_runs,
            measure_runs=request.measure_runs,
            frame_width=request.frame_width,
            frame_height=request.frame_height,
        )
    except Exception as exc:
        logger.exception("Benchmark error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark failed: {exc}",
        )

    if result.status == BenchmarkStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error_message or "Benchmark failed.",
        )
    return result.as_dict()


@router.post("/compare", status_code=status.HTTP_200_OK, response_model=Dict[str, Any])
async def compare_models(request: BenchmarkCompareRequest) -> Dict[str, Any]:
    """
    Benchmark multiple models and return a ranked comparison table.
    Models that fail do not block others from running.
    """
    for name in request.model_names:
        _require_model(name)

    try:
        comparison = await benchmark_engine.compare(
            model_names=request.model_names,
            warmup_runs=request.warmup_runs,
            measure_runs=request.measure_runs,
            frame_width=request.frame_width,
            frame_height=request.frame_height,
        )
    except Exception as exc:
        logger.exception("Comparison error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {exc}",
        )
    return comparison


# ────────────────────────────────────────────── Results queries ───── #

@router.get("/results", response_model=List[Dict[str, Any]])
async def list_results(
    model_name: Optional[str] = Query(default=None),
) -> List[Dict[str, Any]]:
    """List all benchmark results, optionally filtered by model name."""
    return benchmark_engine.list_results(model_name=model_name)

# ───────────────────────────────────────── Comparative eval ─── #

@router.post("/comparative", status_code=status.HTTP_202_ACCEPTED, response_model=Dict[str, Any])
async def start_comparative_benchmark(request: ComparativeBenchmarkRequest) -> Dict[str, Any]:
    """
    Start a same-frame multi-model comparative evaluation (fire-and-forget).
    Each model runs on identical frames; per-frame detections are compared.
    Returns run_id immediately.  Poll GET /benchmark/comparative/{run_id}.
    """
    for name in request.model_names:
        _require_model(name)
    run_id = await benchmark_engine.start_comparative(
        model_names=request.model_names,
        num_frames=request.num_frames,
        frame_width=request.frame_width,
        frame_height=request.frame_height,
        run_tag=request.run_tag,
        run_notes=request.run_notes,
    )
    return {
        "run_id": run_id,
        "status": "running",
        "message": f"Comparative benchmark started for {request.model_names} ({request.num_frames} frames).",
    }


@router.get("/comparative", response_model=List[Dict[str, Any]])
async def list_comparative_results() -> List[Dict[str, Any]]:
    """List all comparative benchmark results."""
    return benchmark_engine.list_comparative()


@router.get("/comparative/{run_id}", response_model=Dict[str, Any])
async def get_comparative_result(run_id: str) -> Dict[str, Any]:
    """Get a specific comparative benchmark result by run ID."""
    result = benchmark_engine.get_comparative(run_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparative run '{run_id}' not found.",
        )
    return result.as_dict()

@router.get("/results/{run_id}", response_model=Dict[str, Any])
async def get_result(run_id: str) -> Dict[str, Any]:
    """Retrieve a specific benchmark result by its run ID."""
    result = benchmark_engine.get_result(run_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark run '{run_id}' not found.",
        )
    return result.as_dict()


@router.get("/results/model/{model_name}", response_model=List[Dict[str, Any]])
async def get_model_results(model_name: str) -> List[Dict[str, Any]]:
    """Get all benchmark results for a specific model."""
    results = benchmark_engine.list_results(model_name=model_name)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No benchmark results for model '{model_name}'.",
        )
    return results


# ────────────────────────────────────────────── Helpers ───────────── #

def _require_model(model_name: str) -> None:
    available = model_manager.available_names()
    if available and model_name not in available:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not registered. Available: {available}",
        )
