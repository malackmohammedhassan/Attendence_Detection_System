"""
/api/internal — Internal observability endpoints.

These routes expose operational metrics that are NOT part of the public
dashboard API (no frontend widget depends on them).  They are used by:
  - Ops dashboards / Prometheus scrapers
  - Health probes that need more detail than /api/health
  - Stress-test scripts asserting system invariants

Routes
──────
  GET /api/internal/metrics      — live counter snapshot
  GET /api/internal/db/stats     — SQLite size + row counts
  POST /api/internal/counters/reset  — reset all counters (test use only)
"""

from __future__ import annotations

import platform
import time
from typing import Any, Dict

import psutil
from fastapi import APIRouter

from app.utils.counters import counters
from app.utils.db import db

router = APIRouter(prefix="/api/internal", tags=["internal"])

_BOOT_TIME: float = time.time()


@router.get("/metrics")
async def get_internal_metrics() -> Dict[str, Any]:
    """
    Return the live observability counter snapshot plus light process info.

    Includes ``process_start_time`` (Unix epoch) and
    ``server_uptime_seconds`` so restart debugging is trivial.
    """
    snap = counters.as_dict()

    # Lightweight process snapshot (does not block)
    try:
        proc = psutil.Process()
        with proc.oneshot():
            create_time = proc.create_time()          # Unix epoch float
            uptime_s    = round(time.time() - create_time, 1)
            proc_info: Dict[str, Any] = {
                "pid":                  proc.pid,
                "process_start_time":   create_time,
                "server_uptime_seconds": uptime_s,
                "cpu_percent":          round(proc.cpu_percent(interval=None), 1),
                "memory_rss_mb":        round(proc.memory_info().rss / 1_048_576, 1),
                "threads":              proc.num_threads(),
            }
    except Exception:
        proc_info = {}

    return {**snap, "process": proc_info}


@router.get("/db/stats")
async def get_db_stats() -> Dict[str, Any]:
    """Return SQLite database statistics (row counts, file size)."""
    return db.stats()


@router.post("/counters/reset", status_code=204)
async def reset_counters() -> None:
    """
    Reset all observability counters to zero.

    **Warning:** intended for automated tests only.  Calling this in
    production will break any monitoring that derives rates from cumulative
    counters.
    """
    counters.reset()
