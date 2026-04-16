"""
FastAPI Application Entry Point.

Responsibilities:
  - App factory with full lifespan management
  - CORS middleware registration
  - All router registration
  - WebSocket manager initialization
  - Background service startup (performance tracker, log streamer)
  - Health check + status endpoints
  - Structured logging configuration
  - Graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
import logging.config
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
try:
    from uvicorn.protocols.utils import ClientDisconnected  # type: ignore[import-untyped]
except ImportError:
    ClientDisconnected = Exception  # fallback — keeps code version-safe

from app.config import Settings, get_settings
from app.routes import benchmark, export, inference, internal, live, metrics, train
from app.services.log_streamer import WS_CHANNEL_LOGS, log_streamer
from app.utils.performance_tracker import performance_tracker
from app.websocket_manager import ws_manager

# ────────────────────────────────────────────── Logging setup ─────── #

def configure_logging(settings: Settings) -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": settings.log_format,
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": settings.log_level,
                "handlers": ["console"],
            },
            "loggers": {
                "uvicorn.access": {"level": "WARNING"},
                "uvicorn.error": {"level": "INFO"},
            },
        }
    )


logger = logging.getLogger(__name__)

# ────────────────────────────────────────────── Lifespan ──────────── #

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown of all background services."""
    settings = get_settings()
    configure_logging(settings)

    logger.info("=== %s v%s starting ===", settings.app_name, settings.app_version)
    logger.info("Debug mode: %s | Log level: %s", settings.debug, settings.log_level)

    # 1. Performance tracker (background thread)
    performance_tracker._interval = settings.perf_sample_interval_sec
    performance_tracker._window = settings.perf_rolling_window
    performance_tracker.start()
    logger.info("PerformanceTracker started")

    # 2. Log streamer — attach handler and start drain task
    log_streamer.attach(min_level=logging.INFO)
    log_streamer.start_streaming(ws_manager)
    logger.info("LogStreamer started")

    # 3. Heartbeat task — keep WS connections alive
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(settings.ws_heartbeat_interval_sec),
        name="ws-heartbeat",
    )
    logger.info("WebSocket heartbeat started (interval=%.0fs)", settings.ws_heartbeat_interval_sec)

    # 4. Register model adapters (lazy-load; weights only pulled on activate())
    from app.services.model_manager import model_manager  # noqa: PLC0415
    from app.adapters.scratch_cnn_adapter import ScratchCNNAdapter  # noqa: PLC0415
    from app.adapters.mtcnn_adapter import MTCNNAdapter              # noqa: PLC0415

    model_manager.register(ScratchCNNAdapter())
    logger.info("Registered adapter: scratch_cnn")

    try:
        model_manager.register(MTCNNAdapter())
        logger.info("Registered adapter: mtcnn")
    except Exception as _mtcnn_exc:
        logger.warning("mtcnn adapter registration skipped: %s", _mtcnn_exc)

    # Auto-activate the default model so benchmarks work immediately
    _default = settings.default_model
    try:
        model_manager.set_active(_default)
        logger.info("Auto-activated default model: %s", _default)
    except Exception as _act_exc:
        logger.warning("Could not auto-activate '%s': %s", _default, _act_exc)

    logger.info("=== Startup complete. Accepting requests. ===")

    try:
        yield  # ── application runs here ──────────────────────────────────
    finally:
        logger.info("=== Shutdown initiated ===")

        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        log_streamer.stop()
        performance_tracker.stop()

        logger.info("=== Shutdown complete ===")


async def _heartbeat_loop(interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await ws_manager.broadcast_heartbeat()
        except Exception as exc:
            logger.warning("Heartbeat error: %s", exc)


# ────────────────────────────────────────────── App factory ───────── #

def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "ML Evaluation Dashboard API — real-time inference, training, "
            "benchmarking, and performance analytics for face detection models."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # ── Request timing middleware ──────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    # ── Global exception handler ───────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s: %s", request.method, request.url, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc) if settings.debug else "An unexpected error occurred.",
            },
        )

    # ── Routers ───────────────────────────────────────────────────────
    PREFIX = "/api/v1"
    app.include_router(train.router, prefix=PREFIX)
    app.include_router(inference.router, prefix=PREFIX)
    app.include_router(metrics.router, prefix=PREFIX)
    app.include_router(benchmark.router, prefix=PREFIX)
    app.include_router(export.router, prefix=PREFIX)
    # Live streaming — registered at root (no /api/v1 prefix) so WS URL
    # matches the Vite proxy rule /ws/** → ws://localhost:8000
    app.include_router(live.router)
    app.include_router(internal.router)  # /api/internal/*

    # ── Health / status ───────────────────────────────────────────────
    @app.get("/health", tags=["system"])
    async def health_check() -> Dict[str, Any]:
        """Lightweight health probe for load balancers / Docker."""
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
        }

    @app.get("/status", tags=["system"])
    async def system_status() -> Dict[str, Any]:
        """Detailed system status for the dashboard overview."""
        snap = performance_tracker.snapshot()
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "websockets": {
                "active_connections": ws_manager.active_connections(),
                "channels": ws_manager.channel_stats(),
            },
            "performance": snap.as_dict() if snap else None,
            "log_buffer_size": log_streamer.buffer_size(),
        }

    # ── Global WebSocket log endpoint ─────────────────────────────────
    @app.websocket("/ws/logs")
    async def ws_logs(websocket: WebSocket) -> None:
        """Stream all application logs to connected dashboard clients."""
        conn = await ws_manager.connect(websocket, channel=WS_CHANNEL_LOGS)
        try:
            # Replay recent log buffer so client is immediately up to date
            recent = log_streamer.get_recent(100)
            for entry in recent:
                await conn.send_json({**entry, "type": "log_replay"})
            while True:
                await asyncio.sleep(30)  # heartbeat handled by global task
        except (WebSocketDisconnect, ClientDisconnected):
            pass
        finally:
            await ws_manager.disconnect(conn)

    @app.websocket("/ws/metrics")
    async def ws_metrics_global(websocket: WebSocket) -> None:
        """Global metrics WebSocket — alias for /api/v1/metrics/live."""
        conn = await ws_manager.connect(websocket, channel="metrics")
        try:
            while True:
                snap = performance_tracker.snapshot()
                await conn.send_json(
                    {
                        "type": "metrics_tick",
                        "system": snap.as_dict() if snap else None,
                    }
                )
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, ClientDisconnected):
            pass
        finally:
            await ws_manager.disconnect(conn)

    return app


# ────────────────────────────────────────────── Application instance ─ #

app = create_app()


# ────────────────────────────────────────────── Dev entrypoint ─────── #

if __name__ == "__main__":
    _settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=_settings.host,
        port=_settings.port,
        reload=_settings.reload or _settings.debug,
        log_level=_settings.log_level.lower(),
        access_log=True,
    )
