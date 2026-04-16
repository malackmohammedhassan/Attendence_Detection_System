"""
/ws/live — Real-Time Bidirectional Streaming WebSocket.

Protocol
────────
Client → Server:
  Binary:  raw JPEG frame bytes
  JSON:    {"type": "config", "threshold": 0.5}

Server → Client:
  {"type": "ready",        "model": "<name>", "has_model": bool, "client_id": str}
  {"type": "live_result",  "detections": [...], "fps": float, "latency_ms": float,
                            "model_name": str, "frame_width": int, "frame_height": int,
                            "detection_count": int, "timestamp": float}
  {"type": "metrics_tick", "cpu": float, "memory_mb": float, "fps": float,
                            "timestamp": float, "system": {...}}
  {"type": "no_model"}              — when no active model is set
  {"type": "error",  "message": str} — per-frame inference error

Architecture
────────────
Each WebSocket connection spawns four co-operative async tasks sharing one
asyncio event loop:

  _recv_loop    – reads WS messages; stores latest binary frame (ring-buffer
                  of 1 so slow inference never builds an unbounded backlog);
                  applies JSON config updates.
  _infer_loop   – waits on _frame_ready Event; takes the frame; runs the
                  async inference pipeline (already uses run_in_executor
                  internally so the loop stays unblocked); enqueues result.
  _metrics_loop – every METRICS_INTERVAL_SEC pushes a system snapshot so
                  charts stay animated even when no frames are being sent.
  _send_loop    – drains the outbound queue and writes to the WebSocket.

All tasks are cancelled and awaited on disconnect — no resource leaks.
Real inference plugs in transparently via inference_service.infer_from_bytes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.services.inference_service import inference_service
from app.services.model_manager import model_manager
from app.utils.counters import counters
from app.utils.performance_tracker import performance_tracker

_settings = get_settings()

logger = logging.getLogger(__name__)

router = APIRouter(tags=["live"])

# ── Tunables ──────────────────────────────────────────────────────────────────

METRICS_INTERVAL_SEC: float = 1.0   # how often to push system snapshot
MAX_OUT_QUEUE: int = 4              # drop oldest result when backlog exceeds this
DEFAULT_THRESHOLD: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  Shared inference broadcaster  (one inference loop, N subscribers)
# ─────────────────────────────────────────────────────────────────────────────

class InferenceBroadcaster:
    """
    Runs a single async inference loop and pushes each result dict to every
    subscribed per-session queue.

    This replaces the previous per-connection _infer_loop so that N connected
    clients share ONE model forward-pass rather than triggering N parallel
    inferences on the same frame.

    Frame submission is a ring-buffer of 1: if inference is slow, stale frames
    are evicted so the model always sees the freshest available frame.

    Threshold semantics: the broadcaster uses the threshold supplied by the
    client who submitted the most recent frame.  All subscribers receive the
    same filtered result.  This is a deliberate trade-off — it avoids N
    re-runs of the same forward-pass at the cost of threshold exactness.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._lock  = asyncio.Lock()
        # (frame_bytes, threshold) — maxsize=1 so newest always wins
        self._frame_q: asyncio.Queue[Tuple[bytes, float]] = asyncio.Queue(maxsize=1)
        self._task: Optional[asyncio.Task] = None
        # Held during the model-check + inference block so model switches can't
        # race with a forward pass.  Model-switch routes must await this lock
        # before calling model_manager.set_active().
        self.model_op_lock: asyncio.Lock = asyncio.Lock()
        # Rolling overflow timestamps for system_overloaded detection.
        # We keep timestamps of the last N overflows; if > OVERLOAD_THRESHOLD
        # overflows occur within OVERLOAD_WINDOW_SEC, system is overloaded.
        self._overflow_timestamps: list = []
        self._OVERLOAD_WINDOW_SEC: float = 10.0
        self._OVERLOAD_THRESHOLD: int    = 5   # >5 overflows in 10s = overloaded

    # ------------------------------------------------------------------
    #  Subscription management
    # ------------------------------------------------------------------

    async def subscribe(self, client_id: str, out_q: asyncio.Queue) -> None:
        """Register *out_q* as a result receiver for *client_id*."""
        async with self._lock:
            self._subscribers[client_id] = out_q
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(
                    self._infer_loop(),
                    name="broadcaster-infer",
                )
                logger.info("InferenceBroadcaster: inference task started")

    async def unsubscribe(self, client_id: str) -> None:
        """Remove *client_id* from subscribers; stop task when last one leaves."""
        async with self._lock:
            self._subscribers.pop(client_id, None)
            if not self._subscribers and self._task and not self._task.done():
                self._task.cancel()
                logger.info("InferenceBroadcaster: no subscribers — inference task stopped")

    # ------------------------------------------------------------------
    #  Frame ingestion
    # ------------------------------------------------------------------

    async def submit_frame(self, frame_bytes: bytes, threshold: float) -> None:
        """
        Offer a new frame for inference (non-blocking, ring-buffer of 1).
        If the queue already holds a pending frame it is evicted.
        """
        # Evict stale pending frame synchronously (non-blocking)
        if self._frame_q.full():
            try:
                self._frame_q.get_nowait()
                counters.inc_inference_queue_overflow()
                self._record_overflow()
            except asyncio.QueueEmpty:
                pass
        try:
            self._frame_q.put_nowait((frame_bytes, threshold))
        except asyncio.QueueFull:
            counters.inc_inference_queue_overflow()  # lost the race
            self._record_overflow()

    def _record_overflow(self) -> None:
        """Record overflow timestamp and prune old entries outside the window."""
        now = time.monotonic()
        self._overflow_timestamps.append(now)
        cutoff = now - self._OVERLOAD_WINDOW_SEC
        # Prune entries older than the window
        self._overflow_timestamps = [t for t in self._overflow_timestamps if t >= cutoff]

    def is_system_overloaded(self) -> bool:
        """Return True when overflow rate exceeds threshold within the rolling window."""
        now = time.monotonic()
        cutoff = now - self._OVERLOAD_WINDOW_SEC
        recent = sum(1 for t in self._overflow_timestamps if t >= cutoff)
        return recent > self._OVERLOAD_THRESHOLD

    # ------------------------------------------------------------------
    #  Inference task
    # ------------------------------------------------------------------

    async def _infer_loop(self) -> None:
        """Drain *_frame_q*, run inference, push result to all subscribers."""
        while True:
            try:
                frame_bytes, threshold = await self._frame_q.get()
            except asyncio.CancelledError:
                break

            # Hold model_op_lock for the entire check + forward-pass so that a
            # concurrent model switch cannot race between get_active_name() and
            # the actual inference call.
            async with self.model_op_lock:
                if model_manager.get_active_name() is None:
                    await self._broadcast({"type": "no_model"})
                    continue

                try:
                    result = await inference_service.infer_from_bytes(
                        frame_bytes,
                        confidence_threshold=threshold,
                    )
                    payload = {"type": "live_result", **result.as_dict()}
                    if self.is_system_overloaded():
                        payload["system_overloaded"] = True
                    await self._broadcast(payload)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.warning("InferenceBroadcaster inference error: %s", exc)
                    counters.inc_inference_errors()
                    await self._broadcast({"type": "error", "message": str(exc)})

    async def _broadcast(self, payload: dict) -> None:
        """Push *payload* to every subscriber's output queue."""
        async with self._lock:
            subscribers = dict(self._subscribers)
        for client_id, q in subscribers.items():
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # subscriber is backed up — drop this frame for them


# Module-level singleton shared across all connections
inference_broadcaster = InferenceBroadcaster()


# ─────────────────────────────────────────────────────────────────────────────
#  Per-connection session
# ─────────────────────────────────────────────────────────────────────────────

class LiveSession:
    """
    Manages one client connection for /ws/live.

    Everything runs inside the same asyncio event loop — no threading
    primitives are needed.  Only asyncio.Event / asyncio.Queue are used.
    """

    __slots__ = (
        "ws", "client_id",
        "_out_q", "_threshold", "_running",
        # rate-limiting / real FPS tracking
        "_last_frame_ts", "_fps_counter", "_fps_window_start", "_live_fps",
    )

    def __init__(self, websocket: WebSocket, client_id: str) -> None:
        self.ws = websocket
        self.client_id = client_id

        # Outbound queue: results + metric ticks before they go to the WS
        self._out_q: asyncio.Queue[dict] = asyncio.Queue(maxsize=MAX_OUT_QUEUE)

        # Client-controlled config
        self._threshold: float = DEFAULT_THRESHOLD
        self._running: bool    = True

        # Server-side rate limiting state
        self._last_frame_ts:    float = 0.0
        # Per-connection real FPS counter (reset every second)
        self._fps_counter:      int   = 0
        self._fps_window_start: float = time.monotonic()
        self._live_fps:         float = 0.0

    # ------------------------------------------------------------------ #
    #  Entry point
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """Subscribe to broadcaster, spawn tasks, wait until the first exits."""
        await inference_broadcaster.subscribe(self.client_id, self._out_q)
        tasks = [
            asyncio.create_task(
                self._recv_loop(),
                name=f"live-recv-{self.client_id[:8]}"
            ),
            asyncio.create_task(
                self._metrics_loop(),
                name=f"live-metrics-{self.client_id[:8]}"
            ),
            asyncio.create_task(
                self._send_loop(),
                name=f"live-send-{self.client_id[:8]}"
            ),
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                if not t.cancelled() and t.exception():
                    logger.debug(
                        "LiveSession task %s raised: %s",
                        t.get_name(), t.exception()
                    )
        finally:
            self._running = False
            await inference_broadcaster.unsubscribe(self.client_id)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(
                "LiveSession %s fully cleaned up", self.client_id[:8]
            )

    # ------------------------------------------------------------------ #
    #  Tasks
    # ------------------------------------------------------------------ #

    async def _recv_loop(self) -> None:
        """
        Read raw WS messages.
        - Binary → store as latest frame (ring-buffer of 1).
        - Text / JSON → apply config update.
        - Disconnect message → exit cleanly.
        """
        while self._running:
            try:
                msg = await self.ws.receive()
            except (WebSocketDisconnect, RuntimeError):
                break

            msg_type = msg.get("type", "")

            if msg_type == "websocket.disconnect":
                break

            raw_bytes: Optional[bytes] = msg.get("bytes")
            raw_text: Optional[str] = msg.get("text")

            if raw_bytes:
                # Server-side rate cap — drop frames arriving too fast
                now = time.monotonic()
                min_interval = 1.0 / _settings.live_max_fps
                if (now - self._last_frame_ts) < min_interval:
                    continue   # drop — too soon since last accepted frame
                self._last_frame_ts = now

                # Real per-connection FPS counter (1-second rolling window)
                self._fps_counter += 1
                elapsed_window = now - self._fps_window_start
                if elapsed_window >= 1.0:
                    self._live_fps         = self._fps_counter / elapsed_window
                    self._fps_counter      = 0
                    self._fps_window_start = now

                # Hand off to the shared broadcaster (ring-buffer of 1)
                await inference_broadcaster.submit_frame(raw_bytes, self._threshold)

    # ------------------------------------------------------------------ #
    #  Metrics, send, and helper tasks (unchanged)
    # ------------------------------------------------------------------ #

    async def _metrics_loop(self) -> None:
        """Push system snapshot every METRICS_INTERVAL_SEC."""
        while self._running:
            await asyncio.sleep(METRICS_INTERVAL_SEC)
            if not self._running:
                break

            snap = performance_tracker.snapshot()
            if snap is None:
                continue

            # Use real per-connection FPS; fall back to system snapshot fps
            live_fps = self._live_fps if self._live_fps > 0 else snap.fps

            await self._enqueue(
                {
                    "type":       "metrics_tick",
                    "cpu":        round(snap.cpu_percent, 1),
                    "memory_mb":  round(snap.memory_used_mb, 1),
                    "fps":        round(live_fps, 2),
                    "timestamp":  time.time(),
                    "system":     snap.as_dict(),   # full payload for charts
                }
            )

    async def _send_loop(self) -> None:
        """
        Drain the outbound queue and write each payload to the WebSocket.
        Exits on any send error (connection closed from client side).
        """
        while self._running:
            try:
                payload = await asyncio.wait_for(
                    self._out_q.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue

            try:
                await self.ws.send_json(payload)
            except Exception as exc:
                logger.debug(
                    "Failed to send to client %s: %s",
                    self.client_id[:8], exc
                )
                break  # connection is gone — exit; run() will cancel all tasks

    # ------------------------------------------------------------------ #
    #  Helper
    # ------------------------------------------------------------------ #

    async def _enqueue(self, payload: dict) -> None:
        """
        Put payload on the outbound queue.
        If the queue is full, drop the oldest item first (we prefer fresh data).
        """
        if self._out_q.full():
            try:
                self._out_q.get_nowait()   # evict oldest
            except asyncio.QueueEmpty:
                pass
        try:
            self._out_q.put_nowait(payload)
        except asyncio.QueueFull:
            pass   # still full after eviction — this frame is skipped


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """
    Real-time bidirectional frame streaming endpoint.

    Clients send binary JPEG frames; the server returns JSON detection
    results and periodic system-metrics snapshots.
    """
    client_id = str(uuid4())
    await websocket.accept()

    logger.info("LiveStream CONNECTED  | client=%s", client_id[:8])
    counters.inc_ws_connected()

    # Announce readiness with current model state
    active_model = model_manager.get_active_name()
    await websocket.send_json(
        {
            "type":      "ready",
            "model":     active_model or "",
            "has_model": active_model is not None,
            "client_id": client_id,
        }
    )

    session = LiveSession(websocket, client_id)
    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(
            "LiveSession %s unexpected error: %s", client_id[:8], exc
        )
    finally:
        counters.dec_ws_connected()
        logger.info("LiveStream DISCONNECTED | client=%s", client_id[:8])
