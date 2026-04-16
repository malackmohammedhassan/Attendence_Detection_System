"""
Log Streamer.

Captures Python logging output and forwards it in real-time to all
WebSocket clients subscribed to the "logs" channel.

Implementation:
  - Custom logging.Handler subclass enqueues log records.
  - Background asyncio task drains the queue and broadcasts via WebSocketManager.
  - In-memory ring buffer stores recent records for "late-joining" clients.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Log channel name used by WebSocketManager
WS_CHANNEL_LOGS = "logs"


@dataclass
class LogEntry:
    level: str
    logger_name: str
    message: str
    timestamp: str
    pathname: str
    lineno: int
    thread_name: str
    exc_text: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": "log",
            "level": self.level,
            "logger": self.logger_name,
            "message": self.message,
            "timestamp": self.timestamp,
            "location": f"{self.pathname}:{self.lineno}",
            "thread": self.thread_name,
            "exc_text": self.exc_text,
        }


class _AsyncQueueHandler(logging.Handler):
    """
    Thread-safe logging handler that puts records into a queue.
    The streamer task drains this queue asynchronously.
    """

    def __init__(self, record_queue: "queue.Queue[LogEntry]") -> None:
        super().__init__()
        self._queue = record_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = LogEntry(
                level=record.levelname,
                logger_name=record.name,
                message=self.format(record),
                timestamp=datetime.utcfromtimestamp(record.created).isoformat(),
                pathname=record.pathname,
                lineno=record.lineno,
                thread_name=record.threadName or "",
                exc_text=record.exc_text or None,
            )
            self._queue.put_nowait(entry)
        except Exception:
            self.handleError(record)


class LogStreamer:
    """
    Bridges the Python logging system to WebSocket clients.

    Lifecycle (called from app lifespan):
        streamer.attach(min_level=logging.INFO)   # install handler
        streamer.start_streaming(ws_manager)       # begin drain loop
        ...
        streamer.stop()                            # shutdown
    """

    def __init__(self, buffer_size: int = 500) -> None:
        self._buffer: Deque[LogEntry] = deque(maxlen=buffer_size)
        self._queue: "queue.Queue[LogEntry]" = queue.Queue(maxsize=2000)
        self._handler: Optional[_AsyncQueueHandler] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._ws_manager: Any = None  # WebSocketManager — injected at start

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def attach(
        self,
        root_logger: logging.Logger = logging.root,
        min_level: int = logging.INFO,
        formatter: Optional[logging.Formatter] = None,
    ) -> None:
        """Install the queue handler into the given logger."""
        if self._handler is not None:
            return  # already attached

        self._handler = _AsyncQueueHandler(self._queue)
        self._handler.setLevel(min_level)

        if formatter is None:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        self._handler.setFormatter(formatter)
        root_logger.addHandler(self._handler)
        logger.info("LogStreamer attached at level %s", logging.getLevelName(min_level))

    def detach(self, root_logger: logging.Logger = logging.root) -> None:
        if self._handler:
            root_logger.removeHandler(self._handler)
            self._handler = None

    def start_streaming(self, ws_manager: Any) -> None:
        """Start the asyncio drain task. Must be called from an async context."""
        self._ws_manager = ws_manager
        self._running = True
        self._task = asyncio.create_task(self._drain_loop(), name="log-streamer")
        logger.debug("LogStreamer drain task started")

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self.detach()
        logger.debug("LogStreamer stopped")

    # ------------------------------------------------------------------ #
    #  Recent logs for late-joining clients
    # ------------------------------------------------------------------ #

    def get_recent(self, n: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent `n` buffered log entries."""
        recent = list(self._buffer)[-n:]
        return [e.as_dict() for e in recent]

    # ------------------------------------------------------------------ #
    #  Drain loop
    # ------------------------------------------------------------------ #

    async def _drain_loop(self) -> None:
        """Continuously drain the log queue and broadcast to WebSocket clients."""
        while self._running:
            try:
                # Batch draining: consume up to 20 entries per iteration
                batch: List[LogEntry] = []
                for _ in range(20):
                    try:
                        entry = self._queue.get_nowait()
                        batch.append(entry)
                        self._buffer.append(entry)
                    except queue.Empty:
                        break

                if batch and self._ws_manager:
                    for entry in batch:
                        await self._ws_manager.broadcast(
                            entry.as_dict(), channel=WS_CHANNEL_LOGS
                        )
                else:
                    await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                # Log to stderr only — avoid recursion
                import sys
                print(f"[LogStreamer] drain_loop error: {exc}", file=sys.stderr)
                await asyncio.sleep(0.1)

    def queue_size(self) -> int:
        return self._queue.qsize()

    def buffer_size(self) -> int:
        return len(self._buffer)


# Module-level singleton
log_streamer = LogStreamer()
