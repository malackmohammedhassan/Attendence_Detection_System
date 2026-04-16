"""
WebSocket connection manager.

Handles multiple concurrent clients with named channels (rooms).
Supports broadcasting to all clients, targeting individual connections,
and automatic cleanup on disconnect.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    websocket: WebSocket
    client_id: str
    channel: str
    connected_at: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0

    def __hash__(self) -> int:
        return hash(self.client_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Connection):
            return self.client_id == other.client_id
        return NotImplemented

    async def send_json(self, data: Dict[str, Any]) -> None:
        await self.websocket.send_json(data)


class WebSocketManager:
    """
    Central manager for all WebSocket connections.

    Channels are lightweight namespaces — e.g. "training", "metrics", "logs".
    Each client is assigned a unique ID on connection.
    """

    def __init__(self) -> None:
        # channel -> set of Connection objects
        self._channels: Dict[str, Set[Connection]] = defaultdict(set)
        # client_id -> Connection
        self._connections: Dict[str, Connection] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    async def connect(
        self, websocket: WebSocket, channel: str = "default"
    ) -> Connection:
        """Accept a new WebSocket and register it under the given channel."""
        await websocket.accept()
        client_id = str(uuid4())
        conn = Connection(
            websocket=websocket,
            client_id=client_id,
            channel=channel,
        )
        async with self._lock:
            self._channels[channel].add(conn)
            self._connections[client_id] = conn

        logger.info(
            "WS connected | client=%s | channel=%s | total=%d",
            client_id,
            channel,
            len(self._connections),
        )
        return conn

    async def disconnect(self, conn: Connection) -> None:
        """Remove a connection from all registries."""
        async with self._lock:
            self._channels[conn.channel].discard(conn)
            if not self._channels[conn.channel]:
                del self._channels[conn.channel]
            self._connections.pop(conn.client_id, None)

        logger.info(
            "WS disconnected | client=%s | channel=%s | total=%d",
            conn.client_id,
            conn.channel,
            len(self._connections),
        )

    # ------------------------------------------------------------------ #
    #  Sending helpers
    # ------------------------------------------------------------------ #

    async def send_to_client(
        self, client_id: str, data: Dict[str, Any]
    ) -> bool:
        """Send a JSON payload to a specific client. Returns False if not found."""
        conn = self._connections.get(client_id)
        if conn is None:
            return False
        try:
            await conn.send_json(data)
            conn.message_count += 1
            return True
        except Exception as exc:
            logger.warning("Failed to send to client %s: %s", client_id, exc)
            await self.disconnect(conn)
            return False

    async def broadcast(
        self,
        data: Dict[str, Any],
        channel: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> int:
        """
        Broadcast a JSON payload.

        Args:
            data:    The payload to send.
            channel: If set, only send to clients in this channel.
                     If None, send to all connected clients.
            exclude: Optional client_id to skip.

        Returns:
            Number of clients successfully reached.
        """
        if channel is not None:
            targets: List[Connection] = list(self._channels.get(channel, set()))
        else:
            targets = list(self._connections.values())

        if exclude:
            targets = [c for c in targets if c.client_id != exclude]

        results = await asyncio.gather(
            *[self._safe_send(conn, data) for conn in targets],
            return_exceptions=True,
        )

        sent = sum(1 for r in results if r is True)
        return sent

    async def _safe_send(self, conn: Connection, data: Dict[str, Any]) -> bool:
        try:
            await conn.send_json(data)
            conn.message_count += 1
            return True
        except Exception as exc:
            logger.warning("Dropping client %s: %s", conn.client_id, exc)
            await self.disconnect(conn)
            return False

    # ------------------------------------------------------------------ #
    #  Heartbeat
    # ------------------------------------------------------------------ #

    async def broadcast_heartbeat(self) -> None:
        """Send a ping-style heartbeat to all connected clients."""
        await self.broadcast(
            {
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat(),
                "connections": len(self._connections),
            }
        )

    # ------------------------------------------------------------------ #
    #  Introspection
    # ------------------------------------------------------------------ #

    def active_connections(self) -> int:
        return len(self._connections)

    def active_channels(self) -> List[str]:
        return list(self._channels.keys())

    def channel_stats(self) -> Dict[str, int]:
        return {ch: len(conns) for ch, conns in self._channels.items()}

    def get_connection(self, client_id: str) -> Optional[Connection]:
        return self._connections.get(client_id)


# Module-level singleton — imported and shared across the app
ws_manager = WebSocketManager()
