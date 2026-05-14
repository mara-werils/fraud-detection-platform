"""WebSocket connection hub backed by Redis Pub/Sub.

Allows horizontal scaling: multiple scoring-service replicas share the same
Redis channel. A scored transaction published on any replica is fanned out
to all WebSocket clients connected to any replica.

Architecture:
  Scorer → Redis PUBLISH fraud:events:{org_id}
         → Redis SUBSCRIBE in ws_hub → broadcast to local WS connections
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

import structlog
from fastapi import WebSocket
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)

CHANNEL_PREFIX = "fraud:events:"
GLOBAL_CHANNEL = "fraud:events:*"


class WebSocketConnection:
    """Wraps a single WebSocket with metadata."""

    __slots__ = ("ws", "org_id", "user_id", "subscriptions")

    def __init__(self, ws: WebSocket, org_id: str, user_id: str) -> None:
        self.ws = ws
        self.org_id = org_id
        self.user_id = user_id
        self.subscriptions: set[str] = {org_id}

    async def send(self, data: dict) -> bool:
        """Send a message; return False if the connection is closed."""
        try:
            await self.ws.send_json(data)
            return True
        except Exception:
            return False


class WebSocketHub:
    """Central hub managing all WebSocket connections across the service.

    Subscribes to Redis Pub/Sub channels for cross-replica fan-out.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._connections: dict[str, set[WebSocketConnection]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._pubsub_task: asyncio.Task | None = None
        self._pubsub = None

    async def start(self) -> None:
        """Start the Redis Pub/Sub listener."""
        self._pubsub = self._redis.pubsub()
        await self._pubsub.psubscribe(GLOBAL_CHANNEL)
        self._pubsub_task = asyncio.create_task(
            self._listen(), name="ws-hub-pubsub"
        )
        await logger.ainfo("websocket_hub_started")

    async def stop(self) -> None:
        if self._pubsub_task:
            self._pubsub_task.cancel()
        if self._pubsub:
            await self._pubsub.punsubscribe(GLOBAL_CHANNEL)
            await self._pubsub.aclose()
        await logger.ainfo("websocket_hub_stopped")

    async def connect(self, conn: WebSocketConnection) -> None:
        """Register a new WebSocket connection."""
        await conn.ws.accept()
        async with self._lock:
            self._connections[conn.org_id].add(conn)
        await logger.ainfo(
            "ws_client_connected",
            org_id=conn.org_id,
            user_id=conn.user_id,
            total=len(self._connections[conn.org_id]),
        )

    async def disconnect(self, conn: WebSocketConnection) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections[conn.org_id].discard(conn)
            if not self._connections[conn.org_id]:
                del self._connections[conn.org_id]
        await logger.ainfo("ws_client_disconnected", org_id=conn.org_id, user_id=conn.user_id)

    async def publish(self, org_id: str, event: dict) -> None:
        """Publish an event to the Redis channel for an org."""
        channel = f"{CHANNEL_PREFIX}{org_id}"
        try:
            await self._redis.publish(channel, json.dumps(event, default=str))
        except Exception as exc:
            await logger.awarning("ws_hub_publish_error", error=str(exc))
            # Fall back to direct local broadcast
            await self._broadcast_local(org_id, event)

    async def _broadcast_local(self, org_id: str, event: dict) -> None:
        """Broadcast to WebSocket connections in this process only."""
        async with self._lock:
            conns = set(self._connections.get(org_id, set()))

        dead: set[WebSocketConnection] = set()
        for conn in conns:
            ok = await conn.send(event)
            if not ok:
                dead.add(conn)

        if dead:
            async with self._lock:
                for conn in dead:
                    self._connections[org_id].discard(conn)

    async def _listen(self) -> None:
        """Listen for Redis Pub/Sub messages and broadcast to local connections."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] not in ("pmessage", "message"):
                    continue
                channel: str = message.get("channel", b"").decode()
                org_id = channel.removeprefix(CHANNEL_PREFIX)
                try:
                    data = json.loads(message["data"])
                except (json.JSONDecodeError, KeyError):
                    continue
                await self._broadcast_local(org_id, data)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await logger.awarning("ws_hub_listen_error", error=str(exc))

    def stats(self) -> dict:
        total = sum(len(conns) for conns in self._connections.values())
        return {
            "total_connections": total,
            "active_orgs": len(self._connections),
        }


# ── FastAPI WebSocket endpoint ─────────────────────────────────────────────────

from fastapi import APIRouter, Query

ws_router = APIRouter(tags=["websocket"])


@ws_router.websocket("/ws/events")
async def websocket_events(
    websocket: WebSocket,
    org_id: str = Query(...),
    user_id: str = Query(default="anonymous"),
):
    """Real-time fraud event stream via WebSocket.

    Connect with: ws://host/ws/events?org_id=<id>&user_id=<id>
    Receives JSON events for every scored transaction in the org.
    """
    hub: WebSocketHub = websocket.app.state.ws_hub
    conn = WebSocketConnection(ws=websocket, org_id=org_id, user_id=user_id)

    await hub.connect(conn)
    try:
        while True:
            # Keep connection alive; server pushes data via publish()
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except Exception:
        pass
    finally:
        await hub.disconnect(conn)
