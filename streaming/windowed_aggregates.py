"""Sliding-window aggregations backed by Redis sorted sets."""

from __future__ import annotations

import math
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis
import structlog

__all__ = [
    "WindowedAggregator",
]

logger = structlog.get_logger(__name__)

# Window definitions: name -> duration in seconds
WINDOWS: dict[str, int] = {
    "1min": 60,
    "5min": 300,
    "1hour": 3600,
    "24hour": 86400,
}

_KEY_PREFIX = "wnd"
_TTL_SECONDS = 2 * 86400  # 48 hours — covers the largest window with margin


class WindowedAggregator:
    """Computes and stores per-user sliding-window aggregates in Redis.

    Data model (per user, per window):
      - sorted set ``wnd:{user_id}:txns:{window}``  members = ``{ts}:{amount}:{merchant}:{device}``  score = ts
    """

    def __init__(
        self,
        redis_url: str,
        max_connections: int = 20,
        socket_timeout: float = 5.0,
    ) -> None:
        self._redis_url = redis_url
        self._max_connections = max_connections
        self._socket_timeout = socket_timeout
        self._redis: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Redis connection pool."""
        self._redis = aioredis.from_url(
            self._redis_url,
            max_connections=self._max_connections,
            socket_timeout=self._socket_timeout,
            decode_responses=True,
        )
        await logger.ainfo("windowed_aggregator_connected", redis_url=self._redis_url)

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            await logger.ainfo("windowed_aggregator_closed")

    async def health_check(self) -> bool:
        """Return True when Redis is reachable."""
        if self._redis is None:
            return False
        try:
            return await self._redis.ping()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record_transaction(
        self,
        user_id: UUID,
        timestamp: float,
        amount: float,
        merchant_id: str | None,
        device_id: str | None,
    ) -> None:
        """Atomically record a transaction across all window sorted sets.

        Uses a Redis pipeline so every window is updated in a single round-trip.
        Expired entries are pruned automatically.
        """
        if self._redis is None:
            raise RuntimeError("WindowedAggregator is not connected")

        uid = str(user_id)
        merchant = merchant_id or "_"
        device = device_id or "_"
        member = f"{timestamp}:{amount}:{merchant}:{device}"

        async with self._redis.pipeline(transaction=False) as pipe:
            for window_name, window_secs in WINDOWS.items():
                key = f"{_KEY_PREFIX}:{uid}:txns:{window_name}"
                pipe.zadd(key, {member: timestamp})
                # Prune entries older than the window
                pipe.zremrangebyscore(key, "-inf", timestamp - window_secs)
                pipe.expire(key, _TTL_SECONDS)
            await pipe.execute()

        await logger.adebug(
            "transaction_recorded",
            user_id=uid,
            amount=amount,
            merchant=merchant,
        )

    async def get_aggregates(
        self,
        user_id: UUID,
        now_ts: float,
    ) -> dict[str, dict[str, Any]]:
        """Retrieve aggregates for all windows.

        Returns a dict keyed by window name, each containing:
            txn_count, txn_sum, unique_merchants, unique_devices, amount_std
        """
        if self._redis is None:
            raise RuntimeError("WindowedAggregator is not connected")

        uid = str(user_id)

        # Fetch all members for each window in one pipeline
        async with self._redis.pipeline(transaction=False) as pipe:
            for window_name, window_secs in WINDOWS.items():
                key = f"{_KEY_PREFIX}:{uid}:txns:{window_name}"
                min_score = now_ts - window_secs
                pipe.zrangebyscore(key, min_score, now_ts)
            raw_results: list[list[str]] = await pipe.execute()

        aggregates: dict[str, dict[str, Any]] = {}

        for (window_name, _), members in zip(WINDOWS.items(), raw_results):
            amounts: list[float] = []
            merchants: set[str] = set()
            devices: set[str] = set()

            for member in members or []:
                parts = member.split(":", 3)
                if len(parts) < 4:
                    continue
                try:
                    amt = float(parts[1])
                except (ValueError, IndexError):
                    continue
                amounts.append(amt)
                if parts[2] != "_":
                    merchants.add(parts[2])
                if parts[3] != "_":
                    devices.add(parts[3])

            txn_count = len(amounts)
            txn_sum = sum(amounts)

            # Standard deviation
            amount_std = 0.0
            if txn_count >= 2:
                mean = txn_sum / txn_count
                variance = sum((a - mean) ** 2 for a in amounts) / txn_count
                amount_std = math.sqrt(variance)

            aggregates[window_name] = {
                "txn_count": txn_count,
                "txn_sum": round(txn_sum, 2),
                "unique_merchants": len(merchants),
                "unique_devices": len(devices),
                "amount_std": round(amount_std, 4),
            }

        return aggregates

    async def cleanup_expired(self, user_id: UUID, now_ts: float) -> None:
        """Explicitly remove expired entries from all windows for a user.

        This is normally handled inline during ``record_transaction``,
        but can be called independently for maintenance.
        """
        if self._redis is None:
            raise RuntimeError("WindowedAggregator is not connected")

        uid = str(user_id)

        async with self._redis.pipeline(transaction=False) as pipe:
            for window_name, window_secs in WINDOWS.items():
                key = f"{_KEY_PREFIX}:{uid}:txns:{window_name}"
                pipe.zremrangebyscore(key, "-inf", now_ts - window_secs)
            await pipe.execute()

        await logger.adebug("expired_entries_cleaned", user_id=uid)
