"""Redis-based online feature store for real-time feature serving."""

from __future__ import annotations

import math
import time
from decimal import Decimal
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis
import structlog

from shared.schemas import FeatureVector, Transaction

logger = structlog.get_logger(__name__)

# Sliding window durations in seconds
_WINDOWS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "1h": 3600,
    "24h": 86400,
}

_SEVEN_DAYS = 7 * 86400
_TTL_SECONDS = 30 * 86400  # 30 days


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in kilometres between two lat/lon points."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class OnlineFeatureStore:
    """Manages real-time feature computation and retrieval backed by Redis."""

    def __init__(self, redis_url: str, max_connections: int = 20, socket_timeout: float = 5.0) -> None:
        self._redis_url = redis_url
        self._max_connections = max_connections
        self._socket_timeout = socket_timeout
        self._redis: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish the Redis connection pool."""
        self._redis = aioredis.from_url(
            self._redis_url,
            max_connections=self._max_connections,
            socket_timeout=self._socket_timeout,
            decode_responses=True,
        )
        await logger.ainfo("online_store_connected", redis_url=self._redis_url)

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            await logger.ainfo("online_store_closed")

    async def health_check(self) -> bool:
        """Return True if Redis is reachable."""
        if self._redis is None:
            return False
        try:
            return await self._redis.ping()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update_features(self, transaction: Transaction) -> None:
        """Update all online features for a user atomically.

        Runs velocity, amount-stats, geo, device, and merchant updates
        inside a single Redis pipeline so the write is atomic.
        """
        if self._redis is None:
            raise RuntimeError("OnlineFeatureStore is not connected")

        user_id = str(transaction.user_id)
        now_ts = transaction.timestamp.timestamp()
        amount = float(transaction.amount)

        async with self._redis.pipeline(transaction=False) as pipe:
            await self._update_velocity(pipe, user_id, amount, now_ts)
            await self._update_amount_stats(pipe, user_id, amount)
            await self._update_geo(pipe, user_id, transaction.latitude, transaction.longitude, now_ts)
            await self._update_device(pipe, user_id, transaction.device_id)
            await self._update_merchant(pipe, user_id, transaction.merchant_id, now_ts)

            # Record last-transaction timestamp
            pipe.hset(f"user:{user_id}:meta", "last_txn_ts", str(now_ts))

            # Refresh TTLs on all user keys
            for suffix in ("velocity", "stats", "geo", "devices", "merchants_7d", "meta"):
                pipe.expire(f"user:{user_id}:{suffix}", _TTL_SECONDS)

            await pipe.execute()

        await logger.adebug("features_updated", user_id=user_id, txn_id=str(transaction.transaction_id))

    async def get_features(self, user_id: UUID, transaction: Transaction) -> FeatureVector:
        """Retrieve the current feature vector for a user/transaction pair."""
        if self._redis is None:
            raise RuntimeError("OnlineFeatureStore is not connected")

        uid = str(user_id)
        now_ts = transaction.timestamp.timestamp()
        amount = float(transaction.amount)

        async with self._redis.pipeline(transaction=False) as pipe:
            # Velocity counts and sums per window
            for window_name, window_secs in _WINDOWS.items():
                min_score = now_ts - window_secs
                pipe.zcount(f"user:{uid}:velocity", min_score, now_ts)
                pipe.zrangebyscore(f"user:{uid}:velocity", min_score, now_ts)

            # Amount stats
            pipe.hgetall(f"user:{uid}:stats")

            # Geo
            pipe.hgetall(f"user:{uid}:geo")

            # Devices
            pipe.scard(f"user:{uid}:devices")
            if transaction.device_id:
                pipe.sismember(f"user:{uid}:devices", transaction.device_id)
            else:
                pipe.scard(f"user:{uid}:devices")  # dummy — will ignore result

            # Merchants 24h (unique count) & 7d
            pipe.zcount(f"user:{uid}:merchants_7d", now_ts - 86400, now_ts)
            pipe.zcount(f"user:{uid}:merchants_7d", now_ts - _SEVEN_DAYS, now_ts)
            if transaction.merchant_id:
                pipe.zscore(f"user:{uid}:merchants_7d", transaction.merchant_id)
            else:
                pipe.zcard(f"user:{uid}:merchants_7d")  # dummy

            # Meta
            pipe.hget(f"user:{uid}:meta", "last_txn_ts")

            results = await pipe.execute()

        idx = 0
        # Parse velocity results — 2 results per window (count, members)
        velocity_counts: dict[str, int] = {}
        velocity_amounts: dict[str, Decimal] = {}
        for window_name in _WINDOWS:
            count = results[idx] or 0
            members = results[idx + 1] or []
            velocity_counts[window_name] = int(count)
            # Members are stored as "timestamp:amount"
            window_sum = Decimal("0")
            amounts_in_window: list[float] = []
            for member in members:
                try:
                    _, amt_str = member.rsplit(":", 1)
                    amt = float(amt_str)
                    amounts_in_window.append(amt)
                    window_sum += Decimal(str(amt))
                except (ValueError, IndexError):
                    pass
            velocity_amounts[window_name] = window_sum
            idx += 2

        # Amount stats
        stats: dict[str, str] = results[idx] or {}
        idx += 1

        # Geo
        geo: dict[str, str] = results[idx] or {}
        idx += 1

        # Devices
        device_count: int = results[idx] or 0
        idx += 1
        is_new_device_raw = results[idx]
        is_new_device = not bool(is_new_device_raw) if transaction.device_id else False
        idx += 1

        # Merchants
        unique_merchants_24h: int = results[idx] or 0
        idx += 1
        _unique_merchants_7d: int = results[idx] or 0
        idx += 1
        merchant_score = results[idx]
        is_new_merchant = merchant_score is None if transaction.merchant_id else False
        idx += 1

        # Meta — last txn timestamp
        last_txn_ts_raw = results[idx]
        idx += 1
        time_since_last: float = 0.0
        if last_txn_ts_raw:
            time_since_last = max(0.0, now_ts - float(last_txn_ts_raw))

        # Compute amount stats
        running_avg = float(stats.get("avg", "0"))
        running_max = float(stats.get("max", "0"))
        running_sum = float(stats.get("sum", "0"))
        running_count = int(stats.get("count", "0"))
        running_sq_sum = float(stats.get("sq_sum", "0"))

        # Z-score
        amount_zscore = 0.0
        if running_count > 1:
            variance = (running_sq_sum / running_count) - (running_avg ** 2)
            stddev = math.sqrt(max(variance, 0))
            if stddev > 0:
                amount_zscore = (amount - running_avg) / stddev

        amount_to_avg_ratio = (amount / running_avg) if running_avg > 0 else 0.0

        # Geo — distance from last
        distance_from_last = 0.0
        distance_from_home = 0.0
        if transaction.latitude is not None and transaction.longitude is not None:
            last_lat = float(geo.get("last_lat", "0"))
            last_lon = float(geo.get("last_lon", "0"))
            if last_lat != 0.0 or last_lon != 0.0:
                distance_from_last = _haversine(last_lat, last_lon, transaction.latitude, transaction.longitude)
            home_lat = float(geo.get("home_lat", "0"))
            home_lon = float(geo.get("home_lon", "0"))
            if home_lat != 0.0 or home_lon != 0.0:
                distance_from_home = _haversine(home_lat, home_lon, transaction.latitude, transaction.longitude)

        # Compute windowed averages and maxes
        def _window_avg(window: str) -> Decimal:
            c = velocity_counts.get(window, 0)
            return velocity_amounts.get(window, Decimal("0")) / c if c else Decimal("0")

        return FeatureVector(
            transaction_id=transaction.transaction_id,
            user_id=user_id,
            timestamp=transaction.timestamp,
            txn_count_1h=velocity_counts.get("1h", 0),
            txn_count_24h=velocity_counts.get("24h", 0),
            txn_amount_sum_1h=velocity_amounts.get("1h", Decimal("0")),
            txn_amount_sum_24h=velocity_amounts.get("24h", Decimal("0")),
            txn_amount_avg_24h=_window_avg("24h"),
            unique_merchants_24h=unique_merchants_24h,
            unique_countries_24h=0,  # requires IP-geo lookup — not implemented here
            max_amount_24h=Decimal(str(running_max)),
            time_since_last_txn_seconds=time_since_last,
            distance_from_last_txn_km=distance_from_last,
            is_new_device=is_new_device,
            is_new_merchant=is_new_merchant,
            amount_zscore=amount_zscore,
            amount_to_avg_ratio=amount_to_avg_ratio,
        )

    # ------------------------------------------------------------------
    # Private helpers — each appends commands to an open pipeline
    # ------------------------------------------------------------------

    async def _update_velocity(
        self,
        pipe: aioredis.client.Pipeline,
        user_id: str,
        amount: float,
        now_ts: float,
    ) -> None:
        """Add a timestamped entry to the velocity sorted set and prune old entries."""
        key = f"user:{user_id}:velocity"
        member = f"{now_ts}:{amount}"
        pipe.zadd(key, {member: now_ts})
        # Remove entries older than 24h
        pipe.zremrangebyscore(key, "-inf", now_ts - 86400)

    async def _update_amount_stats(
        self,
        pipe: aioredis.client.Pipeline,
        user_id: str,
        amount: float,
    ) -> None:
        """Update running amount statistics using a Redis hash.

        Maintains: count, sum, avg, max, sq_sum (for variance/z-score).
        """
        key = f"user:{user_id}:stats"
        pipe.hincrbyfloat(key, "sum", amount)
        pipe.hincrbyfloat(key, "sq_sum", amount * amount)
        pipe.hincrby(key, "count", 1)
        # We store a Lua-free running average by reading after pipeline.
        # Instead, we use a Lua script embedded in the pipeline for atomicity.
        lua_script = """
        local key = KEYS[1]
        local amount = tonumber(ARGV[1])
        local current_max = tonumber(redis.call('HGET', key, 'max') or '0')
        if amount > current_max then
            redis.call('HSET', key, 'max', tostring(amount))
        end
        local sum = tonumber(redis.call('HGET', key, 'sum') or '0')
        local count = tonumber(redis.call('HGET', key, 'count') or '0')
        if count > 0 then
            redis.call('HSET', key, 'avg', tostring(sum / count))
        end
        return 1
        """
        pipe.eval(lua_script, 1, key, str(amount))

    async def _update_geo(
        self,
        pipe: aioredis.client.Pipeline,
        user_id: str,
        latitude: float | None,
        longitude: float | None,
        now_ts: float,
    ) -> None:
        """Store latest geo coordinates and update the home location estimate."""
        if latitude is None or longitude is None:
            return

        key = f"user:{user_id}:geo"
        # Update home location as exponential moving average (Lua for atomicity)
        lua_script = """
        local key = KEYS[1]
        local lat = tonumber(ARGV[1])
        local lon = tonumber(ARGV[2])
        local alpha = 0.05

        local home_lat = tonumber(redis.call('HGET', key, 'home_lat') or '0')
        local home_lon = tonumber(redis.call('HGET', key, 'home_lon') or '0')

        if home_lat == 0 and home_lon == 0 then
            redis.call('HSET', key, 'home_lat', tostring(lat))
            redis.call('HSET', key, 'home_lon', tostring(lon))
        else
            local new_home_lat = home_lat * (1 - alpha) + lat * alpha
            local new_home_lon = home_lon * (1 - alpha) + lon * alpha
            redis.call('HSET', key, 'home_lat', tostring(new_home_lat))
            redis.call('HSET', key, 'home_lon', tostring(new_home_lon))
        end

        redis.call('HSET', key, 'last_lat', tostring(lat))
        redis.call('HSET', key, 'last_lon', tostring(lon))
        return 1
        """
        pipe.eval(lua_script, 1, key, str(latitude), str(longitude))

    async def _update_device(
        self,
        pipe: aioredis.client.Pipeline,
        user_id: str,
        device_fingerprint: str | None,
    ) -> None:
        """Track unique device fingerprints in a set."""
        if device_fingerprint is None:
            return
        key = f"user:{user_id}:devices"
        pipe.sadd(key, device_fingerprint)

    async def _update_merchant(
        self,
        pipe: aioredis.client.Pipeline,
        user_id: str,
        merchant_id: str | None,
        now_ts: float,
    ) -> None:
        """Track unique merchants in a 7-day sliding window sorted set."""
        if merchant_id is None:
            return
        key = f"user:{user_id}:merchants_7d"
        pipe.zadd(key, {merchant_id: now_ts})
        pipe.zremrangebyscore(key, "-inf", now_ts - _SEVEN_DAYS)
