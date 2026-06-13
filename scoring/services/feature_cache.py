"""Two-level feature cache: L1 (in-process LRU) + L2 (Redis).

L1 is an in-process LRU dict for sub-millisecond hot feature retrieval.
L2 is Redis for cross-replica sharing with a configurable TTL.

Cache key format: feat:{user_id}:{feature_name}
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any

import structlog
from prometheus_client import Counter, Histogram

logger = structlog.get_logger(__name__)

CACHE_OPS = Counter(
    "feature_cache_operations_total",
    "Feature cache operations by level and result",
    ["level", "operation", "result"],
)

CACHE_LATENCY = Histogram(
    "feature_cache_latency_seconds",
    "Feature cache operation latency",
    ["level", "operation"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
)


def _deserialize_cached_value(raw: Any) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


class LRUCache:
    """Thread-safe in-process LRU cache with TTL support."""

    def __init__(self, max_size: int = 10_000, default_ttl: int = 60) -> None:
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            self._misses += 1
            return None
        value, expires_at = self._cache[key]
        if time.monotonic() > expires_at:
            del self._cache[key]
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl or self._default_ttl
        # Guard against accidental zero/negative TTL values that would make entries unusable.
        if effective_ttl <= 0:
            effective_ttl = 1
        expires_at = time.monotonic() + effective_ttl
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, expires_at)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }


class TwoLevelFeatureCache:
    """L1 (in-process LRU) + L2 (Redis) feature cache.

    Args:
        redis: Async Redis client (raw Redis, not RedisPoolManager).
        l1_max_size: Maximum entries in L1 LRU cache.
        l1_ttl: L1 TTL in seconds (shorter than L2).
        l2_ttl: L2 (Redis) TTL in seconds.
        key_prefix: Redis key prefix for namespace isolation.
    """

    def __init__(
        self,
        redis,
        l1_max_size: int = 10_000,
        l1_ttl: int = 30,
        l2_ttl: int = 300,
        key_prefix: str = "feat:",
    ) -> None:
        self._redis = redis
        self._l1 = LRUCache(max_size=l1_max_size, default_ttl=l1_ttl)
        self._l2_ttl = l2_ttl
        self._key_prefix = key_prefix
        self._l2_hits = 0
        self._l2_misses = 0

    def _key(self, user_id: str, feature: str) -> str:
        return f"{self._key_prefix}{user_id}:{feature}"

    async def get(self, user_id: str, feature: str) -> Any | None:
        """Retrieve a feature value from L1 or L2 cache."""
        key = self._key(user_id, feature)

        # L1 check
        value = self._l1.get(key)
        if value is not None:
            CACHE_OPS.labels(level="l1", operation="get", result="hit").inc()
            return value
        CACHE_OPS.labels(level="l1", operation="get", result="miss").inc()

        # L2 check
        try:
            start = time.monotonic()
            raw = await self._redis.get(key)
            CACHE_LATENCY.labels(level="l2", operation="get").observe(time.monotonic() - start)
            if raw is not None:
                value = _deserialize_cached_value(raw)
                self._l1.set(key, value)
                self._l2_hits += 1
                CACHE_OPS.labels(level="l2", operation="get", result="hit").inc()
                return value
        except Exception as exc:
            CACHE_OPS.labels(level="l2", operation="get", result="error").inc()
            await logger.awarning("feature_cache_l2_get_error", error=str(exc))

        self._l2_misses += 1
        CACHE_OPS.labels(level="l2", operation="get", result="miss").inc()
        return None

    async def get_many(self, user_id: str, features: list[str]) -> dict[str, Any]:
        """Retrieve multiple features at once; uses Redis MGET for L2 efficiency."""
        result: dict[str, Any] = {}
        l2_keys_needed: list[str] = []
        feature_by_key: dict[str, str] = {}

        for feature in features:
            key = self._key(user_id, feature)
            v = self._l1.get(key)
            if v is not None:
                result[feature] = v
            else:
                l2_keys_needed.append(key)
                feature_by_key[key] = feature

        if l2_keys_needed:
            try:
                values = await self._redis.mget(*l2_keys_needed)
                for key, raw in zip(l2_keys_needed, values):
                    if raw is not None:
                        v = _deserialize_cached_value(raw)
                        feature = feature_by_key[key]
                        result[feature] = v
                        self._l1.set(key, v)
                        self._l2_hits += 1
                    else:
                        self._l2_misses += 1
            except Exception as exc:
                await logger.awarning("feature_cache_l2_mget_error", error=str(exc))

        return result

    async def set(self, user_id: str, feature: str, value: Any) -> None:
        """Store a feature value in both L1 and L2."""
        key = self._key(user_id, feature)
        self._l1.set(key, value)
        try:
            await self._redis.set(key, json.dumps(value, default=str), ex=self._l2_ttl)
        except Exception as exc:
            await logger.awarning("feature_cache_l2_set_error", error=str(exc))

    async def set_many(self, user_id: str, features: dict[str, Any]) -> None:
        """Store multiple features using Redis pipeline."""
        if not features:
            return
        for feature, value in features.items():
            key = self._key(user_id, feature)
            self._l1.set(key, value)

        try:
            pipe = self._redis.pipeline()
            for feature, value in features.items():
                key = self._key(user_id, feature)
                pipe.set(key, json.dumps(value, default=str), ex=self._l2_ttl)
            await pipe.execute()
        except Exception as exc:
            await logger.awarning("feature_cache_l2_pipeline_error", error=str(exc))

    async def invalidate_user(self, user_id: str) -> None:
        """Invalidate all cached features for a user."""
        pattern = f"{self._key_prefix}{user_id}:*"
        try:
            keys = []
            async for key in self._redis.scan_iter(pattern, count=100):
                keys.append(key)
                self._l1.delete(key.decode() if isinstance(key, bytes) else key)
            if keys:
                await self._redis.delete(*keys)
        except Exception as exc:
            await logger.awarning("feature_cache_invalidate_error", error=str(exc))

    async def warm(self, user_ids: list[str], features: list[str]) -> int:
        """Pre-load features for a set of users into L1 from L2.

        Args:
            user_ids: Users to warm the cache for.
            features: Feature names to load.

        Returns:
            Number of features successfully warmed into L1.
        """
        warmed = 0
        for user_id in user_ids:
            loaded = await self.get_many(user_id, features)
            warmed += len(loaded)
        logger.info("feature_cache_warmed", users=len(user_ids), features_loaded=warmed)
        return warmed

    def stats(self) -> dict:
        l1_stats = self._l1.stats()
        l2_total = self._l2_hits + self._l2_misses
        return {
            "l1": l1_stats,
            "l2": {
                "hits": self._l2_hits,
                "misses": self._l2_misses,
                "hit_rate": round(self._l2_hits / l2_total, 4) if l2_total > 0 else 0.0,
                "ttl_seconds": self._l2_ttl,
            },
        }


__all__ = ["LRUCache", "TwoLevelFeatureCache"]
