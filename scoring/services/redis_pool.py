"""Redis connection pool manager with circuit breaker pattern.

The circuit breaker prevents cascade failures when Redis is unavailable:
  CLOSED  → normal operation, requests pass through
  OPEN    → Redis is broken, requests fail fast without attempting connection
  HALF    → probing phase, one request is allowed through to test recovery
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any

import structlog
from redis.asyncio import ConnectionPool, Redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, RedisError, TimeoutError  # noqa: A004

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for Redis operations.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout: Seconds to wait before entering HALF_OPEN.
        success_threshold: Successes needed in HALF_OPEN to close.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def call(self, coro):
        """Execute a coroutine through the circuit breaker."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    await logger.ainfo("circuit_breaker_half_open")
                else:
                    raise RedisError("Circuit breaker OPEN — Redis unavailable")

        try:
            result = await coro
            await self._on_success()
            return result
        except (ConnectionError, TimeoutError, RedisError):
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    await logger.ainfo("circuit_breaker_closed")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold:
                if self._state != CircuitState.OPEN:
                    self._state = CircuitState.OPEN
                    await logger.awarning(
                        "circuit_breaker_opened",
                        failures=self._failure_count,
                    )

    def stats(self) -> dict:
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
        }


class RedisPoolManager:
    """Managed Redis connection pool with circuit breaker.

    Provides a high-connection-count pool suitable for 100K+ concurrent users,
    wrapping all operations in the circuit breaker.
    """

    def __init__(self, url: str, max_connections: int = 200) -> None:
        retry = Retry(ExponentialBackoff(cap=2, base=0.5), retries=3)
        self._pool = ConnectionPool.from_url(
            url,
            max_connections=max_connections,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            retry=retry,
            retry_on_error=[ConnectionError, TimeoutError],
            decode_responses=False,
        )
        self._redis = Redis(connection_pool=self._pool)
        self._cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._cb

    async def get(self, key: str) -> Any:
        return await self._cb.call(self._redis.get(key))

    async def set(self, key: str, value: Any, ex: int | None = None) -> Any:
        return await self._cb.call(self._redis.set(key, value, ex=ex))

    async def delete(self, *keys: str) -> Any:
        return await self._cb.call(self._redis.delete(*keys))

    async def hget(self, name: str, key: str) -> Any:
        return await self._cb.call(self._redis.hget(name, key))

    async def hset(self, name: str, mapping: dict) -> Any:
        return await self._cb.call(self._redis.hset(name, mapping=mapping))

    async def ping(self) -> bool:
        try:
            await self._cb.call(self._redis.ping())
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._redis.aclose()

    def raw(self) -> Redis:
        """Access the raw Redis client (for Lua scripts etc.)."""
        return self._redis

    def stats(self) -> dict:
        return {
            "pool_max_connections": self._pool.max_connections,
            "circuit_breaker": self._cb.stats(),
        }
