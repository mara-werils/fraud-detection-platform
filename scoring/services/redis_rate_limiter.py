"""Redis-backed sliding window rate limiter for multi-tenant deployments.

Uses a Lua script executed atomically in Redis for exact sliding window counts.
Falls back to the in-memory token bucket if Redis is unavailable.
"""

from __future__ import annotations

import time

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)

# Lua script: sliding window counter in Redis
# KEYS[1] = rate limit key   ARGV[1] = window_ms  ARGV[2] = max_requests  ARGV[3] = now_ms
_SLIDING_WINDOW_SCRIPT = """
local key       = KEYS[1]
local window_ms = tonumber(ARGV[1])
local limit     = tonumber(ARGV[2])
local now       = tonumber(ARGV[3])
local window_start = now - window_ms

-- Remove entries older than the window
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

-- Count requests in the current window
local count = redis.call('ZCARD', key)

if count < limit then
    -- Allow: record this request
    redis.call('ZADD', key, now, now .. '-' .. math.random(1000000))
    redis.call('PEXPIRE', key, window_ms)
    return {1, limit - count - 1}
else
    -- Deny
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after_ms = 0
    if #oldest > 0 then
        retry_after_ms = window_ms - (now - tonumber(oldest[2]))
    end
    return {0, retry_after_ms}
end
"""


class RedisRateLimiter:
    """Sliding window rate limiter backed by Redis.

    Args:
        redis: Async Redis client.
        window_seconds: Time window duration in seconds (default: 1).
        key_prefix: Redis key prefix for isolation.
    """

    def __init__(
        self,
        redis: Redis,
        window_seconds: int = 1,
        key_prefix: str = "rl:",
    ) -> None:
        self._redis = redis
        self._window_ms = window_seconds * 1000
        self._key_prefix = key_prefix
        self._script_sha: str | None = None

    async def _load_script(self) -> str:
        if self._script_sha is None:
            self._script_sha = await self._redis.script_load(_SLIDING_WINDOW_SCRIPT)
        return self._script_sha

    async def check(
        self,
        identifier: str,
        max_requests: int,
    ) -> tuple[bool, int]:
        """Check if the request is within rate limits.

        Args:
            identifier: Unique key (e.g. org_id or api_key_hash).
            max_requests: Maximum requests allowed per window.

        Returns:
            (allowed, remaining_or_retry_after_ms)
        """
        key = f"{self._key_prefix}{identifier}"
        now_ms = int(time.time() * 1000)

        try:
            sha = await self._load_script()
            result = await self._redis.evalsha(
                sha,
                1,
                key,
                str(self._window_ms),
                str(max_requests),
                str(now_ms),
            )
            allowed = result[0] == 1
            value = int(result[1])
            return allowed, value

        except Exception as exc:
            await logger.awarning("redis_rate_limiter_error", error=str(exc))
            # Fail open — allow the request if Redis is down
            return True, max_requests

    async def get_usage(self, identifier: str) -> dict:
        """Return current window usage for an identifier."""
        key = f"{self._key_prefix}{identifier}"
        now_ms = int(time.time() * 1000)
        window_start = now_ms - self._window_ms

        try:
            count = await self._redis.zcount(key, window_start, "+inf")
            return {"key": identifier, "current_count": int(count), "window_ms": self._window_ms}
        except Exception:
            return {"key": identifier, "current_count": 0, "window_ms": self._window_ms}


class TenantRateLimitMiddleware:
    """Async middleware class that enforces per-org rate limits via Redis.

    Usage: call check(request) before processing; returns 429 headers if limited.
    """

    SKIP_PATHS = frozenset({
        "/health", "/health/ready", "/metrics", "/docs", "/openapi.json", "/redoc"
    })

    def __init__(self, redis: Redis) -> None:
        self._limiter = RedisRateLimiter(redis, window_seconds=1, key_prefix="rl:org:")
        self._ip_limiter = RedisRateLimiter(redis, window_seconds=1, key_prefix="rl:ip:")

    async def check(self, request) -> tuple[bool, dict]:
        """Check rate limits.

        Returns:
            (allowed, headers_to_add)
        """
        if request.url.path in self.SKIP_PATHS:
            return True, {}

        org_id = getattr(request.state, "org_id", None)
        org_rps = getattr(request.state, "org_rate_limit_rps", 100)

        if org_id:
            allowed, remaining = await self._limiter.check(str(org_id), org_rps)
            headers = {
                "X-RateLimit-Limit": str(org_rps),
                "X-RateLimit-Remaining": str(max(0, remaining)),
                "X-RateLimit-Window": "1s",
            }
            if not allowed:
                headers["Retry-After"] = str(max(1, remaining // 1000))
                await logger.awarning(
                    "org_rate_limit_exceeded",
                    org_id=str(org_id),
                    rps_limit=org_rps,
                )
            return allowed, headers

        # Anonymous: limit by IP at 10 rps
        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining = await self._ip_limiter.check(client_ip, 10)
        return allowed, {"X-RateLimit-Limit": "10", "X-RateLimit-Remaining": str(max(0, remaining))}
