"""Token bucket rate limiter for the scoring API."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import structlog
from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting a single client."""

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = self.capacity

    def consume(self) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def retry_after(self) -> float:
        """Seconds until the next token is available."""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.refill_rate


class RateLimiter:
    """In-memory rate limiter using token buckets per client key.

    Args:
        default_rpm: Default requests per minute for unauthenticated clients.
        burst_multiplier: Burst capacity as a multiplier of the per-second rate.
    """

    def __init__(self, default_rpm: int = 60, burst_multiplier: float = 2.0) -> None:
        self._default_rpm = default_rpm
        self._burst_multiplier = burst_multiplier
        self._buckets: dict[str, TokenBucket] = defaultdict(self._make_default_bucket)

    def _make_default_bucket(self) -> TokenBucket:
        rps = self._default_rpm / 60.0
        return TokenBucket(capacity=rps * self._burst_multiplier, refill_rate=rps)

    def get_bucket(self, key: str, rpm: int | None = None) -> TokenBucket:
        """Get or create a token bucket for a client key.

        Args:
            key: Client identifier (API key name or IP).
            rpm: Custom RPM limit for this key.

        Returns:
            The client's TokenBucket.
        """
        if key not in self._buckets:
            effective_rpm = rpm or self._default_rpm
            rps = effective_rpm / 60.0
            self._buckets[key] = TokenBucket(
                capacity=rps * self._burst_multiplier,
                refill_rate=rps,
            )
        return self._buckets[key]

    def check(self, key: str, rpm: int | None = None) -> tuple[bool, float]:
        """Check if a request is allowed.

        Args:
            key: Client identifier.
            rpm: Optional custom RPM.

        Returns:
            Tuple of (allowed, retry_after_seconds).
        """
        bucket = self.get_bucket(key, rpm)
        allowed = bucket.consume()
        return allowed, bucket.retry_after


# Global rate limiter instance
_rate_limiter = RateLimiter(default_rpm=600)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces per-client rate limiting.

    Uses API key name if authenticated, otherwise falls back to client IP.
    """

    SKIP_PATHS = {"/health", "/health/ready", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Determine client key and rate limit
        api_key_name = getattr(request.state, "api_key_name", None)
        api_key_rpm = getattr(request.state, "api_key_rate_limit", None)

        if api_key_name and api_key_name not in ("internal", "dev-mode"):
            client_key = f"key:{api_key_name}"
            rpm = api_key_rpm
        else:
            client_ip = request.client.host if request.client else "unknown"
            client_key = f"ip:{client_ip}"
            rpm = None

        allowed, retry_after = _rate_limiter.check(client_key, rpm)

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                client_key=client_key,
                retry_after=round(retry_after, 2),
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        response = await call_next(request)
        return response
