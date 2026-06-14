"""Adaptive rate limiter with token bucket algorithm and burst detection.

Uses sliding window counters in Redis with per-tenant rate limits.
Automatically adjusts limits based on traffic patterns (burst detection)
and applies penalties for suspicious IPs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)

# Lua script: token bucket with sliding window
# KEYS[1] = bucket key   KEYS[2] = window key
# ARGV[1] = max_tokens  ARGV[2] = refill_rate_per_sec  ARGV[3] = now_ms
# ARGV[4] = window_ms
_TOKEN_BUCKET_SCRIPT = """
local bucket_key  = KEYS[1]
local window_key  = KEYS[2]
local max_tokens  = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now         = tonumber(ARGV[3])
local window_ms   = tonumber(ARGV[4])

-- Token bucket logic
local bucket = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
local tokens     = tonumber(bucket[1])
local last_refill = tonumber(bucket[2])

if tokens == nil then
    tokens = max_tokens
    last_refill = now
end

-- Refill tokens based on elapsed time
local elapsed_ms = now - last_refill
local refill = (elapsed_ms / 1000.0) * refill_rate
tokens = math.min(max_tokens, tokens + refill)

-- Sliding window: record request timestamp
local window_start = now - window_ms
redis.call('ZREMRANGEBYSCORE', window_key, '-inf', window_start)
local window_count = redis.call('ZCARD', window_key)

if tokens >= 1 then
    tokens = tokens - 1
    redis.call('HMSET', bucket_key, 'tokens', tostring(tokens), 'last_refill', tostring(now))
    redis.call('PEXPIRE', bucket_key, math.ceil(window_ms / 1000) + 10)

    redis.call('ZADD', window_key, now, now .. '-' .. math.random(1000000))
    redis.call('PEXPIRE', window_key, window_ms + 1000)

    return {1, math.floor(tokens), window_count + 1}
else
    -- Denied: calculate when the next token will be available
    local deficit = 1 - tokens
    local wait_ms = math.ceil((deficit / refill_rate) * 1000)
    redis.call('HMSET', bucket_key, 'tokens', tostring(tokens), 'last_refill', tostring(now))
    redis.call('PEXPIRE', bucket_key, math.ceil(window_ms / 1000) + 10)

    return {0, wait_ms, window_count}
end
"""

# Lua script: get burst metrics from the sliding window
_BURST_CHECK_SCRIPT = """
local window_key = KEYS[1]
local now        = tonumber(ARGV[1])
local short_ms   = tonumber(ARGV[2])
local long_ms    = tonumber(ARGV[3])

local short_start = now - short_ms
local long_start  = now - long_ms

redis.call('ZREMRANGEBYSCORE', window_key, '-inf', long_start)

local short_count = redis.call('ZCOUNT', window_key, short_start, '+inf')
local long_count  = redis.call('ZCARD', window_key)

return {short_count, long_count}
"""


@dataclass
class TenantConfig:
    """Rate limit configuration for a single tenant."""

    max_tokens: int = 100
    refill_rate: float = 50.0  # tokens per second
    window_seconds: int = 60
    burst_threshold_ratio: float = 3.0
    min_tokens_floor: int = 10
    penalty_multiplier: float = 0.5


@dataclass
class BurstInfo:
    """Burst detection result."""

    is_burst: bool = False
    short_window_count: int = 0
    long_window_count: int = 0
    ratio: float = 0.0


@dataclass
class _TenantState:
    """Internal mutable state for a tenant's adaptive limits."""

    effective_max_tokens: int = 100
    effective_refill_rate: float = 50.0
    last_adjustment_time: float = 0.0
    consecutive_bursts: int = 0


class AdaptiveRateLimiter:
    """Adaptive rate limiter with token bucket algorithm and burst detection.

    Features:
        - Token bucket algorithm backed by Redis for distributed rate limiting
        - Sliding window counters for accurate request tracking
        - Per-tenant configurable rate limits
        - Automatic limit adjustment based on detected traffic bursts
        - IP-based penalty system for suspicious sources

    Args:
        redis: Async Redis client.
        key_prefix: Redis key prefix for namespace isolation.
        default_config: Default tenant config for unconfigured tenants.
        burst_short_window_seconds: Short window for burst detection.
        burst_long_window_seconds: Long window for burst detection baseline.
        adjustment_cooldown_seconds: Minimum seconds between adaptive adjustments.
        recovery_rate: Rate at which limits recover after burst subsides (0-1).
    """

    def __init__(
        self,
        redis: Redis,
        key_prefix: str = "arl:",
        default_config: TenantConfig | None = None,
        burst_short_window_seconds: int = 5,
        burst_long_window_seconds: int = 60,
        adjustment_cooldown_seconds: float = 30.0,
        recovery_rate: float = 0.1,
    ) -> None:
        self._redis = redis
        self._key_prefix = key_prefix
        self._default_config = default_config or TenantConfig()
        self._burst_short_ms = burst_short_window_seconds * 1000
        self._burst_long_ms = burst_long_window_seconds * 1000
        self._adjustment_cooldown = adjustment_cooldown_seconds
        self._recovery_rate = recovery_rate

        self._tenant_configs: dict[str, TenantConfig] = {}
        self._tenant_states: dict[str, _TenantState] = {}
        self._suspicious_ips: dict[str, float] = {}  # ip -> penalty_factor (0-1)

        self._bucket_script_sha: str | None = None
        self._burst_script_sha: str | None = None

    # -- Configuration -------------------------------------------------------

    def configure_tenant(self, tenant_id: str, config: TenantConfig) -> None:
        """Set or update rate limit configuration for a tenant."""
        self._tenant_configs[tenant_id] = config
        if tenant_id not in self._tenant_states:
            self._tenant_states[tenant_id] = _TenantState(
                effective_max_tokens=config.max_tokens,
                effective_refill_rate=config.refill_rate,
            )

    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        """Return the config for a tenant, falling back to the default."""
        return self._tenant_configs.get(tenant_id, self._default_config)

    def mark_ip_suspicious(self, ip: str, penalty_factor: float = 0.5) -> None:
        """Mark an IP as suspicious with a penalty factor.

        Args:
            ip: The IP address to penalise.
            penalty_factor: Multiplier applied to the tenant limit (0-1).
                0.0 means fully blocked, 1.0 means no penalty.
        """
        self._suspicious_ips[ip] = max(0.0, min(1.0, penalty_factor))

    def clear_ip_suspicion(self, ip: str) -> None:
        """Remove suspicion flag from an IP."""
        self._suspicious_ips.pop(ip, None)

    def is_ip_suspicious(self, ip: str) -> bool:
        """Check whether an IP is marked as suspicious."""
        return ip in self._suspicious_ips

    def get_ip_penalty(self, ip: str) -> float:
        """Return the penalty factor for an IP (1.0 if not suspicious)."""
        return self._suspicious_ips.get(ip, 1.0)

    # -- Script loading ------------------------------------------------------

    async def _load_scripts(self) -> tuple[str, str]:
        if self._bucket_script_sha is None:
            self._bucket_script_sha = await self._redis.script_load(
                _TOKEN_BUCKET_SCRIPT
            )
        if self._burst_script_sha is None:
            self._burst_script_sha = await self._redis.script_load(
                _BURST_CHECK_SCRIPT
            )
        return self._bucket_script_sha, self._burst_script_sha

    # -- Core API ------------------------------------------------------------

    async def check(
        self,
        tenant_id: str,
        client_ip: str | None = None,
    ) -> tuple[bool, dict]:
        """Check whether a request should be allowed.

        Args:
            tenant_id: The tenant identifier.
            client_ip: Optional client IP for penalty lookup.

        Returns:
            (allowed, metadata) where metadata includes remaining tokens,
            window count, and retry-after (ms) if denied.
        """
        config = self.get_tenant_config(tenant_id)
        state = self._tenant_states.setdefault(
            tenant_id,
            _TenantState(
                effective_max_tokens=config.max_tokens,
                effective_refill_rate=config.refill_rate,
            ),
        )

        effective_max = state.effective_max_tokens
        effective_rate = state.effective_refill_rate

        # Apply IP penalty
        penalty = 1.0
        if client_ip and client_ip in self._suspicious_ips:
            penalty = self._suspicious_ips[client_ip]
            effective_max = max(
                config.min_tokens_floor,
                int(effective_max * penalty),
            )
            effective_rate = max(1.0, effective_rate * penalty)

        bucket_key = f"{self._key_prefix}bucket:{tenant_id}"
        window_key = f"{self._key_prefix}window:{tenant_id}"
        window_ms = config.window_seconds * 1000

        if client_ip and client_ip in self._suspicious_ips:
            bucket_key = f"{self._key_prefix}bucket:{tenant_id}:ip:{client_ip}"
            window_key = f"{self._key_prefix}window:{tenant_id}:ip:{client_ip}"

        now_ms = int(time.time() * 1000)

        try:
            bucket_sha, _ = await self._load_scripts()
            result = await self._redis.evalsha(
                bucket_sha,
                2,
                bucket_key,
                window_key,
                str(effective_max),
                str(effective_rate),
                str(now_ms),
                str(window_ms),
            )

            allowed = result[0] == 1
            second_val = int(result[1])
            window_count = int(result[2])

            metadata: dict = {
                "tenant_id": tenant_id,
                "window_count": window_count,
                "effective_limit": effective_max,
                "effective_rate": effective_rate,
                "penalty_factor": penalty,
            }

            if allowed:
                metadata["remaining"] = second_val
            else:
                metadata["retry_after_ms"] = second_val

            return allowed, metadata

        except Exception as exc:
            await logger.awarning(
                "adaptive_rate_limiter_error",
                tenant_id=tenant_id,
                error=str(exc),
            )
            # Fail open
            return True, {"tenant_id": tenant_id, "error": str(exc), "fallback": True}

    # -- Burst detection -----------------------------------------------------

    async def detect_burst(self, tenant_id: str) -> BurstInfo:
        """Analyse recent traffic for burst patterns.

        Compares short-window request count to the expected average rate
        derived from the long window.

        Returns:
            BurstInfo with detection results.
        """
        config = self.get_tenant_config(tenant_id)
        window_key = f"{self._key_prefix}window:{tenant_id}"
        now_ms = int(time.time() * 1000)

        try:
            _, burst_sha = await self._load_scripts()
            result = await self._redis.evalsha(
                burst_sha,
                1,
                window_key,
                str(now_ms),
                str(self._burst_short_ms),
                str(self._burst_long_ms),
            )

            short_count = int(result[0])
            long_count = int(result[1])

            # Compute expected rate: requests per ms in long window, projected to short window
            if long_count > 0 and self._burst_long_ms > 0:
                long_rate = long_count / self._burst_long_ms
                expected_short = long_rate * self._burst_short_ms
                ratio = short_count / max(expected_short, 1.0)
            else:
                ratio = 0.0

            is_burst = ratio >= config.burst_threshold_ratio

            return BurstInfo(
                is_burst=is_burst,
                short_window_count=short_count,
                long_window_count=long_count,
                ratio=round(ratio, 2),
            )

        except Exception as exc:
            await logger.awarning(
                "burst_detection_error",
                tenant_id=tenant_id,
                error=str(exc),
            )
            return BurstInfo()

    # -- Adaptive adjustment -------------------------------------------------

    async def adjust_limits(self, tenant_id: str) -> dict:
        """Auto-adjust rate limits based on traffic patterns.

        When a burst is detected, limits are reduced. When traffic normalises,
        limits gradually recover toward the configured baseline.

        Returns:
            Dict with adjustment details.
        """
        config = self.get_tenant_config(tenant_id)
        state = self._tenant_states.setdefault(
            tenant_id,
            _TenantState(
                effective_max_tokens=config.max_tokens,
                effective_refill_rate=config.refill_rate,
            ),
        )

        now = time.monotonic()
        if (now - state.last_adjustment_time) < self._adjustment_cooldown:
            return {
                "adjusted": False,
                "reason": "cooldown",
                "effective_max_tokens": state.effective_max_tokens,
                "effective_refill_rate": state.effective_refill_rate,
            }

        burst = await self.detect_burst(tenant_id)
        state.last_adjustment_time = now

        if burst.is_burst:
            state.consecutive_bursts += 1
            # Reduce limits: more consecutive bursts = harsher penalty
            reduction = config.penalty_multiplier ** state.consecutive_bursts
            new_max = max(
                config.min_tokens_floor,
                int(config.max_tokens * reduction),
            )
            new_rate = max(1.0, config.refill_rate * reduction)

            state.effective_max_tokens = new_max
            state.effective_refill_rate = new_rate

            await logger.awarning(
                "adaptive_rate_limit_reduced",
                tenant_id=tenant_id,
                burst_ratio=burst.ratio,
                consecutive_bursts=state.consecutive_bursts,
                new_max_tokens=new_max,
                new_refill_rate=new_rate,
            )

            return {
                "adjusted": True,
                "reason": "burst_detected",
                "burst_ratio": burst.ratio,
                "consecutive_bursts": state.consecutive_bursts,
                "effective_max_tokens": new_max,
                "effective_refill_rate": new_rate,
            }
        else:
            # Recover toward baseline
            if state.consecutive_bursts > 0:
                state.consecutive_bursts = max(0, state.consecutive_bursts - 1)

            recovery_max = state.effective_max_tokens + int(
                (config.max_tokens - state.effective_max_tokens) * self._recovery_rate
            )
            recovery_rate = state.effective_refill_rate + (
                (config.refill_rate - state.effective_refill_rate) * self._recovery_rate
            )

            state.effective_max_tokens = min(config.max_tokens, recovery_max)
            state.effective_refill_rate = min(config.refill_rate, recovery_rate)

            return {
                "adjusted": True,
                "reason": "recovery",
                "consecutive_bursts": state.consecutive_bursts,
                "effective_max_tokens": state.effective_max_tokens,
                "effective_refill_rate": state.effective_refill_rate,
            }

    # -- Introspection -------------------------------------------------------

    async def get_tenant_usage(self, tenant_id: str) -> dict:
        """Return current usage stats for a tenant."""
        window_key = f"{self._key_prefix}window:{tenant_id}"
        now_ms = int(time.time() * 1000)
        config = self.get_tenant_config(tenant_id)
        state = self._tenant_states.get(tenant_id)

        try:
            window_start = now_ms - (config.window_seconds * 1000)
            count = await self._redis.zcount(window_key, window_start, "+inf")
            return {
                "tenant_id": tenant_id,
                "current_count": int(count),
                "window_seconds": config.window_seconds,
                "effective_max_tokens": state.effective_max_tokens if state else config.max_tokens,
                "effective_refill_rate": state.effective_refill_rate if state else config.refill_rate,
            }
        except Exception:
            return {
                "tenant_id": tenant_id,
                "current_count": 0,
                "window_seconds": config.window_seconds,
                "effective_max_tokens": config.max_tokens,
                "effective_refill_rate": config.refill_rate,
            }

    def stats(self) -> dict:
        """Return summary stats across all tenants."""
        return {
            "configured_tenants": len(self._tenant_configs),
            "active_tenants": len(self._tenant_states),
            "suspicious_ips": len(self._suspicious_ips),
            "tenant_states": {
                tid: {
                    "effective_max_tokens": s.effective_max_tokens,
                    "effective_refill_rate": s.effective_refill_rate,
                    "consecutive_bursts": s.consecutive_bursts,
                }
                for tid, s in self._tenant_states.items()
            },
        }
