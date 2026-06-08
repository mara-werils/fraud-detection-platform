"""Tests for the adaptive rate limiter with burst detection."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from scoring.services.adaptive_rate_limiter import (
    AdaptiveRateLimiter,
    BurstInfo,
    TenantConfig,
    _TenantState,
)


@pytest.fixture
def mock_redis() -> AsyncMock:
    redis = AsyncMock()
    redis.script_load = AsyncMock(side_effect=["bucket_sha", "burst_sha"])
    redis.evalsha = AsyncMock(return_value=[1, 99, 1])
    redis.zcount = AsyncMock(return_value=5)
    return redis


@pytest.fixture
def limiter(mock_redis: AsyncMock) -> AdaptiveRateLimiter:
    return AdaptiveRateLimiter(
        redis=mock_redis,
        key_prefix="test_arl:",
        adjustment_cooldown_seconds=0.0,
    )


# -- TenantConfig defaults ---------------------------------------------------


class TestTenantConfig:
    def test_default_values(self) -> None:
        cfg = TenantConfig()
        assert cfg.max_tokens == 100
        assert cfg.refill_rate == 50.0
        assert cfg.window_seconds == 60
        assert cfg.burst_threshold_ratio == 3.0
        assert cfg.min_tokens_floor == 10
        assert cfg.penalty_multiplier == 0.5

    def test_custom_values(self) -> None:
        cfg = TenantConfig(max_tokens=500, refill_rate=200.0, window_seconds=120)
        assert cfg.max_tokens == 500
        assert cfg.refill_rate == 200.0
        assert cfg.window_seconds == 120


# -- Tenant configuration ----------------------------------------------------


class TestTenantConfiguration:
    def test_configure_tenant(self, limiter: AdaptiveRateLimiter) -> None:
        config = TenantConfig(max_tokens=200)
        limiter.configure_tenant("t1", config)
        assert limiter.get_tenant_config("t1").max_tokens == 200

    def test_get_default_config_for_unknown_tenant(self, limiter: AdaptiveRateLimiter) -> None:
        cfg = limiter.get_tenant_config("unknown")
        assert cfg.max_tokens == 100  # default

    def test_configure_tenant_creates_state(self, limiter: AdaptiveRateLimiter) -> None:
        config = TenantConfig(max_tokens=300, refill_rate=150.0)
        limiter.configure_tenant("t2", config)
        state = limiter._tenant_states["t2"]
        assert state.effective_max_tokens == 300
        assert state.effective_refill_rate == 150.0

    def test_reconfigure_tenant_preserves_state(self, limiter: AdaptiveRateLimiter) -> None:
        limiter.configure_tenant("t3", TenantConfig(max_tokens=100))
        limiter._tenant_states["t3"].consecutive_bursts = 5
        limiter.configure_tenant("t3", TenantConfig(max_tokens=200))
        # State should be preserved (not reset) on reconfiguration
        assert limiter._tenant_states["t3"].consecutive_bursts == 5


# -- IP suspicion management -------------------------------------------------


class TestIPSuspicion:
    def test_mark_ip_suspicious(self, limiter: AdaptiveRateLimiter) -> None:
        limiter.mark_ip_suspicious("10.0.0.1", 0.3)
        assert limiter.is_ip_suspicious("10.0.0.1")
        assert limiter.get_ip_penalty("10.0.0.1") == 0.3

    def test_clear_ip_suspicion(self, limiter: AdaptiveRateLimiter) -> None:
        limiter.mark_ip_suspicious("10.0.0.2", 0.5)
        limiter.clear_ip_suspicion("10.0.0.2")
        assert not limiter.is_ip_suspicious("10.0.0.2")
        assert limiter.get_ip_penalty("10.0.0.2") == 1.0

    def test_clear_nonexistent_ip_is_noop(self, limiter: AdaptiveRateLimiter) -> None:
        limiter.clear_ip_suspicion("never_seen")
        assert not limiter.is_ip_suspicious("never_seen")

    def test_penalty_factor_clamped_to_0_1(self, limiter: AdaptiveRateLimiter) -> None:
        limiter.mark_ip_suspicious("a", -0.5)
        assert limiter.get_ip_penalty("a") == 0.0

        limiter.mark_ip_suspicious("b", 1.5)
        assert limiter.get_ip_penalty("b") == 1.0

    def test_non_suspicious_ip_returns_full_allowance(self, limiter: AdaptiveRateLimiter) -> None:
        assert limiter.get_ip_penalty("clean_ip") == 1.0


# -- check() -----------------------------------------------------------------


class TestCheck:
    @pytest.mark.asyncio
    async def test_allowed_request(self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock) -> None:
        mock_redis.evalsha = AsyncMock(return_value=[1, 49, 3])
        allowed, meta = await limiter.check("tenant_a")
        assert allowed is True
        assert meta["remaining"] == 49
        assert meta["window_count"] == 3
        assert meta["penalty_factor"] == 1.0

    @pytest.mark.asyncio
    async def test_denied_request(self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock) -> None:
        mock_redis.evalsha = AsyncMock(return_value=[0, 250, 100])
        allowed, meta = await limiter.check("tenant_a")
        assert allowed is False
        assert meta["retry_after_ms"] == 250
        assert "remaining" not in meta

    @pytest.mark.asyncio
    async def test_check_with_suspicious_ip_reduces_limits(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        limiter.configure_tenant("t_ip", TenantConfig(max_tokens=100, refill_rate=50.0))
        limiter.mark_ip_suspicious("bad_ip", 0.5)

        mock_redis.evalsha = AsyncMock(return_value=[1, 24, 2])
        allowed, meta = await limiter.check("t_ip", client_ip="bad_ip")

        assert allowed is True
        assert meta["penalty_factor"] == 0.5
        assert meta["effective_limit"] == 50  # 100 * 0.5
        assert meta["effective_rate"] == 25.0  # 50 * 0.5

    @pytest.mark.asyncio
    async def test_check_with_suspicious_ip_respects_min_floor(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        limiter.configure_tenant(
            "t_floor", TenantConfig(max_tokens=20, min_tokens_floor=10)
        )
        limiter.mark_ip_suspicious("harsh_ip", 0.1)  # 20 * 0.1 = 2, but floor is 10

        mock_redis.evalsha = AsyncMock(return_value=[1, 9, 1])
        allowed, meta = await limiter.check("t_floor", client_ip="harsh_ip")

        assert allowed is True
        assert meta["effective_limit"] == 10  # min floor

    @pytest.mark.asyncio
    async def test_check_uses_ip_specific_keys_for_suspicious(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        limiter.mark_ip_suspicious("flagged_ip", 0.5)
        mock_redis.evalsha = AsyncMock(return_value=[1, 10, 1])

        await limiter.check("tenant_x", client_ip="flagged_ip")

        call_args = mock_redis.evalsha.call_args
        # KEYS[1] should contain the ip
        assert "ip:flagged_ip" in call_args[0][2]
        # KEYS[2] should contain the ip
        assert "ip:flagged_ip" in call_args[0][3]

    @pytest.mark.asyncio
    async def test_check_with_clean_ip_uses_tenant_keys(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        mock_redis.evalsha = AsyncMock(return_value=[1, 99, 1])

        await limiter.check("tenant_y", client_ip="good_ip")

        call_args = mock_redis.evalsha.call_args
        assert "ip:" not in call_args[0][2]

    @pytest.mark.asyncio
    async def test_check_fails_open_on_redis_error(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        mock_redis.evalsha = AsyncMock(side_effect=Exception("connection lost"))

        allowed, meta = await limiter.check("tenant_err")
        assert allowed is True
        assert meta["fallback"] is True
        assert "error" in meta

    @pytest.mark.asyncio
    async def test_check_creates_state_for_new_tenant(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        mock_redis.evalsha = AsyncMock(return_value=[1, 99, 1])
        await limiter.check("new_tenant")
        assert "new_tenant" in limiter._tenant_states

    @pytest.mark.asyncio
    async def test_scripts_loaded_once(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        mock_redis.evalsha = AsyncMock(return_value=[1, 99, 1])
        await limiter.check("t1")
        await limiter.check("t1")
        # script_load should be called exactly twice (one for each script), not on every check
        assert mock_redis.script_load.call_count == 2


# -- Burst detection ---------------------------------------------------------


class TestBurstDetection:
    @pytest.mark.asyncio
    async def test_burst_detected(self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock) -> None:
        # short_count=50, long_count=100 over 60s window
        # expected_short = (100/60000)*5000 = ~8.33; ratio = 50/8.33 = ~6.0
        mock_redis.evalsha = AsyncMock(return_value=[50, 100])

        burst = await limiter.detect_burst("tenant_burst")
        assert burst.is_burst is True
        assert burst.short_window_count == 50
        assert burst.long_window_count == 100
        assert burst.ratio > 3.0

    @pytest.mark.asyncio
    async def test_no_burst_normal_traffic(self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock) -> None:
        # short_count=8, long_count=100
        # expected_short = (100/60000)*5000 = ~8.33; ratio = 8/8.33 = ~0.96
        mock_redis.evalsha = AsyncMock(return_value=[8, 100])

        burst = await limiter.detect_burst("tenant_normal")
        assert burst.is_burst is False
        assert burst.ratio < 3.0

    @pytest.mark.asyncio
    async def test_burst_with_zero_long_count(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        mock_redis.evalsha = AsyncMock(return_value=[0, 0])
        burst = await limiter.detect_burst("tenant_empty")
        assert burst.is_burst is False
        assert burst.ratio == 0.0

    @pytest.mark.asyncio
    async def test_burst_detection_handles_redis_error(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        mock_redis.evalsha = AsyncMock(side_effect=Exception("timeout"))
        burst = await limiter.detect_burst("tenant_err")
        assert burst.is_burst is False
        assert burst.short_window_count == 0


# -- Adaptive limit adjustment -----------------------------------------------


class TestAdjustLimits:
    @pytest.mark.asyncio
    async def test_reduce_limits_on_burst(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        limiter.configure_tenant("t_adj", TenantConfig(max_tokens=100, refill_rate=50.0))
        # Simulate burst
        mock_redis.evalsha = AsyncMock(return_value=[50, 100])

        result = await limiter.adjust_limits("t_adj")
        assert result["adjusted"] is True
        assert result["reason"] == "burst_detected"
        assert result["effective_max_tokens"] < 100
        assert result["effective_refill_rate"] < 50.0

    @pytest.mark.asyncio
    async def test_recovery_when_no_burst(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        limiter.configure_tenant("t_rec", TenantConfig(max_tokens=100, refill_rate=50.0))
        # Manually reduce the effective limits
        state = limiter._tenant_states["t_rec"]
        state.effective_max_tokens = 50
        state.effective_refill_rate = 25.0
        state.consecutive_bursts = 2

        # No burst
        mock_redis.evalsha = AsyncMock(return_value=[8, 100])

        result = await limiter.adjust_limits("t_rec")
        assert result["adjusted"] is True
        assert result["reason"] == "recovery"
        assert result["effective_max_tokens"] > 50
        assert result["effective_refill_rate"] > 25.0

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_adjustments(
        self, mock_redis: AsyncMock
    ) -> None:
        limiter = AdaptiveRateLimiter(
            redis=mock_redis,
            adjustment_cooldown_seconds=999.0,
        )
        limiter.configure_tenant("t_cool", TenantConfig())
        # Force burst result
        mock_redis.evalsha = AsyncMock(return_value=[50, 100])

        # First call adjusts
        r1 = await limiter.adjust_limits("t_cool")
        assert r1["adjusted"] is True

        # Second call should be blocked by cooldown
        r2 = await limiter.adjust_limits("t_cool")
        assert r2["adjusted"] is False
        assert r2["reason"] == "cooldown"

    @pytest.mark.asyncio
    async def test_consecutive_bursts_increase_penalty(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        limiter.configure_tenant("t_consec", TenantConfig(max_tokens=100, penalty_multiplier=0.5))
        mock_redis.evalsha = AsyncMock(return_value=[50, 100])

        r1 = await limiter.adjust_limits("t_consec")
        tokens_after_first = r1["effective_max_tokens"]

        r2 = await limiter.adjust_limits("t_consec")
        tokens_after_second = r2["effective_max_tokens"]

        assert tokens_after_second < tokens_after_first

    @pytest.mark.asyncio
    async def test_min_tokens_floor_respected_during_adjustment(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        limiter.configure_tenant(
            "t_min", TenantConfig(max_tokens=20, min_tokens_floor=10, penalty_multiplier=0.1)
        )
        mock_redis.evalsha = AsyncMock(return_value=[50, 100])

        # Multiple burst adjustments should not go below floor
        for _ in range(5):
            await limiter.adjust_limits("t_min")

        state = limiter._tenant_states["t_min"]
        assert state.effective_max_tokens >= 10

    @pytest.mark.asyncio
    async def test_recovery_does_not_exceed_baseline(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        limiter.configure_tenant("t_cap", TenantConfig(max_tokens=100, refill_rate=50.0))
        state = limiter._tenant_states["t_cap"]
        state.effective_max_tokens = 95
        state.effective_refill_rate = 48.0

        # No burst
        mock_redis.evalsha = AsyncMock(return_value=[8, 100])

        for _ in range(20):
            state.last_adjustment_time = 0.0  # reset cooldown
            await limiter.adjust_limits("t_cap")

        assert state.effective_max_tokens <= 100
        assert state.effective_refill_rate <= 50.0


# -- Usage and stats ---------------------------------------------------------


class TestUsageAndStats:
    @pytest.mark.asyncio
    async def test_get_tenant_usage(self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock) -> None:
        limiter.configure_tenant("t_usage", TenantConfig(max_tokens=200, window_seconds=30))
        mock_redis.zcount = AsyncMock(return_value=42)

        usage = await limiter.get_tenant_usage("t_usage")
        assert usage["tenant_id"] == "t_usage"
        assert usage["current_count"] == 42
        assert usage["window_seconds"] == 30
        assert usage["effective_max_tokens"] == 200

    @pytest.mark.asyncio
    async def test_get_tenant_usage_handles_redis_error(
        self, limiter: AdaptiveRateLimiter, mock_redis: AsyncMock
    ) -> None:
        mock_redis.zcount = AsyncMock(side_effect=Exception("down"))
        usage = await limiter.get_tenant_usage("t_err")
        assert usage["current_count"] == 0

    def test_stats_summary(self, limiter: AdaptiveRateLimiter) -> None:
        limiter.configure_tenant("t1", TenantConfig())
        limiter.configure_tenant("t2", TenantConfig())
        limiter.mark_ip_suspicious("1.2.3.4", 0.3)

        stats = limiter.stats()
        assert stats["configured_tenants"] == 2
        assert stats["active_tenants"] == 2
        assert stats["suspicious_ips"] == 1
        assert "t1" in stats["tenant_states"]
        assert "t2" in stats["tenant_states"]

    def test_stats_empty(self, limiter: AdaptiveRateLimiter) -> None:
        stats = limiter.stats()
        assert stats["configured_tenants"] == 0
        assert stats["suspicious_ips"] == 0


# -- BurstInfo dataclass -----------------------------------------------------


class TestBurstInfo:
    def test_defaults(self) -> None:
        info = BurstInfo()
        assert info.is_burst is False
        assert info.short_window_count == 0
        assert info.long_window_count == 0
        assert info.ratio == 0.0

    def test_custom_values(self) -> None:
        info = BurstInfo(is_burst=True, short_window_count=50, long_window_count=200, ratio=6.0)
        assert info.is_burst is True
        assert info.ratio == 6.0


# -- _TenantState dataclass --------------------------------------------------


class TestTenantState:
    def test_defaults(self) -> None:
        state = _TenantState()
        assert state.effective_max_tokens == 100
        assert state.consecutive_bursts == 0

    def test_mutation(self) -> None:
        state = _TenantState(effective_max_tokens=200)
        state.consecutive_bursts = 3
        assert state.consecutive_bursts == 3
