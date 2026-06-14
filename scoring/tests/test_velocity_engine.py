"""Tests for the velocity check engine."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from scoring.services.velocity_engine import (
    DEFAULT_THRESHOLDS,
    Dimension,
    TimeWindow,
    VelocityEngine,
    VelocityReport,
    VelocityResult,
    VelocityThreshold,
    VelocityType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_redis_mock() -> AsyncMock:
    """Create a mock Redis client with sorted-set support."""
    redis = AsyncMock()
    # Internal storage: key -> list of (member, score)
    _store: dict[str, list[tuple[str, float]]] = {}

    def _pipeline():
        pipe = AsyncMock()
        _ops: list = []

        def _zadd(key, mapping):
            for member, score in mapping.items():
                _store.setdefault(key, [])
                # Replace if same member exists
                _store[key] = [(m, s) for m, s in _store[key] if m != member]
                _store[key].append((member, score))

        def _expire(key, ttl):
            pass  # no-op for tests

        def _delete(key):
            _store.pop(key, None)

        pipe.zadd = MagicMock(side_effect=lambda k, m: _ops.append(("zadd", k, m)))
        pipe.expire = MagicMock(side_effect=lambda k, t: _ops.append(("expire", k, t)))
        pipe.delete = MagicMock(side_effect=lambda k: _ops.append(("delete", k)))

        async def _execute():
            results = []
            for op in _ops:
                if op[0] == "zadd":
                    _zadd(op[1], op[2])
                    results.append(1)
                elif op[0] == "expire":
                    _expire(op[1], op[2])
                    results.append(True)
                elif op[0] == "delete":
                    _delete(op[1])
                    results.append(1)
            _ops.clear()
            return results

        pipe.execute = _execute
        return pipe

    redis.pipeline = _pipeline

    async def _zcount(key, min_score, max_score):
        entries = _store.get(key, [])
        min_val = float("-inf") if min_score == "-inf" else float(min_score)
        max_val = float("inf") if max_score == "+inf" else float(max_score)
        return sum(1 for _, s in entries if min_val <= s <= max_val)

    async def _zrangebyscore(key, min_score, max_score):
        entries = _store.get(key, [])
        min_val = float("-inf") if min_score == "-inf" else float(min_score)
        max_val = float("inf") if max_score == "+inf" else float(max_score)
        return [m for m, s in entries if min_val <= s <= max_val]

    async def _zremrangebyscore(key, min_score, max_score):
        if key not in _store:
            return 0
        min_val = float("-inf") if min_score == "-inf" else float(min_score)
        max_val = float("inf") if max_score == "+inf" else float(max_score)
        before = len(_store[key])
        _store[key] = [(m, s) for m, s in _store[key] if not (min_val <= s <= max_val)]
        return before - len(_store[key])

    redis.zcount = _zcount
    redis.zrangebyscore = _zrangebyscore
    redis.zremrangebyscore = _zremrangebyscore
    redis._store = _store

    return redis


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

class TestTimeWindow:
    def test_labels(self) -> None:
        assert TimeWindow.ONE_MIN.label == "1m"
        assert TimeWindow.FIVE_MIN.label == "5m"
        assert TimeWindow.ONE_HOUR.label == "1h"
        assert TimeWindow.TWENTY_FOUR_HOUR.label == "24h"

    def test_values_in_seconds(self) -> None:
        assert TimeWindow.ONE_MIN.value == 60
        assert TimeWindow.FIVE_MIN.value == 300
        assert TimeWindow.ONE_HOUR.value == 3600
        assert TimeWindow.TWENTY_FOUR_HOUR.value == 86400


class TestVelocityResult:
    def test_to_dict(self) -> None:
        result = VelocityResult(
            dimension=Dimension.USER,
            velocity_type=VelocityType.COUNT,
            window=TimeWindow.ONE_HOUR,
            current_value=5.0,
            limit=10.0,
            breached=False,
        )
        d = result.to_dict()
        assert d["dimension"] == "user"
        assert d["velocity_type"] == "count"
        assert d["window"] == "1h"
        assert d["current_value"] == 5.0
        assert d["limit"] == 10.0
        assert d["breached"] is False


class TestVelocityReport:
    def test_to_dict_empty(self) -> None:
        report = VelocityReport(entity_id="u123")
        d = report.to_dict()
        assert d["entity_id"] == "u123"
        assert d["score"] == 0.0
        assert d["any_breach"] is False
        assert d["results"] == []

    def test_to_dict_with_results(self) -> None:
        report = VelocityReport(
            entity_id="u123",
            results=[
                VelocityResult(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 3.0, 3.0, True),
            ],
            score=0.85,
            any_breach=True,
        )
        d = report.to_dict()
        assert d["any_breach"] is True
        assert len(d["results"]) == 1


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestVelocityEngineRecord:
    @pytest.mark.asyncio
    async def test_record_single_entity(self) -> None:
        redis = _make_redis_mock()
        engine = VelocityEngine(redis, thresholds=[])
        await engine.record(
            entities={Dimension.USER: "u1"},
            transaction_id="tx1",
            amount=100.0,
            merchant_id="m1",
            timestamp=1000.0,
        )
        # Should have keys for user x {count, amount, unique_merchant}
        keys = list(redis._store.keys())
        assert len(keys) == 3
        assert any("user:u1:count" in k for k in keys)
        assert any("user:u1:amount" in k for k in keys)
        assert any("user:u1:unique_merchant" in k for k in keys)

    @pytest.mark.asyncio
    async def test_record_multiple_dimensions(self) -> None:
        redis = _make_redis_mock()
        engine = VelocityEngine(redis, thresholds=[])
        await engine.record(
            entities={Dimension.USER: "u1", Dimension.CARD: "c1", Dimension.IP: "1.2.3.4"},
            transaction_id="tx1",
            amount=50.0,
            timestamp=1000.0,
        )
        # 3 dimensions x 3 velocity types = 9 keys
        assert len(redis._store) == 9

    @pytest.mark.asyncio
    async def test_record_auto_generates_txn_id(self) -> None:
        redis = _make_redis_mock()
        engine = VelocityEngine(redis, thresholds=[])
        await engine.record(
            entities={Dimension.USER: "u1"},
            amount=10.0,
            timestamp=1000.0,
        )
        key = "vel:user:u1:count"
        members = redis._store[key]
        assert len(members) == 1
        data = json.loads(members[0][0])
        assert len(data["txn"]) == 32  # uuid4 hex


class TestVelocityEngineCountVelocity:
    @pytest.mark.asyncio
    async def test_count_within_window(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 3),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        now = 10000.0

        # Record 2 transactions within 1 min
        await engine.record({Dimension.USER: "u1"}, "tx1", 10.0, "m1", now - 30)
        await engine.record({Dimension.USER: "u1"}, "tx2", 20.0, "m2", now - 10)

        report = await engine.check({Dimension.USER: "u1"}, now=now)
        assert len(report.results) == 1
        assert report.results[0].current_value == 2.0
        assert report.results[0].breached is False

    @pytest.mark.asyncio
    async def test_count_breach(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 3),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        now = 10000.0

        for i in range(4):
            await engine.record({Dimension.USER: "u1"}, f"tx{i}", 10.0, "m1", now - 50 + i * 10)

        report = await engine.check({Dimension.USER: "u1"}, now=now)
        assert report.results[0].current_value == 4.0
        assert report.results[0].breached is True
        assert report.any_breach is True

    @pytest.mark.asyncio
    async def test_count_excludes_expired(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 10),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        now = 10000.0

        # One old, one recent
        await engine.record({Dimension.USER: "u1"}, "old", 10.0, "m1", now - 120)
        await engine.record({Dimension.USER: "u1"}, "new", 10.0, "m1", now - 5)

        report = await engine.check({Dimension.USER: "u1"}, now=now)
        assert report.results[0].current_value == 1.0


class TestVelocityEngineAmountVelocity:
    @pytest.mark.asyncio
    async def test_amount_sum_in_window(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.AMOUNT, TimeWindow.ONE_HOUR, 5000.0),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        now = 10000.0

        await engine.record({Dimension.USER: "u1"}, "tx1", 1500.0, "m1", now - 1800)
        await engine.record({Dimension.USER: "u1"}, "tx2", 2000.0, "m2", now - 600)

        report = await engine.check({Dimension.USER: "u1"}, now=now)
        assert report.results[0].current_value == 3500.0
        assert report.results[0].breached is False

    @pytest.mark.asyncio
    async def test_amount_breach(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.AMOUNT, TimeWindow.ONE_HOUR, 1000.0),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        now = 10000.0

        await engine.record({Dimension.USER: "u1"}, "tx1", 600.0, "m1", now - 100)
        await engine.record({Dimension.USER: "u1"}, "tx2", 500.0, "m2", now - 50)

        report = await engine.check({Dimension.USER: "u1"}, now=now)
        assert report.results[0].current_value == 1100.0
        assert report.results[0].breached is True


class TestVelocityEngineUniqueMerchant:
    @pytest.mark.asyncio
    async def test_unique_merchants_counted(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.UNIQUE_MERCHANT, TimeWindow.ONE_HOUR, 5),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        now = 10000.0

        await engine.record({Dimension.USER: "u1"}, "tx1", 10.0, "merchantA", now - 100)
        await engine.record({Dimension.USER: "u1"}, "tx2", 20.0, "merchantB", now - 80)
        await engine.record({Dimension.USER: "u1"}, "tx3", 30.0, "merchantA", now - 60)  # duplicate

        report = await engine.check({Dimension.USER: "u1"}, now=now)
        assert report.results[0].current_value == 2.0
        assert report.results[0].breached is False

    @pytest.mark.asyncio
    async def test_unique_merchant_breach(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.UNIQUE_MERCHANT, TimeWindow.ONE_HOUR, 3),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        now = 10000.0

        for i in range(4):
            await engine.record({Dimension.USER: "u1"}, f"tx{i}", 10.0, f"merchant_{i}", now - 100 + i)

        report = await engine.check({Dimension.USER: "u1"}, now=now)
        assert report.results[0].current_value == 4.0
        assert report.results[0].breached is True

    @pytest.mark.asyncio
    async def test_empty_merchant_not_counted(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.UNIQUE_MERCHANT, TimeWindow.ONE_HOUR, 10),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        now = 10000.0

        await engine.record({Dimension.USER: "u1"}, "tx1", 10.0, None, now - 100)
        await engine.record({Dimension.USER: "u1"}, "tx2", 10.0, "", now - 80)

        report = await engine.check({Dimension.USER: "u1"}, now=now)
        assert report.results[0].current_value == 0.0


class TestVelocityEngineMultiWindow:
    @pytest.mark.asyncio
    async def test_multiple_windows_different_results(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 5),
            VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_HOUR, 20),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        now = 10000.0

        # 2 recent (within 1 min), 3 older (within 1 hour but outside 1 min)
        await engine.record({Dimension.USER: "u1"}, "tx1", 10.0, "m1", now - 30)
        await engine.record({Dimension.USER: "u1"}, "tx2", 10.0, "m1", now - 20)
        await engine.record({Dimension.USER: "u1"}, "tx3", 10.0, "m1", now - 600)
        await engine.record({Dimension.USER: "u1"}, "tx4", 10.0, "m1", now - 1200)
        await engine.record({Dimension.USER: "u1"}, "tx5", 10.0, "m1", now - 1800)

        report = await engine.check({Dimension.USER: "u1"}, now=now)
        assert len(report.results) == 2

        one_min = report.results[0]
        one_hour = report.results[1]

        assert one_min.window == TimeWindow.ONE_MIN
        assert one_min.current_value == 2.0
        assert one_min.breached is False

        assert one_hour.window == TimeWindow.ONE_HOUR
        assert one_hour.current_value == 5.0
        assert one_hour.breached is False


class TestVelocityEngineScoring:
    @pytest.mark.asyncio
    async def test_score_zero_when_no_activity(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 5),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        report = await engine.check({Dimension.USER: "u1"})
        assert report.score == 0.0
        assert report.any_breach is False

    @pytest.mark.asyncio
    async def test_score_capped_at_one(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 2),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        now = 10000.0

        for i in range(10):
            await engine.record({Dimension.USER: "u1"}, f"tx{i}", 1.0, "m1", now - 5 + i * 0.1)

        report = await engine.check({Dimension.USER: "u1"}, now=now)
        assert report.score <= 1.0

    def test_compute_score_empty(self) -> None:
        assert VelocityEngine._compute_score([]) == 0.0

    def test_compute_score_half_saturated(self) -> None:
        results = [
            VelocityResult(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 5.0, 10.0, False),
        ]
        assert VelocityEngine._compute_score(results) == 0.5

    def test_compute_score_over_limit_capped(self) -> None:
        results = [
            VelocityResult(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 20.0, 10.0, True),
        ]
        assert VelocityEngine._compute_score(results) == 1.0


class TestVelocityEngineSkipsMissingDimension:
    @pytest.mark.asyncio
    async def test_skips_dimension_not_in_entities(self) -> None:
        redis = _make_redis_mock()
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 5),
            VelocityThreshold(Dimension.CARD, VelocityType.COUNT, TimeWindow.ONE_MIN, 3),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        report = await engine.check({Dimension.USER: "u1"})
        # Only user threshold evaluated, card skipped
        assert len(report.results) == 1
        assert report.results[0].dimension == Dimension.USER


class TestVelocityEngineReset:
    @pytest.mark.asyncio
    async def test_reset_removes_all_velocity_types(self) -> None:
        redis = _make_redis_mock()
        engine = VelocityEngine(redis, thresholds=[])
        now = 10000.0

        await engine.record({Dimension.USER: "u1"}, "tx1", 100.0, "m1", now)
        assert len(redis._store) == 3

        await engine.reset(Dimension.USER, "u1")
        assert len(redis._store) == 0


class TestVelocityEngineStats:
    def test_stats_returns_config_info(self) -> None:
        redis = _make_redis_mock()
        engine = VelocityEngine(redis)
        stats = engine.stats()
        assert stats["thresholds_configured"] == len(DEFAULT_THRESHOLDS)
        assert "user" in stats["dimensions"]
        assert "1m" in stats["windows"]
        assert stats["windows"]["1m"] == 60


class TestVelocityEngineMemberEncoding:
    def test_member_encode_decode_roundtrip(self) -> None:
        member = VelocityEngine._member("tx123", 99.5, "shop_abc")
        decoded = VelocityEngine._decode_member(member)
        assert decoded["txn"] == "tx123"
        assert decoded["amt"] == 99.5
        assert decoded["mer"] == "shop_abc"

    def test_decode_bytes(self) -> None:
        member = VelocityEngine._member("tx1", 10.0, "m1")
        decoded = VelocityEngine._decode_member(member.encode("utf-8"))
        assert decoded["txn"] == "tx1"

    def test_member_no_merchant(self) -> None:
        member = VelocityEngine._member("tx1", 10.0)
        decoded = VelocityEngine._decode_member(member)
        assert decoded["mer"] == ""


class TestVelocityEngineErrorHandling:
    @pytest.mark.asyncio
    async def test_record_propagates_redis_error(self) -> None:
        redis = _make_redis_mock()
        redis.pipeline = MagicMock(side_effect=Exception("connection refused"))
        engine = VelocityEngine(redis, thresholds=[])
        with pytest.raises(Exception, match="connection refused"):
            await engine.record({Dimension.USER: "u1"}, "tx1", 10.0, "m1", 1000.0)

    @pytest.mark.asyncio
    async def test_check_propagates_redis_error(self) -> None:
        redis = _make_redis_mock()
        redis.zremrangebyscore = AsyncMock(side_effect=Exception("timeout"))
        thresholds = [
            VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 5),
        ]
        engine = VelocityEngine(redis, thresholds=thresholds)
        with pytest.raises(Exception, match="timeout"):
            await engine.check({Dimension.USER: "u1"})


class TestDefaultThresholds:
    def test_default_thresholds_non_empty(self) -> None:
        assert len(DEFAULT_THRESHOLDS) > 0

    def test_all_dimensions_covered(self) -> None:
        dims = {t.dimension for t in DEFAULT_THRESHOLDS}
        assert Dimension.USER in dims
        assert Dimension.CARD in dims
        assert Dimension.DEVICE in dims
        assert Dimension.IP in dims

    def test_all_velocity_types_covered(self) -> None:
        types = {t.velocity_type for t in DEFAULT_THRESHOLDS}
        assert VelocityType.COUNT in types
        assert VelocityType.AMOUNT in types
        assert VelocityType.UNIQUE_MERCHANT in types

    def test_all_windows_used(self) -> None:
        windows = {t.window for t in DEFAULT_THRESHOLDS}
        assert TimeWindow.ONE_MIN in windows
        assert TimeWindow.FIVE_MIN in windows
        assert TimeWindow.ONE_HOUR in windows
        assert TimeWindow.TWENTY_FOUR_HOUR in windows
