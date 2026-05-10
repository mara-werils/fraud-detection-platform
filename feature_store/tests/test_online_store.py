"""Tests for the Redis-based online feature store using fakeredis."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import fakeredis.aioredis
import pytest
import pytest_asyncio

from feature_store.online_store import _TTL_SECONDS, OnlineFeatureStore
from shared.schemas import FeatureVector, Transaction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transaction(
    user_id: UUID | None = None,
    amount: Decimal = Decimal("100.00"),
    merchant_id: str | None = "merchant_1",
    device_id: str | None = "device_abc",
    latitude: float | None = 40.7128,
    longitude: float | None = -74.0060,
    ts: datetime | None = None,
) -> Transaction:
    return Transaction(
        transaction_id=uuid4(),
        user_id=user_id or uuid4(),
        amount=amount,
        currency="USD",
        transaction_type="purchase",
        merchant_id=merchant_id,
        device_id=device_id,
        latitude=latitude,
        longitude=longitude,
        timestamp=ts or datetime.now(tz=UTC),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store():
    """Create an OnlineFeatureStore backed by fakeredis."""
    s = OnlineFeatureStore(redis_url="redis://localhost:6379/0")
    # Swap in a fakeredis instance
    s._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield s
    await s._redis.aclose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpdateFeaturesCreatesCorrectKeys:
    """Verify that update_features populates the expected Redis key patterns."""

    @pytest.mark.asyncio
    async def test_creates_velocity_key(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)

        exists = await store._redis.exists(f"user:{user_id}:velocity")
        assert exists, "velocity sorted set should exist after update"

    @pytest.mark.asyncio
    async def test_creates_stats_key(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)

        exists = await store._redis.exists(f"user:{user_id}:stats")
        assert exists, "stats hash should exist after update"

    @pytest.mark.asyncio
    async def test_creates_geo_key(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id, latitude=51.5074, longitude=-0.1278)
        await store.update_features(txn)

        exists = await store._redis.exists(f"user:{user_id}:geo")
        assert exists, "geo hash should exist after update"

    @pytest.mark.asyncio
    async def test_creates_devices_key(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id, device_id="fingerprint_xyz")
        await store.update_features(txn)

        exists = await store._redis.exists(f"user:{user_id}:devices")
        assert exists, "devices set should exist after update"

    @pytest.mark.asyncio
    async def test_creates_merchants_key(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id, merchant_id="shop_42")
        await store.update_features(txn)

        exists = await store._redis.exists(f"user:{user_id}:merchants_7d")
        assert exists, "merchants sorted set should exist after update"

    @pytest.mark.asyncio
    async def test_creates_meta_key(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)

        exists = await store._redis.exists(f"user:{user_id}:meta")
        assert exists, "meta hash should exist after update"


class TestVelocitySlidingWindows:
    """Verify that velocity counting works within sliding windows."""

    @pytest.mark.asyncio
    async def test_single_txn_counted_in_all_windows(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)

        features = await store.get_features(user_id, txn)
        assert features.txn_count_1h >= 1
        assert features.txn_count_24h >= 1

    @pytest.mark.asyncio
    async def test_multiple_txns_accumulate(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        now = datetime.now(tz=UTC)

        for _ in range(5):
            txn = _make_transaction(user_id=user_id, amount=Decimal("10.00"), ts=now)
            await store.update_features(txn)

        probe = _make_transaction(user_id=user_id, amount=Decimal("10.00"), ts=now)
        features = await store.get_features(user_id, probe)

        assert features.txn_count_1h == 5
        assert features.txn_count_24h == 5

    @pytest.mark.asyncio
    async def test_amount_sums_in_window(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        now = datetime.now(tz=UTC)

        for amt in [Decimal("10.00"), Decimal("20.00"), Decimal("30.00")]:
            txn = _make_transaction(user_id=user_id, amount=amt, ts=now)
            await store.update_features(txn)

        probe = _make_transaction(user_id=user_id, amount=Decimal("0"), ts=now)
        features = await store.get_features(user_id, probe)

        assert features.txn_amount_sum_1h == Decimal("60.00") or float(
            features.txn_amount_sum_1h
        ) == pytest.approx(60.0, abs=0.01)


class TestGetFeaturesReturnsCompleteVector:
    """Ensure get_features returns a properly populated FeatureVector."""

    @pytest.mark.asyncio
    async def test_returns_feature_vector_type(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)
        features = await store.get_features(user_id, txn)
        assert isinstance(features, FeatureVector)

    @pytest.mark.asyncio
    async def test_feature_vector_has_correct_ids(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)
        features = await store.get_features(user_id, txn)
        assert features.user_id == user_id
        assert features.transaction_id == txn.transaction_id

    @pytest.mark.asyncio
    async def test_new_device_flag(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn1 = _make_transaction(user_id=user_id, device_id="device_A")
        await store.update_features(txn1)

        txn2 = _make_transaction(user_id=user_id, device_id="device_B")
        features = await store.get_features(user_id, txn2)
        assert features.is_new_device is True

    @pytest.mark.asyncio
    async def test_known_device_flag(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn1 = _make_transaction(user_id=user_id, device_id="device_A")
        await store.update_features(txn1)

        txn2 = _make_transaction(user_id=user_id, device_id="device_A")
        features = await store.get_features(user_id, txn2)
        assert features.is_new_device is False

    @pytest.mark.asyncio
    async def test_new_merchant_flag(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn1 = _make_transaction(user_id=user_id, merchant_id="shop_A")
        await store.update_features(txn1)

        txn2 = _make_transaction(user_id=user_id, merchant_id="shop_B")
        features = await store.get_features(user_id, txn2)
        assert features.is_new_merchant is True

    @pytest.mark.asyncio
    async def test_amount_zscore_nonzero_after_multiple(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        now = datetime.now(tz=UTC)

        for amt in [Decimal("10"), Decimal("10"), Decimal("10"), Decimal("10")]:
            txn = _make_transaction(user_id=user_id, amount=amt, ts=now)
            await store.update_features(txn)

        outlier = _make_transaction(user_id=user_id, amount=Decimal("1000"), ts=now)
        await store.update_features(outlier)
        features = await store.get_features(user_id, outlier)

        # A 1000 after four 10s should have a large positive z-score
        assert features.amount_zscore > 1.0

    @pytest.mark.asyncio
    async def test_distance_from_last_txn(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn1 = _make_transaction(user_id=user_id, latitude=40.7128, longitude=-74.0060)
        await store.update_features(txn1)

        # London coordinates — ~5500 km from NYC
        txn2 = _make_transaction(user_id=user_id, latitude=51.5074, longitude=-0.1278)
        features = await store.get_features(user_id, txn2)
        assert features.distance_from_last_txn_km > 5000


class TestTTLIsSetCorrectly:
    """Verify that Redis keys have the expected TTL."""

    @pytest.mark.asyncio
    async def test_velocity_ttl(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)

        ttl = await store._redis.ttl(f"user:{user_id}:velocity")
        assert ttl > 0
        assert ttl <= _TTL_SECONDS

    @pytest.mark.asyncio
    async def test_stats_ttl(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)

        ttl = await store._redis.ttl(f"user:{user_id}:stats")
        assert ttl > 0
        assert ttl <= _TTL_SECONDS

    @pytest.mark.asyncio
    async def test_devices_ttl(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)

        ttl = await store._redis.ttl(f"user:{user_id}:devices")
        assert ttl > 0
        assert ttl <= _TTL_SECONDS

    @pytest.mark.asyncio
    async def test_merchants_ttl(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)

        ttl = await store._redis.ttl(f"user:{user_id}:merchants_7d")
        assert ttl > 0
        assert ttl <= _TTL_SECONDS

    @pytest.mark.asyncio
    async def test_meta_ttl(self, store: OnlineFeatureStore) -> None:
        user_id = uuid4()
        txn = _make_transaction(user_id=user_id)
        await store.update_features(txn)

        ttl = await store._redis.ttl(f"user:{user_id}:meta")
        assert ttl > 0
        assert ttl <= _TTL_SECONDS
