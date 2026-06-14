"""Tests for the BehavioralAnalytics engine."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from scoring.services.behavioral_analytics import (
    BehavioralAnalytics,
    DeviationResult,
    TransactionRecord,
    UserProfileSummary,
    _EMAAccumulator,
    _sigmoid,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _txn(
    user_id: str = "user-1",
    amount: float = 50.0,
    hour: int = 14,
    weekday: int = 2,  # Wednesday
    category: str = "groceries",
    payment_method: str = "credit_card",
) -> TransactionRecord:
    """Build a TransactionRecord with convenient defaults."""
    # Build a datetime for the given hour and weekday
    # 2025-06-16 is a Monday; offset by weekday
    base = datetime(2025, 6, 16, hour, 0, 0, tzinfo=UTC)
    ts = base + timedelta(days=weekday)
    return TransactionRecord(
        user_id=user_id,
        amount=amount,
        timestamp=ts,
        merchant_category=category,
        payment_method=payment_method,
    )


def _build_baseline(
    ba: BehavioralAnalytics,
    user_id: str = "user-1",
    n: int = 20,
    amount: float = 50.0,
    hour: int = 14,
    weekday: int = 2,
    category: str = "groceries",
    payment_method: str = "credit_card",
) -> None:
    """Ingest *n* identical transactions to build a stable baseline."""
    for _ in range(n):
        ba.ingest(_txn(
            user_id=user_id,
            amount=amount,
            hour=hour,
            weekday=weekday,
            category=category,
            payment_method=payment_method,
        ))


# ── EMAAccumulator tests ────────────────────────────────────────────────────


class TestEMAAccumulator:
    def test_initial_state(self):
        ema = _EMAAccumulator()
        assert ema.count == 0
        assert ema.mean == 0.0
        assert ema.stddev == 0.0

    def test_first_observation(self):
        ema = _EMAAccumulator(alpha=0.1)
        ema.update(100.0)
        assert ema.count == 1
        assert ema.mean == 100.0
        assert ema.variance == 0.0

    def test_converges_toward_value(self):
        ema = _EMAAccumulator(alpha=0.5)
        for _ in range(50):
            ema.update(100.0)
        assert abs(ema.mean - 100.0) < 0.01

    def test_tracks_variance(self):
        ema = _EMAAccumulator(alpha=0.2)
        # Alternate between 0 and 100
        for _ in range(100):
            ema.update(0.0)
            ema.update(100.0)
        # Mean should be near 50, stddev should be substantial
        assert 30.0 < ema.mean < 70.0
        assert ema.stddev > 10.0

    def test_stddev_non_negative(self):
        ema = _EMAAccumulator(alpha=0.1)
        for v in [10, 20, 10, 20, 10]:
            ema.update(v)
        assert ema.stddev >= 0.0


# ── Sigmoid helper ───────────────────────────────────────────────────────────


class TestSigmoid:
    def test_at_zero(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-6

    def test_large_positive(self):
        assert _sigmoid(20.0) == 1.0

    def test_large_negative(self):
        assert _sigmoid(-20.0) == 0.0

    def test_monotonic(self):
        values = [_sigmoid(x) for x in range(-5, 6)]
        for a, b in zip(values, values[1:]):
            assert b >= a


# ── BehavioralAnalytics init ─────────────────────────────────────────────────


class TestBehavioralAnalyticsInit:
    def test_default_init(self):
        ba = BehavioralAnalytics()
        assert ba.profile_count == 0

    def test_custom_alpha(self):
        ba = BehavioralAnalytics(alpha=0.5)
        assert ba._alpha == 0.5

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha"):
            BehavioralAnalytics(alpha=0.0)

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha"):
            BehavioralAnalytics(alpha=-0.1)

    def test_alpha_one_is_valid(self):
        ba = BehavioralAnalytics(alpha=1.0)
        assert ba._alpha == 1.0


# ── Ingest ───────────────────────────────────────────────────────────────────


class TestIngest:
    def test_creates_profile_on_first_ingest(self):
        ba = BehavioralAnalytics()
        ba.ingest(_txn(user_id="u1"))
        assert ba.profile_count == 1
        assert "u1" in ba.get_all_user_ids()

    def test_increments_observation_count(self):
        ba = BehavioralAnalytics()
        for _ in range(10):
            ba.ingest(_txn())
        summary = ba.get_profile_summary("user-1")
        assert summary is not None
        assert summary.observation_count == 10

    def test_multiple_users(self):
        ba = BehavioralAnalytics()
        ba.ingest(_txn(user_id="alice"))
        ba.ingest(_txn(user_id="bob"))
        ba.ingest(_txn(user_id="charlie"))
        assert ba.profile_count == 3

    def test_eviction_at_capacity(self):
        ba = BehavioralAnalytics(max_profiles=10)
        for i in range(15):
            ba.ingest(_txn(user_id=f"user-{i}"))
        # After eviction of 10%, should be <= 10
        assert ba.profile_count <= 15  # some evicted
        # The oldest user should have been evicted
        assert "user-0" not in ba.get_all_user_ids()


# ── Scoring with insufficient history ───────────────────────────────────────


class TestScoreInsufficientHistory:
    def test_no_profile(self):
        ba = BehavioralAnalytics()
        result = ba.score(_txn(user_id="unknown"))
        assert result.composite_score == 0.0
        assert result.observations == 0
        assert result.details.get("reason") == "insufficient_history"

    def test_below_min_observations(self):
        ba = BehavioralAnalytics(min_observations=10)
        for _ in range(5):
            ba.ingest(_txn())
        result = ba.score(_txn())
        assert result.composite_score == 0.0
        assert result.observations == 5
        assert result.details.get("reason") == "insufficient_history"


# ── Amount deviation scoring ─────────────────────────────────────────────────


class TestAmountDeviation:
    def test_normal_amount_low_deviation(self):
        ba = BehavioralAnalytics(alpha=0.3)
        _build_baseline(ba, amount=50.0, n=30)
        result = ba.score(_txn(amount=55.0))
        assert result.amount_deviation < 0.3

    def test_extreme_amount_high_deviation(self):
        ba = BehavioralAnalytics(alpha=0.3)
        _build_baseline(ba, amount=50.0, n=30)
        result = ba.score(_txn(amount=5000.0))
        assert result.amount_deviation > 0.5

    def test_zero_amount_baseline(self):
        ba = BehavioralAnalytics(alpha=0.5)
        # Ingest transactions with zero amount
        for _ in range(10):
            ba.ingest(_txn(amount=0.0))
        result = ba.score(_txn(amount=0.0))
        assert result.amount_deviation == 0.0


# ── Time deviation scoring ───────────────────────────────────────────────────


class TestTimeDeviation:
    def test_usual_time_low_deviation(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, hour=14, weekday=2, n=30)
        result = ba.score(_txn(hour=14, weekday=2))
        assert result.time_deviation < 0.5

    def test_unusual_hour_high_deviation(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, hour=14, weekday=2, n=30)
        # Score at 3 AM — a time never seen before
        result = ba.score(_txn(hour=3, weekday=2))
        assert result.time_deviation > result.amount_deviation  # time is the outlier

    def test_unusual_day_contributes_to_deviation(self):
        ba = BehavioralAnalytics(alpha=0.2)
        # Build baseline on Wednesday (weekday=2)
        _build_baseline(ba, hour=14, weekday=2, n=30)
        # Score on Sunday (weekday=6)
        result = ba.score(_txn(hour=14, weekday=6))
        assert result.time_deviation > 0.0


# ── Category deviation scoring ───────────────────────────────────────────────


class TestCategoryDeviation:
    def test_known_category_low_deviation(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, category="groceries", n=20)
        result = ba.score(_txn(category="groceries"))
        assert result.category_deviation < 0.5

    def test_unknown_category_max_deviation(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, category="groceries", n=20)
        result = ba.score(_txn(category="luxury_watches"))
        assert result.category_deviation == 1.0

    def test_mixed_categories(self):
        ba = BehavioralAnalytics(alpha=0.2)
        # Interleave categories so EMA decay doesn't distort proportions
        for i in range(20):
            if i % 4 == 0:
                ba.ingest(_txn(category="electronics"))
            else:
                ba.ingest(_txn(category="groceries"))
        # Groceries (dominant) should have lower deviation than electronics
        result_groceries = ba.score(_txn(category="groceries"))
        result_electronics = ba.score(_txn(category="electronics"))
        assert result_groceries.category_deviation <= result_electronics.category_deviation


# ── Payment method deviation scoring ─────────────────────────────────────────


class TestPaymentMethodDeviation:
    def test_known_method_low_deviation(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, payment_method="credit_card", n=20)
        result = ba.score(_txn(payment_method="credit_card"))
        assert result.payment_method_deviation < 0.5

    def test_unknown_method_max_deviation(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, payment_method="credit_card", n=20)
        result = ba.score(_txn(payment_method="crypto"))
        assert result.payment_method_deviation == 1.0


# ── Composite scoring ────────────────────────────────────────────────────────


class TestCompositeScoring:
    def test_all_normal_low_composite(self):
        ba = BehavioralAnalytics(alpha=0.3)
        _build_baseline(ba, n=30)
        result = ba.score(_txn())  # same as baseline
        assert result.composite_score < 0.5

    def test_all_anomalous_high_composite(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, amount=50.0, hour=14, weekday=2,
                        category="groceries", payment_method="credit_card", n=30)
        # Everything is different
        result = ba.score(_txn(
            amount=50000.0,
            hour=3,
            weekday=6,
            category="casino",
            payment_method="crypto",
        ))
        assert result.composite_score > 0.5

    def test_composite_bounded_at_one(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, n=20)
        result = ba.score(_txn(
            amount=999999.0,
            hour=3,
            weekday=6,
            category="never_seen",
            payment_method="never_seen",
        ))
        assert result.composite_score <= 1.0

    def test_composite_non_negative(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, n=20)
        result = ba.score(_txn())
        assert result.composite_score >= 0.0


# ── score_and_ingest ─────────────────────────────────────────────────────────


class TestScoreAndIngest:
    def test_scores_then_updates_profile(self):
        ba = BehavioralAnalytics(min_observations=1, alpha=0.5)
        # Build initial baseline
        _build_baseline(ba, n=5)
        obs_before = ba.get_profile_summary("user-1").observation_count

        result = ba.score_and_ingest(_txn())
        assert isinstance(result, DeviationResult)

        obs_after = ba.get_profile_summary("user-1").observation_count
        assert obs_after == obs_before + 1


# ── Profile summary ──────────────────────────────────────────────────────────


class TestProfileSummary:
    def test_returns_none_for_unknown_user(self):
        ba = BehavioralAnalytics()
        assert ba.get_profile_summary("ghost") is None

    def test_summary_fields(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, n=20, amount=100.0, hour=10, category="groceries",
                        payment_method="debit_card")
        summary = ba.get_profile_summary("user-1")
        assert summary is not None
        assert summary.user_id == "user-1"
        assert summary.observation_count == 20
        assert summary.avg_amount > 0
        assert 10 in summary.preferred_hours
        assert len(summary.top_categories) >= 1
        assert summary.top_categories[0]["category"] == "groceries"
        assert len(summary.top_payment_methods) >= 1
        assert summary.top_payment_methods[0]["method"] == "debit_card"

    def test_preferred_days(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, n=20, weekday=4)  # Friday
        summary = ba.get_profile_summary("user-1")
        assert 4 in summary.preferred_days


# ── Profile management ───────────────────────────────────────────────────────


class TestProfileManagement:
    def test_remove_existing_profile(self):
        ba = BehavioralAnalytics()
        ba.ingest(_txn(user_id="u1"))
        assert ba.remove_profile("u1") is True
        assert ba.profile_count == 0

    def test_remove_nonexistent_profile(self):
        ba = BehavioralAnalytics()
        assert ba.remove_profile("ghost") is False

    def test_get_all_user_ids(self):
        ba = BehavioralAnalytics()
        ba.ingest(_txn(user_id="alice"))
        ba.ingest(_txn(user_id="bob"))
        ids = ba.get_all_user_ids()
        assert set(ids) == {"alice", "bob"}


# ── Adaptive learning (EMA behavior over time) ──────────────────────────────


class TestAdaptiveLearning:
    def test_ema_adapts_to_new_pattern(self):
        ba = BehavioralAnalytics(alpha=0.3, min_observations=3)
        # Phase 1: establish baseline with small amounts
        for _ in range(20):
            ba.ingest(_txn(amount=50.0))

        # A large amount should initially be anomalous
        result_before = ba.score(_txn(amount=500.0))

        # Phase 2: shift to larger amounts
        for _ in range(50):
            ba.ingest(_txn(amount=500.0))

        # Now 500 should be less anomalous
        result_after = ba.score(_txn(amount=500.0))
        assert result_after.amount_deviation < result_before.amount_deviation

    def test_higher_alpha_adapts_faster(self):
        ba_slow = BehavioralAnalytics(alpha=0.05, min_observations=3)
        ba_fast = BehavioralAnalytics(alpha=0.5, min_observations=3)

        # Build same baseline
        for ba in (ba_slow, ba_fast):
            for _ in range(20):
                ba.ingest(_txn(amount=50.0))

        # Shift pattern
        for ba in (ba_slow, ba_fast):
            for _ in range(10):
                ba.ingest(_txn(amount=200.0))

        # Fast adapter should see 200 as more normal
        result_slow = ba_slow.score(_txn(amount=200.0))
        result_fast = ba_fast.score(_txn(amount=200.0))
        assert result_fast.amount_deviation <= result_slow.amount_deviation


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_amount_transaction(self):
        ba = BehavioralAnalytics(min_observations=1)
        ba.ingest(_txn(amount=0.0))
        result = ba.score(_txn(amount=0.0))
        # Should not crash; deviation should be minimal
        assert result.amount_deviation >= 0.0

    def test_very_large_amount(self):
        ba = BehavioralAnalytics(alpha=0.2, min_observations=1)
        _build_baseline(ba, amount=10.0, n=10)
        result = ba.score(_txn(amount=1_000_000.0))
        assert result.amount_deviation > 0.5

    def test_single_category_profile(self):
        ba = BehavioralAnalytics(alpha=0.5, min_observations=1)
        _build_baseline(ba, category="groceries", n=10)
        # Same category
        result = ba.score(_txn(category="groceries"))
        assert result.category_deviation < 0.5

    def test_transaction_record_validation(self):
        with pytest.raises(ValidationError):
            TransactionRecord(
                user_id="u1",
                amount=-10.0,  # negative not allowed
                timestamp=datetime.now(UTC),
            )

    def test_deviation_result_bounds(self):
        ba = BehavioralAnalytics(alpha=0.2)
        _build_baseline(ba, n=20)
        result = ba.score(_txn())
        assert 0.0 <= result.composite_score <= 1.0
        assert 0.0 <= result.amount_deviation <= 1.0
        assert 0.0 <= result.time_deviation <= 1.0
        assert 0.0 <= result.category_deviation <= 1.0
        assert 0.0 <= result.payment_method_deviation <= 1.0
