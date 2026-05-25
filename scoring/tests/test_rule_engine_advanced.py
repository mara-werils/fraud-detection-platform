"""Advanced tests for the rule-based scoring engine.

Covers custom threshold configurations, edge cases, all rule combinations,
score clamping behaviour, and property-style tests across a range of amounts.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

import pytest

from scoring.models.rule_engine import RuleEngine, RuleResult, ScoringResult
from shared.schemas import FeatureVector, Transaction, TransactionType

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_USER_ID = UUID("aaaaaaaa-0000-0000-0000-000000000001")
_TXN_ID = UUID("bbbbbbbb-0000-0000-0000-000000000002")


def _txn(
    amount: Decimal = Decimal("100.00"),
    txn_type: TransactionType = TransactionType.PURCHASE,
    hour: int = 12,
) -> Transaction:
    """Build a minimal Transaction for testing."""
    return Transaction(
        transaction_id=_TXN_ID,
        user_id=_USER_ID,
        amount=amount,
        transaction_type=txn_type,
        timestamp=datetime(2025, 6, 15, hour, 0, 0),
    )


def _features(
    txn_count_1h: int = 1,
    txn_amount_avg_24h: Decimal = Decimal("100.00"),
    distance_km: float = 0.0,
    is_new_device: bool = False,
    unique_countries: int = 1,
) -> FeatureVector:
    """Build a minimal FeatureVector for testing."""
    return FeatureVector(
        transaction_id=_TXN_ID,
        user_id=_USER_ID,
        txn_count_1h=txn_count_1h,
        txn_amount_avg_24h=txn_amount_avg_24h,
        distance_from_last_txn_km=distance_km,
        is_new_device=is_new_device,
        unique_countries_24h=unique_countries,
    )


# ---------------------------------------------------------------------------
# Custom threshold configuration
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    """Verify that constructor parameters are honoured correctly."""

    def test_custom_high_amount_multiplier(self) -> None:
        """A tight multiplier of 1.5 should trigger on amounts 1.6x the avg."""
        engine = RuleEngine(high_amount_multiplier=1.5, high_amount_score=0.4)
        txn = _txn(amount=Decimal("160.00"))  # 1.6x avg of 100
        feat = _features(txn_amount_avg_24h=Decimal("100.00"))
        result = engine.rule_high_amount(txn, feat)
        assert result.triggered is True
        assert result.score == pytest.approx(0.4)

    def test_custom_high_amount_does_not_trigger_below_threshold(self) -> None:
        """At exactly the multiplier boundary (not exceeding) rule should not fire."""
        engine = RuleEngine(high_amount_multiplier=2.0)
        txn = _txn(amount=Decimal("200.00"))  # ratio == 2.0, not > 2.0
        feat = _features(txn_amount_avg_24h=Decimal("100.00"))
        result = engine.rule_high_amount(txn, feat)
        assert result.triggered is False

    def test_custom_velocity_threshold(self) -> None:
        """A lower velocity threshold of 3 should fire at 4 transactions."""
        engine = RuleEngine(velocity_threshold=3, velocity_score=0.25)
        feat = _features(txn_count_1h=4)
        result = engine.rule_velocity(_txn(), feat)
        assert result.triggered is True
        assert result.score == pytest.approx(0.25)

    def test_custom_velocity_not_at_exact_threshold(self) -> None:
        """Exactly at threshold (txn_count_1h == threshold) should NOT trigger."""
        engine = RuleEngine(velocity_threshold=5)
        feat = _features(txn_count_1h=5)  # not *greater* than 5
        result = engine.rule_velocity(_txn(), feat)
        assert result.triggered is False

    def test_custom_geo_distance(self) -> None:
        """A tighter geo threshold of 100 km should catch medium-distance travel."""
        engine = RuleEngine(geo_distance_km=100.0, geo_score=0.35)
        feat = _features(distance_km=150.0)
        result = engine.rule_geo_anomaly(_txn(), feat)
        assert result.triggered is True
        assert result.score == pytest.approx(0.35)

    def test_custom_night_window(self) -> None:
        """A wider night window (22-06) should trigger at 23:00."""
        engine = RuleEngine(night_start_hour=22, night_end_hour=6, night_score=0.2)
        txn = _txn(hour=23)
        result = engine.rule_night_transaction(txn, _features())
        assert result.triggered is True
        assert result.score == pytest.approx(0.2)

    def test_custom_new_device_score(self) -> None:
        """Custom new_device_score should appear in RuleResult."""
        engine = RuleEngine(new_device_score=0.5)
        feat = _features(is_new_device=True)
        result = engine.rule_new_device(_txn(), feat)
        assert result.score == pytest.approx(0.5)

    def test_custom_international_score(self) -> None:
        """Custom international_score should appear in RuleResult."""
        engine = RuleEngine(international_score=0.2)
        feat = _features(unique_countries=3)
        result = engine.rule_international(_txn(), feat)
        assert result.score == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_zero_amount(self, rule_engine: RuleEngine) -> None:
        """Zero-amount transaction: high_amount rule must not trigger."""
        txn = _txn(amount=Decimal("0.00"))
        feat = _features(txn_amount_avg_24h=Decimal("100.00"))
        result = rule_engine.rule_high_amount(txn, feat)
        assert result.triggered is False

    def test_zero_average_skips_high_amount(self, rule_engine: RuleEngine) -> None:
        """When avg is zero, high_amount rule should return score=0 (no division)."""
        txn = _txn(amount=Decimal("9999.99"))
        feat = _features(txn_amount_avg_24h=Decimal("0.00"))
        result = rule_engine.rule_high_amount(txn, feat)
        assert result.triggered is False
        assert result.score == 0.0

    def test_exactly_zero_distance(self, rule_engine: RuleEngine) -> None:
        """Distance of exactly 0 km should not trigger geo rule."""
        feat = _features(distance_km=0.0)
        result = rule_engine.rule_geo_anomaly(_txn(), feat)
        assert result.triggered is False

    def test_exactly_threshold_distance(self, rule_engine: RuleEngine) -> None:
        """Distance exactly at threshold (500 km) should NOT trigger (must be >)."""
        feat = _features(distance_km=500.0)
        result = rule_engine.rule_geo_anomaly(_txn(), feat)
        assert result.triggered is False

    def test_one_country_no_trigger(self, rule_engine: RuleEngine) -> None:
        """unique_countries_24h == 1 should not trigger international rule."""
        feat = _features(unique_countries=1)
        result = rule_engine.rule_international(_txn(), feat)
        assert result.triggered is False

    def test_zero_countries_no_trigger(self, rule_engine: RuleEngine) -> None:
        """unique_countries_24h == 0 should not trigger international rule."""
        feat = _features(unique_countries=0)
        result = rule_engine.rule_international(_txn(), feat)
        assert result.triggered is False

    def test_night_boundary_start(self, rule_engine: RuleEngine) -> None:
        """Hour == night_start_hour (1 AM) is inside window and must trigger."""
        txn = _txn(hour=1)
        result = rule_engine.rule_night_transaction(txn, _features())
        assert result.triggered is True

    def test_night_boundary_end(self, rule_engine: RuleEngine) -> None:
        """Hour == night_end_hour (5 AM) is NOT inside [1, 5) and must not trigger."""
        txn = _txn(hour=5)
        result = rule_engine.rule_night_transaction(txn, _features())
        assert result.triggered is False

    def test_midnight_no_trigger_default(self, rule_engine: RuleEngine) -> None:
        """Midnight (hour 0) is outside default window [1, 5) and should not trigger."""
        txn = _txn(hour=0)
        result = rule_engine.rule_night_transaction(txn, _features())
        assert result.triggered is False

    def test_very_large_amount(self, rule_engine: RuleEngine) -> None:
        """Very large amounts do not cause overflow or unexpected behaviour."""
        txn = _txn(amount=Decimal("999999999.99"))
        feat = _features(txn_amount_avg_24h=Decimal("1.00"))
        result = rule_engine.rule_high_amount(txn, feat)
        assert result.triggered is True


# ---------------------------------------------------------------------------
# All rule combinations
# ---------------------------------------------------------------------------


class TestRuleCombinations:
    """Confirm interaction between rules in the aggregated score."""

    def test_no_rules_trigger(self, rule_engine: RuleEngine) -> None:
        """All-safe inputs should produce an empty triggered list."""
        txn = _txn(amount=Decimal("50.00"), hour=14)
        feat = _features(
            txn_count_1h=2,
            txn_amount_avg_24h=Decimal("100.00"),
            distance_km=10.0,
            is_new_device=False,
            unique_countries=1,
        )
        result = rule_engine.score(txn, feat)
        assert result.triggered_rules == []
        assert result.fraud_score == 0.0

    def test_only_velocity_triggers(self, rule_engine: RuleEngine) -> None:
        """Only velocity should appear in triggered_rules."""
        txn = _txn(amount=Decimal("50.00"), hour=14)
        feat = _features(
            txn_count_1h=10,  # triggers velocity
            txn_amount_avg_24h=Decimal("100.00"),
            distance_km=10.0,
            is_new_device=False,
            unique_countries=1,
        )
        result = rule_engine.score(txn, feat)
        assert result.triggered_rules == ["velocity"]
        assert result.fraud_score == pytest.approx(0.15)

    def test_two_rules_additive(self, rule_engine: RuleEngine) -> None:
        """Two triggered rules should add their scores."""
        txn = _txn(amount=Decimal("50.00"), hour=14)
        feat = _features(
            txn_count_1h=10,
            is_new_device=True,
            txn_amount_avg_24h=Decimal("100.00"),
        )
        result = rule_engine.score(txn, feat)
        assert "velocity" in result.triggered_rules
        assert "new_device" in result.triggered_rules
        assert result.fraud_score == pytest.approx(0.15 + 0.20)

    def test_all_rules_trigger_and_clamp(self, rule_engine: RuleEngine) -> None:
        """All six rules trigger; score should be clamped to 1.0."""
        txn = _txn(amount=Decimal("99999.99"), hour=3, txn_type=TransactionType.TRANSFER)
        feat = _features(
            txn_count_1h=20,
            txn_amount_avg_24h=Decimal("10.00"),
            distance_km=5000.0,
            is_new_device=True,
            unique_countries=5,
        )
        result = rule_engine.score(txn, feat)
        assert set(result.triggered_rules) == {
            "high_amount",
            "new_device",
            "velocity",
            "geo_anomaly",
            "night_transaction",
            "international",
        }
        assert result.fraud_score == 1.0

    def test_rule_scores_dict_keys(self, rule_engine: RuleEngine) -> None:
        """ScoringResult.rule_scores must have an entry for every rule."""
        result = rule_engine.score(_txn(), _features())
        expected_keys = {
            "high_amount",
            "new_device",
            "velocity",
            "geo_anomaly",
            "night_transaction",
            "international",
        }
        assert set(result.rule_scores.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------


class TestScoreClamping:
    """Score must always remain in [0.0, 1.0]."""

    def test_clamp_at_maximum(self) -> None:
        """Scores summing > 1.0 must be clamped to exactly 1.0."""
        # Use custom weights so the sum is guaranteed to exceed 1.0
        engine = RuleEngine(
            high_amount_score=0.5,
            new_device_score=0.5,
            velocity_score=0.5,
            geo_score=0.5,
            night_score=0.5,
            international_score=0.5,
        )
        txn = _txn(amount=Decimal("9999.99"), hour=3)
        feat = _features(
            txn_count_1h=20,
            txn_amount_avg_24h=Decimal("10.00"),
            distance_km=5000.0,
            is_new_device=True,
            unique_countries=5,
        )
        result = engine.score(txn, feat)
        assert result.fraud_score == 1.0

    def test_clamp_at_minimum(self, rule_engine: RuleEngine) -> None:
        """All-zero scores must produce exactly 0.0."""
        txn = _txn(amount=Decimal("5.00"), hour=12)
        feat = _features(
            txn_count_1h=1,
            txn_amount_avg_24h=Decimal("100.00"),
            distance_km=1.0,
            is_new_device=False,
            unique_countries=1,
        )
        result = rule_engine.score(txn, feat)
        assert result.fraud_score == 0.0

    def test_fraud_score_always_in_range(self, rule_engine: RuleEngine) -> None:
        """Exhaustive check: score is in [0, 1] for multiple amount values."""
        amounts = [0, 1, 10, 100, 500, 1000, 5000, 50000]
        for amt in amounts:
            txn = _txn(amount=Decimal(str(amt)), hour=3)
            feat = _features(
                txn_count_1h=10,
                txn_amount_avg_24h=Decimal("100.00"),
                distance_km=600.0,
                is_new_device=True,
                unique_countries=3,
            )
            result = rule_engine.score(txn, feat)
            assert 0.0 <= result.fraud_score <= 1.0, (
                f"score {result.fraud_score} out of range for amount={amt}"
            )


# ---------------------------------------------------------------------------
# Property-style tests with various amounts
# ---------------------------------------------------------------------------


class TestPropertyStyle:
    """Parametrised checks across diverse amounts and contexts."""

    @pytest.mark.parametrize(
        "amount,avg,expect_trigger",
        [
            (Decimal("1.00"), Decimal("100.00"), False),   # 0.01x
            (Decimal("299.99"), Decimal("100.00"), False),  # 2.9x, just below 3x
            (Decimal("300.01"), Decimal("100.00"), True),   # just above 3x
            (Decimal("1000.00"), Decimal("100.00"), True),  # 10x
            (Decimal("0.01"), Decimal("0.01"), False),      # ratio == 1
        ],
    )
    def test_high_amount_boundary(
        self,
        rule_engine: RuleEngine,
        amount: Decimal,
        avg: Decimal,
        expect_trigger: bool,
    ) -> None:
        """high_amount rule respects the 3x multiplier boundary exactly."""
        txn = _txn(amount=amount)
        feat = _features(txn_amount_avg_24h=avg)
        result = rule_engine.rule_high_amount(txn, feat)
        assert result.triggered is expect_trigger

    @pytest.mark.parametrize("count,expected", [(0, False), (5, False), (6, True), (100, True)])
    def test_velocity_parametric(
        self,
        rule_engine: RuleEngine,
        count: int,
        expected: bool,
    ) -> None:
        """Velocity triggers strictly above the threshold."""
        feat = _features(txn_count_1h=count)
        result = rule_engine.rule_velocity(_txn(), feat)
        assert result.triggered is expected

    @pytest.mark.parametrize(
        "distance,expected",
        [(0.0, False), (499.9, False), (500.0, False), (500.1, True), (9999.0, True)],
    )
    def test_geo_parametric(
        self,
        rule_engine: RuleEngine,
        distance: float,
        expected: bool,
    ) -> None:
        """Geo rule triggers strictly above 500 km."""
        feat = _features(distance_km=distance)
        result = rule_engine.rule_geo_anomaly(_txn(), feat)
        assert result.triggered is expected

    @pytest.mark.parametrize(
        "hour,expected",
        [(0, False), (1, True), (2, True), (4, True), (5, False), (12, False), (23, False)],
    )
    def test_night_hours(
        self,
        rule_engine: RuleEngine,
        hour: int,
        expected: bool,
    ) -> None:
        """Night rule uses [1, 5) window: 1,2,3,4 trigger; 0 and 5+ do not."""
        txn = _txn(hour=hour)
        result = rule_engine.rule_night_transaction(txn, _features())
        assert result.triggered is expected

    @pytest.mark.parametrize("countries,expected", [(0, False), (1, False), (2, True), (10, True)])
    def test_international_parametric(
        self,
        rule_engine: RuleEngine,
        countries: int,
        expected: bool,
    ) -> None:
        """International rule triggers when unique_countries_24h > 1."""
        feat = _features(unique_countries=countries)
        result = rule_engine.rule_international(_txn(), feat)
        assert result.triggered is expected

    def test_scoring_result_is_dataclass(self, rule_engine: RuleEngine) -> None:
        """score() must return a ScoringResult with expected attributes."""
        result = rule_engine.score(_txn(), _features())
        assert isinstance(result, ScoringResult)
        assert hasattr(result, "fraud_score")
        assert hasattr(result, "rule_scores")
        assert hasattr(result, "triggered_rules")
        assert hasattr(result, "details")

    def test_rule_result_frozen(self, rule_engine: RuleEngine) -> None:
        """RuleResult is frozen; mutation should raise AttributeError."""
        result = rule_engine.rule_velocity(_txn(), _features())
        assert isinstance(result, RuleResult)
        with pytest.raises(AttributeError):
            result.score = 99.0  # type: ignore[misc]
