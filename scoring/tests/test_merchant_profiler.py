"""Tests for the MerchantProfiler service."""

from datetime import UTC, datetime, timedelta

import pytest

from scoring.services.merchant_profiler import (
    MCC_TIER_MAP,
    MCCRiskTier,
    MerchantProfiler,
    MerchantRiskProfile,
    is_high_risk_category,
    mcc_to_tier,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def _build_profiler(**kwargs) -> MerchantProfiler:
    return MerchantProfiler(**kwargs)


def _register_and_seed(
    profiler: MerchantProfiler,
    merchant_id: str = "m-001",
    name: str = "Test Merchant",
    mcc: str = "5999",
    category: str = "general",
    country: str = "US",
    registered_at: datetime | None = None,
    txn_count: int = 10,
    fraud_count: int = 0,
    chargeback_count: int = 0,
    amount: float = 100.0,
    base_time: datetime | None = None,
) -> None:
    """Register a merchant and populate it with transactions."""
    reg_time = registered_at or (NOW - timedelta(days=180))
    base = base_time or (NOW - timedelta(hours=2))
    profiler.register_merchant(
        merchant_id, name, mcc, category, country, registered_at=reg_time
    )
    for i in range(txn_count):
        ts = base + timedelta(minutes=i * 5)
        is_fraud = i < fraud_count
        is_cb = i < chargeback_count
        profiler.record_transaction(
            merchant_id,
            amount,
            user_id=f"user-{i}",
            is_fraud=is_fraud,
            is_chargeback=is_cb,
            timestamp=ts,
        )


# ── MCC tier mapping ────────────────────────────────────────────────────────


class TestMCCRiskTiers:
    def test_known_gambling_mcc_is_critical(self):
        assert mcc_to_tier("7995") == MCCRiskTier.CRITICAL

    def test_known_crypto_mcc_is_critical(self):
        assert mcc_to_tier("6051") == MCCRiskTier.CRITICAL

    def test_known_adult_mcc_is_high(self):
        assert mcc_to_tier("5967") == MCCRiskTier.HIGH

    def test_unknown_mcc_defaults_to_standard(self):
        assert mcc_to_tier("0000") == MCCRiskTier.STANDARD

    def test_grocery_mcc_is_low(self):
        assert mcc_to_tier("5411") == MCCRiskTier.LOW

    def test_electronics_mcc_is_elevated(self):
        assert mcc_to_tier("5732") == MCCRiskTier.ELEVATED

    def test_all_mapped_mccs_resolve(self):
        for mcc, expected_tier in MCC_TIER_MAP.items():
            assert mcc_to_tier(mcc) == expected_tier


# ── High-risk category helper ───────────────────────────────────────────────


class TestHighRiskCategory:
    def test_gambling_is_high_risk(self):
        assert is_high_risk_category("gambling") is True

    def test_crypto_is_high_risk(self):
        assert is_high_risk_category("crypto") is True

    def test_adult_is_high_risk(self):
        assert is_high_risk_category("adult") is True

    def test_case_insensitive(self):
        assert is_high_risk_category("GAMBLING") is True
        assert is_high_risk_category("Crypto") is True

    def test_general_not_high_risk(self):
        assert is_high_risk_category("general") is False

    def test_food_not_high_risk(self):
        assert is_high_risk_category("food") is False


# ── Merchant registration ────────────────────────────────────────────────────


class TestMerchantRegistration:
    def test_register_returns_info(self):
        p = _build_profiler()
        info = p.register_merchant("m-1", "Shop", "5999")
        assert info.merchant_id == "m-1"
        assert info.name == "Shop"
        assert info.mcc == "5999"

    def test_register_is_idempotent(self):
        p = _build_profiler()
        info1 = p.register_merchant("m-1", "Shop", "5999")
        info2 = p.register_merchant("m-1", "Different Name", "7995")
        assert info1 is info2  # same object returned
        assert info2.name == "Shop"  # original name preserved

    def test_country_uppercased(self):
        p = _build_profiler()
        info = p.register_merchant("m-1", "Shop", "5999", country="gb")
        assert info.country == "GB"


# ── Transaction recording ────────────────────────────────────────────────────


class TestRecordTransaction:
    def test_record_increments_stats(self):
        p = _build_profiler()
        p.register_merchant("m-1", "Shop", "5999")
        p.record_transaction("m-1", 50.0, "u-1")
        p.record_transaction("m-1", 75.0, "u-2")
        cb = p.get_chargeback_stats("m-1")
        assert cb.total_transactions == 2
        assert cb.total_chargebacks == 0

    def test_record_chargeback(self):
        p = _build_profiler()
        p.register_merchant("m-1", "Shop", "5999")
        p.record_transaction("m-1", 100.0, "u-1", is_chargeback=True)
        p.record_transaction("m-1", 200.0, "u-2", is_chargeback=False)
        cb = p.get_chargeback_stats("m-1")
        assert cb.total_chargebacks == 1
        assert cb.ratio == pytest.approx(0.5)
        assert cb.chargeback_amounts == pytest.approx(100.0)

    def test_record_unknown_merchant_raises(self):
        p = _build_profiler()
        with pytest.raises(KeyError, match="Unknown merchant"):
            p.record_transaction("no-such", 10.0, "u-1")


# ── Chargeback stats ─────────────────────────────────────────────────────────


class TestChargebackStats:
    def test_ratio_zero_when_no_transactions(self):
        p = _build_profiler()
        p.register_merchant("m-1", "Shop", "5999")
        assert p.get_chargeback_stats("m-1").ratio == 0.0

    def test_amount_ratio(self):
        p = _build_profiler()
        p.register_merchant("m-1", "Shop", "5999")
        p.record_transaction("m-1", 200.0, "u-1", is_chargeback=True)
        p.record_transaction("m-1", 800.0, "u-2", is_chargeback=False)
        cb = p.get_chargeback_stats("m-1")
        assert cb.amount_ratio == pytest.approx(0.2)

    def test_unknown_merchant_raises(self):
        p = _build_profiler()
        with pytest.raises(KeyError):
            p.get_chargeback_stats("ghost")


# ── New merchant detection ───────────────────────────────────────────────────


class TestNewMerchantDetection:
    def test_merchant_within_scrutiny_period(self):
        p = _build_profiler(new_merchant_days=90)
        p.register_merchant(
            "m-new", "New Shop", "5999",
            registered_at=NOW - timedelta(days=30),
        )
        assert p.is_new_merchant("m-new", now=NOW) is True

    def test_merchant_past_scrutiny_period(self):
        p = _build_profiler(new_merchant_days=90)
        p.register_merchant(
            "m-old", "Old Shop", "5999",
            registered_at=NOW - timedelta(days=120),
        )
        assert p.is_new_merchant("m-old", now=NOW) is False

    def test_edge_case_exactly_on_boundary(self):
        p = _build_profiler(new_merchant_days=90)
        p.register_merchant(
            "m-edge", "Edge", "5999",
            registered_at=NOW - timedelta(days=90),
        )
        # 90 days old with 90-day window -> NOT new (age >= threshold)
        assert p.is_new_merchant("m-edge", now=NOW) is False

    def test_unknown_merchant_raises(self):
        p = _build_profiler()
        with pytest.raises(KeyError):
            p.is_new_merchant("nope")


# ── Velocity anomaly detection ───────────────────────────────────────────────


class TestVelocityAnomaly:
    def test_no_events_no_anomaly(self):
        p = _build_profiler()
        p.register_merchant("m-1", "Shop", "5999")
        is_anom, curr, mean = p.detect_velocity_anomaly("m-1", now=NOW)
        assert is_anom is False
        assert curr == 0

    def test_normal_velocity_no_anomaly(self):
        p = _build_profiler()
        p.register_merchant("m-1", "Shop", "5999", registered_at=NOW - timedelta(days=200))
        # Spread 100 txns evenly over 10 hours before NOW
        for i in range(100):
            hour_offset = (i % 10) + 2  # hours 2..11 ago
            ts = NOW - timedelta(hours=hour_offset, minutes=i % 60)
            p.record_transaction("m-1", 50.0, f"u-{i}", timestamp=ts)
        is_anom, _, _ = p.detect_velocity_anomaly("m-1", now=NOW)
        assert is_anom is False

    def test_spike_detected(self):
        p = _build_profiler(velocity_stddev_multiplier=2.0)
        p.register_merchant("m-1", "Shop", "5999", registered_at=NOW - timedelta(days=200))
        # Baseline: 5 txns per hour for hours 2–12 ago
        for h in range(2, 13):
            for j in range(5):
                ts = NOW - timedelta(hours=h, minutes=j * 10)
                p.record_transaction("m-1", 50.0, f"u-{h}-{j}", timestamp=ts)
        # Spike: 50 txns in the last hour
        for j in range(50):
            ts = NOW - timedelta(minutes=j)
            p.record_transaction("m-1", 50.0, f"u-spike-{j}", timestamp=ts)
        is_anom, curr, mean = p.detect_velocity_anomaly("m-1", now=NOW)
        assert is_anom is True
        assert curr == 50

    def test_unknown_merchant_raises(self):
        p = _build_profiler()
        with pytest.raises(KeyError):
            p.detect_velocity_anomaly("nope")


# ── Full profiling ───────────────────────────────────────────────────────────


class TestProfile:
    def test_clean_merchant_low_risk(self):
        p = _build_profiler()
        _register_and_seed(
            p, mcc="5411", category="food",
            txn_count=100, fraud_count=0, chargeback_count=0,
        )
        profile = p.profile("m-001", now=NOW)
        assert isinstance(profile, MerchantRiskProfile)
        assert profile.risk_score < 0.30
        assert profile.reputation_score > 0.50
        assert profile.fraud_rate == 0.0
        assert profile.chargeback_ratio == 0.0
        assert profile.is_new_merchant is False

    def test_high_fraud_rate_elevates_risk(self):
        p = _build_profiler()
        _register_and_seed(
            p, mcc="5999", txn_count=100, fraud_count=30,
        )
        profile = p.profile("m-001", now=NOW)
        assert profile.fraud_rate == pytest.approx(0.30)
        assert profile.risk_score > 0.40
        assert any("Fraud rate" in f for f in profile.risk_factors)

    def test_high_chargeback_triggers_critical_factor(self):
        p = _build_profiler(chargeback_critical_threshold=0.02)
        _register_and_seed(
            p, mcc="5999", txn_count=100, chargeback_count=5,
        )
        profile = p.profile("m-001", now=NOW)
        assert profile.chargeback_ratio == pytest.approx(0.05)
        assert any("critical threshold" in f for f in profile.risk_factors)

    def test_chargeback_warn_threshold(self):
        p = _build_profiler(
            chargeback_warn_threshold=0.01,
            chargeback_critical_threshold=0.05,
        )
        _register_and_seed(
            p, mcc="5999", txn_count=100, chargeback_count=2,
        )
        profile = p.profile("m-001", now=NOW)
        assert profile.chargeback_ratio == pytest.approx(0.02)
        assert any("warning threshold" in f for f in profile.risk_factors)

    def test_gambling_mcc_critical_tier(self):
        p = _build_profiler()
        _register_and_seed(p, mcc="7995", category="gambling", txn_count=20)
        profile = p.profile("m-001", now=NOW)
        assert profile.mcc_risk_tier == MCCRiskTier.CRITICAL
        assert any("MCC 7995" in f for f in profile.risk_factors)
        assert any("High-risk merchant category" in f for f in profile.risk_factors)

    def test_crypto_category_high_risk(self):
        p = _build_profiler()
        _register_and_seed(p, mcc="6051", category="crypto", txn_count=20)
        profile = p.profile("m-001", now=NOW)
        assert profile.mcc_risk_tier == MCCRiskTier.CRITICAL
        assert is_high_risk_category(profile.category)

    def test_new_merchant_flagged(self):
        p = _build_profiler(new_merchant_days=90)
        _register_and_seed(
            p, registered_at=NOW - timedelta(days=10), txn_count=5,
        )
        profile = p.profile("m-001", now=NOW)
        assert profile.is_new_merchant is True
        assert any("New merchant" in f for f in profile.risk_factors)

    def test_old_merchant_not_flagged_as_new(self):
        p = _build_profiler(new_merchant_days=90)
        _register_and_seed(
            p, registered_at=NOW - timedelta(days=365), txn_count=50,
        )
        profile = p.profile("m-001", now=NOW)
        assert profile.is_new_merchant is False
        assert not any("New merchant" in f for f in profile.risk_factors)

    def test_to_dict_keys(self):
        p = _build_profiler()
        _register_and_seed(p, txn_count=5)
        d = p.profile("m-001", now=NOW).to_dict()
        expected_keys = {
            "merchant_id", "name", "mcc", "mcc_risk_tier", "category",
            "country", "risk_score", "reputation_score", "risk_factors",
            "fraud_rate", "chargeback_ratio", "chargeback_amount_ratio",
            "is_new_merchant", "merchant_age_days", "first_seen", "last_seen",
            "total_transactions", "total_volume", "velocity_current",
            "velocity_mean", "velocity_anomaly", "assessed_at",
        }
        assert set(d.keys()) == expected_keys

    def test_unknown_merchant_raises(self):
        p = _build_profiler()
        with pytest.raises(KeyError):
            p.profile("nonexistent")


# ── Reputation scoring ───────────────────────────────────────────────────────


class TestReputationScoring:
    def test_clean_old_merchant_high_reputation(self):
        p = _build_profiler()
        _register_and_seed(
            p, mcc="5411", category="food",
            registered_at=NOW - timedelta(days=500),
            txn_count=500, fraud_count=0, chargeback_count=0,
        )
        profile = p.profile("m-001", now=NOW)
        assert profile.reputation_score > 0.70

    def test_dirty_merchant_low_reputation(self):
        p = _build_profiler()
        _register_and_seed(
            p, mcc="7995", category="gambling",
            txn_count=100, fraud_count=30, chargeback_count=10,
        )
        profile = p.profile("m-001", now=NOW)
        assert profile.reputation_score < 0.30

    def test_new_merchant_moderate_reputation(self):
        p = _build_profiler()
        _register_and_seed(
            p, registered_at=NOW - timedelta(days=5), txn_count=10,
        )
        profile = p.profile("m-001", now=NOW)
        # New merchant with clean history: moderate (age penalty).
        assert 0.20 < profile.reputation_score < 0.80

    def test_reputation_bounded_zero_one(self):
        p = _build_profiler()
        _register_and_seed(
            p, txn_count=200, fraud_count=180, chargeback_count=150,
        )
        profile = p.profile("m-001", now=NOW)
        assert 0.0 <= profile.reputation_score <= 1.0


# ── Bulk profiling ───────────────────────────────────────────────────────────


class TestProfileAll:
    def test_returns_sorted_by_risk_desc(self):
        p = _build_profiler()
        # Low risk
        _register_and_seed(
            p, merchant_id="m-clean", name="Clean",
            mcc="5411", category="food", txn_count=100,
        )
        # High risk
        _register_and_seed(
            p, merchant_id="m-dirty", name="Dirty",
            mcc="7995", category="gambling",
            txn_count=100, fraud_count=25, chargeback_count=10,
        )
        profiles = p.profile_all()
        assert len(profiles) == 2
        assert profiles[0].merchant_id == "m-dirty"
        assert profiles[1].merchant_id == "m-clean"
        assert profiles[0].risk_score >= profiles[1].risk_score

    def test_empty_profiler_returns_empty(self):
        p = _build_profiler()
        assert p.profile_all() == []
