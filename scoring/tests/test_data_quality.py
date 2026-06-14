"""Tests for the data quality monitoring module."""

from __future__ import annotations

import random
import time

import pytest

from scoring.services.data_quality import (
    DataDriftDetector,
    DataQualityMonitor,
    FreshnessMonitor,
    NullFieldMonitor,
    OutlierDetector,
    SchemaValidator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_record(**overrides) -> dict:
    base = {
        "transaction_id": "txn-001",
        "user_id": "user-001",
        "amount": 100.0,
        "currency": "USD",
        "transaction_type": "purchase",
        "timestamp": "2025-06-15T14:30:00",
        "merchant_id": "merchant_001",
        "device_id": "device_abc",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------


class TestSchemaValidator:
    def setup_method(self):
        self.validator = SchemaValidator()

    def test_valid_record_passes(self):
        result = self.validator.validate(_valid_record())
        assert result.valid is True
        assert result.violations == []

    def test_missing_required_field(self):
        rec = _valid_record()
        del rec["amount"]
        result = self.validator.validate(rec)
        assert result.valid is False
        assert any(v.field == "amount" and v.violation_type == "missing" for v in result.violations)

    def test_null_required_field(self):
        rec = _valid_record(user_id=None)
        result = self.validator.validate(rec)
        assert result.valid is False
        assert "user_id" in result.null_fields

    def test_wrong_type_required_field(self):
        rec = _valid_record(amount="not_a_number")
        result = self.validator.validate(rec)
        assert result.valid is False
        assert any(v.field == "amount" and v.violation_type == "wrong_type" for v in result.violations)

    def test_invalid_transaction_type(self):
        rec = _valid_record(transaction_type="invalid_type")
        result = self.validator.validate(rec)
        assert result.valid is False
        assert any(v.field == "transaction_type" and v.violation_type == "invalid_value" for v in result.violations)

    def test_invalid_currency(self):
        rec = _valid_record(currency="us")
        result = self.validator.validate(rec)
        assert result.valid is False
        assert any(v.field == "currency" for v in result.violations)

    def test_negative_amount(self):
        rec = _valid_record(amount=-50.0)
        result = self.validator.validate(rec)
        assert result.valid is False
        assert any(v.field == "amount" and v.violation_type == "invalid_value" for v in result.violations)

    def test_optional_field_wrong_type(self):
        rec = _valid_record(latitude="not_a_float")
        result = self.validator.validate(rec)
        assert result.valid is False
        assert any(v.field == "latitude" for v in result.violations)

    def test_optional_field_null_tracked(self):
        rec = _valid_record(merchant_id=None)
        result = self.validator.validate(rec)
        # null optional field is valid but tracked
        assert result.valid is True
        assert "merchant_id" in result.null_fields

    def test_to_dict(self):
        rec = _valid_record()
        del rec["amount"]
        result = self.validator.validate(rec)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["valid"] is False
        assert len(d["violations"]) > 0


# ---------------------------------------------------------------------------
# NullFieldMonitor
# ---------------------------------------------------------------------------


class TestNullFieldMonitor:
    def test_no_nulls_yields_zero_rates(self):
        monitor = NullFieldMonitor(default_threshold=0.1)
        for _ in range(10):
            monitor.observe(_valid_record())
        rates = monitor.null_rates()
        # Required fields should have 0 null rate
        assert rates.get("transaction_id", 0.0) == 0.0
        assert rates.get("amount", 0.0) == 0.0

    def test_high_null_rate_detected(self):
        monitor = NullFieldMonitor(default_threshold=0.2, window_size=100)
        # 50% of records missing merchant_id
        for i in range(100):
            rec = _valid_record(merchant_id=None if i % 2 == 0 else "m1")
            monitor.observe(rec)
        rates = monitor.null_rates()
        assert rates["merchant_id"] == pytest.approx(0.5, abs=0.05)

    def test_breached_fields(self):
        monitor = NullFieldMonitor(
            thresholds={"merchant_id": 0.3},
            default_threshold=0.9,
            window_size=100,
        )
        for i in range(100):
            rec = _valid_record(merchant_id=None if i % 2 == 0 else "m1")
            monitor.observe(rec)
        breached = monitor.breached_fields()
        assert "merchant_id" in breached

    def test_no_breach_under_threshold(self):
        # Supply all optional fields so nothing registers as null
        full_record = _valid_record(
            merchant_category="retail",
            card_number_hash="abc123",
            ip_address="1.2.3.4",
            latitude=40.0,
            longitude=-74.0,
        )
        monitor = NullFieldMonitor(default_threshold=0.9, window_size=100)
        for _ in range(100):
            monitor.observe(full_record)
        breached = monitor.breached_fields()
        assert len(breached) == 0

    def test_configurable_thresholds(self):
        monitor = NullFieldMonitor(
            thresholds={"device_id": 0.05},
            default_threshold=0.9,
            window_size=100,
        )
        for i in range(100):
            rec = _valid_record(device_id=None if i < 10 else "dev1")
            monitor.observe(rec)
        breached = monitor.breached_fields()
        assert "device_id" in breached


# ---------------------------------------------------------------------------
# FreshnessMonitor
# ---------------------------------------------------------------------------


class TestFreshnessMonitor:
    def test_no_data_is_stale(self):
        monitor = FreshnessMonitor(max_staleness_seconds=60)
        assert monitor.is_stale("source_a") is True

    def test_fresh_after_arrival(self):
        monitor = FreshnessMonitor(max_staleness_seconds=60)
        monitor.record_arrival("source_a")
        assert monitor.is_stale("source_a") is False

    def test_freshness_seconds(self):
        monitor = FreshnessMonitor(max_staleness_seconds=60)
        assert monitor.freshness_seconds("source_a") is None
        monitor.record_arrival("source_a")
        fs = monitor.freshness_seconds("source_a")
        assert fs is not None
        assert fs < 1.0  # should be nearly instant

    def test_delay_calculation(self):
        monitor = FreshnessMonitor()
        past_ts = time.time() - 10.0
        delay = monitor.record_arrival("source_a", record_timestamp=past_ts)
        assert delay >= 9.5  # approximately 10 seconds

    def test_avg_delay(self):
        monitor = FreshnessMonitor()
        now = time.time()
        for _ in range(5):
            monitor.record_arrival("src", record_timestamp=now - 10)
        avg = monitor.avg_delay("src")
        assert avg is not None
        assert avg >= 9.5

    def test_summary(self):
        monitor = FreshnessMonitor(max_staleness_seconds=60)
        monitor.record_arrival("src_a")
        monitor.record_arrival("src_b")
        summary = monitor.summary()
        assert "src_a" in summary
        assert "src_b" in summary
        assert summary["src_a"]["stale"] is False


# ---------------------------------------------------------------------------
# OutlierDetector
# ---------------------------------------------------------------------------


class TestOutlierDetector:
    def test_needs_min_samples(self):
        detector = OutlierDetector(min_samples=30)
        # fewer than 30 samples -> no z-scores
        zscores = detector.update({"amount": 100.0})
        assert zscores == {}

    def test_normal_values_not_outliers(self):
        detector = OutlierDetector(zscore_threshold=3.0, min_samples=10)
        rng = random.Random(42)
        for _ in range(50):
            detector.update({"amount": rng.gauss(100, 10)})
        # A value close to the mean should not be an outlier
        outliers = detector.is_outlier({"amount": 105.0})
        assert outliers.get("amount") is False

    def test_extreme_value_is_outlier(self):
        detector = OutlierDetector(zscore_threshold=3.0, min_samples=10)
        rng = random.Random(42)
        for _ in range(100):
            detector.update({"amount": rng.gauss(100, 5)})
        outliers = detector.is_outlier({"amount": 500.0})
        assert outliers.get("amount") is True

    def test_stats_returned(self):
        detector = OutlierDetector(min_samples=5)
        rng = random.Random(42)
        for _ in range(20):
            detector.update({"feat_a": rng.gauss(0, 1)})
        stats = detector.stats()
        assert "feat_a" in stats
        assert stats["feat_a"]["count"] == 20
        assert "mean" in stats["feat_a"]
        assert "std" in stats["feat_a"]

    def test_multiple_features(self):
        detector = OutlierDetector(zscore_threshold=3.0, min_samples=10)
        rng = random.Random(42)
        for _ in range(50):
            detector.update({"a": rng.gauss(0, 1), "b": rng.gauss(100, 10)})
        outliers = detector.is_outlier({"a": 0.5, "b": 500.0})
        assert outliers.get("a") is False
        assert outliers.get("b") is True


# ---------------------------------------------------------------------------
# DataDriftDetector
# ---------------------------------------------------------------------------


class TestDataDriftDetector:
    def test_insufficient_data_returns_none(self):
        dd = DataDriftDetector(reference_window=100, current_window=50)
        assert dd.check_drift() is None

    def test_no_drift_on_identical_distributions(self):
        dd = DataDriftDetector(
            reference_window=2000,
            current_window=1000,
            psi_threshold=0.2,
            kl_threshold=0.5,
            monitored_features=["feat_a"],
        )
        rng = random.Random(42)
        for _ in range(2000):
            dd.add_sample({"feat_a": rng.gauss(0, 1)})
        for _ in range(1000):
            dd.add_sample({"feat_a": rng.gauss(0, 1)})
        results = dd.check_drift()
        assert results is not None
        assert len(results) == 1
        assert results[0].drift_detected is False

    def test_drift_on_shifted_distribution(self):
        dd = DataDriftDetector(
            reference_window=200,
            current_window=100,
            psi_threshold=0.1,
            kl_threshold=0.3,
            monitored_features=["feat_a"],
        )
        rng = random.Random(42)
        # reference: mean=0
        for _ in range(200):
            dd.add_sample({"feat_a": rng.gauss(0, 1)})
        # current: mean=10 (big shift)
        for _ in range(100):
            dd.add_sample({"feat_a": rng.gauss(10, 1)})
        results = dd.check_drift()
        assert results is not None
        assert results[0].drift_detected is True
        assert results[0].psi > 0.1
        assert results[0].kl_divergence > 0.3

    def test_psi_identical_is_near_zero(self):
        data = [float(i) for i in range(200)]
        psi = DataDriftDetector._compute_psi(data, data)
        assert psi < 0.01

    def test_kl_identical_is_near_zero(self):
        data = [float(i) for i in range(200)]
        kl = DataDriftDetector._compute_kl_divergence(data, data)
        assert kl < 0.01

    def test_kl_divergence_positive_on_shift(self):
        rng = random.Random(42)
        ref = [rng.gauss(0, 1) for _ in range(500)]
        cur = [rng.gauss(5, 1) for _ in range(500)]
        kl = DataDriftDetector._compute_kl_divergence(ref, cur)
        assert kl > 0.5

    def test_psi_empty_data(self):
        assert DataDriftDetector._compute_psi([], [1.0, 2.0]) == 0.0
        assert DataDriftDetector._compute_psi([1.0, 2.0], []) == 0.0

    def test_kl_empty_data(self):
        assert DataDriftDetector._compute_kl_divergence([], [1.0]) == 0.0

    def test_latest_results(self):
        dd = DataDriftDetector(
            reference_window=100,
            current_window=50,
            monitored_features=["f"],
        )
        assert dd.latest_results() is None
        rng = random.Random(42)
        for _ in range(100):
            dd.add_sample({"f": rng.gauss(0, 1)})
        for _ in range(50):
            dd.add_sample({"f": rng.gauss(0, 1)})
        dd.check_drift()
        assert dd.latest_results() is not None

    def test_constant_distribution_no_drift(self):
        """Constant values should yield PSI=0."""
        ref = [5.0] * 100
        cur = [5.0] * 100
        assert DataDriftDetector._compute_psi(ref, cur) == 0.0
        assert DataDriftDetector._compute_kl_divergence(ref, cur) == 0.0


# ---------------------------------------------------------------------------
# DataQualityMonitor (integration)
# ---------------------------------------------------------------------------


class TestDataQualityMonitor:
    def test_process_valid_record(self):
        monitor = DataQualityMonitor()
        result = monitor.process_record(_valid_record(), source="api")
        assert result.valid is True

    def test_process_invalid_record(self):
        monitor = DataQualityMonitor()
        rec = _valid_record()
        del rec["user_id"]
        result = monitor.process_record(rec, source="api")
        assert result.valid is False

    def test_quality_score_no_data(self):
        monitor = DataQualityMonitor()
        score = monitor.compute_quality_score("nonexistent")
        assert score.score == 0.0
        assert score.total_records == 0

    def test_quality_score_all_valid(self):
        monitor = DataQualityMonitor(max_staleness_seconds=60)
        for _ in range(50):
            monitor.process_record(_valid_record(), source="src")
        score = monitor.compute_quality_score("src")
        assert score.total_records == 50
        assert score.valid_records == 50
        assert score.score >= 0.5  # should be high since all valid

    def test_quality_score_with_invalid_records(self):
        monitor = DataQualityMonitor(max_staleness_seconds=60)
        for i in range(100):
            rec = _valid_record() if i % 2 == 0 else _valid_record(amount="bad")
            monitor.process_record(rec, source="src")
        score = monitor.compute_quality_score("src")
        assert score.valid_records == 50
        assert score.score < 0.8  # should be penalised

    def test_quality_score_to_dict(self):
        monitor = DataQualityMonitor(max_staleness_seconds=60)
        for _ in range(10):
            monitor.process_record(_valid_record(), source="s1")
        score = monitor.compute_quality_score("s1")
        d = score.to_dict()
        assert isinstance(d, dict)
        assert "score" in d
        assert "source" in d
        assert d["source"] == "s1"

    def test_get_latest_score(self):
        monitor = DataQualityMonitor()
        assert monitor.get_latest_score("src") is None
        for _ in range(5):
            monitor.process_record(_valid_record(), source="src")
        monitor.compute_quality_score("src")
        latest = monitor.get_latest_score("src")
        assert latest is not None
        assert latest.source == "src"

    def test_summary(self):
        monitor = DataQualityMonitor()
        for _ in range(5):
            monitor.process_record(_valid_record(), source="src")
        monitor.compute_quality_score("src")
        s = monitor.summary()
        assert "sources" in s
        assert "freshness" in s
        assert "null_breaches" in s
        assert "outlier_stats" in s

    def test_multiple_sources_tracked_independently(self):
        monitor = DataQualityMonitor(max_staleness_seconds=60)
        for _ in range(20):
            monitor.process_record(_valid_record(), source="good")
        for _ in range(20):
            monitor.process_record(_valid_record(amount="bad"), source="bad")
        good_score = monitor.compute_quality_score("good")
        bad_score = monitor.compute_quality_score("bad")
        assert good_score.valid_records == 20
        assert bad_score.valid_records == 0
        assert good_score.score > bad_score.score

    def test_drift_detection_integration(self):
        monitor = DataQualityMonitor(
            reference_window=200,
            current_window=100,
            psi_threshold=0.1,
            kl_threshold=0.3,
            monitored_features=["amount"],
            max_staleness_seconds=60,
        )
        rng = random.Random(42)
        # reference phase: normal amounts
        for _ in range(200):
            rec = _valid_record(amount=rng.gauss(100, 10))
            monitor.process_record(rec, source="src")
        # current phase: shifted amounts
        for _ in range(100):
            rec = _valid_record(amount=rng.gauss(500, 10))
            monitor.process_record(rec, source="src")
        score = monitor.compute_quality_score("src")
        assert score.drift_detected is True

    def test_freshness_tracking_in_quality_score(self):
        monitor = DataQualityMonitor(max_staleness_seconds=60)
        monitor.process_record(_valid_record(), source="src")
        score = monitor.compute_quality_score("src")
        assert score.freshness_seconds is not None
        assert score.freshness_seconds < 2.0
