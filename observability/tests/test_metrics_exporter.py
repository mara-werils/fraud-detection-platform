"""Tests for observability.metrics_exporter."""

from __future__ import annotations

import time

import pytest
from prometheus_client import CollectorRegistry

from observability.metrics_exporter import MetricsExporter


@pytest.fixture()
def exporter() -> MetricsExporter:
    """Return a fresh MetricsExporter with an isolated registry."""
    return MetricsExporter(registry=CollectorRegistry())


# ------------------------------------------------------------------
# Transaction Scoring Latency
# ------------------------------------------------------------------


class TestScoringLatency:
    def test_observe_scoring_latency(self, exporter: MetricsExporter) -> None:
        exporter.observe_scoring_latency(model="xgboost", risk_level="high", duration=0.05)
        exporter.observe_scoring_latency(model="xgboost", risk_level="low", duration=0.01)

        # Verify the histogram was populated (count should be 1 per label combo)
        sample_value = exporter.registry.get_sample_value(
            "transaction_scoring_latency_seconds_count",
            {"model": "xgboost", "risk_level": "high"},
        )
        assert sample_value == 1.0

    def test_observe_scoring_latency_multiple(self, exporter: MetricsExporter) -> None:
        for _ in range(5):
            exporter.observe_scoring_latency(model="gnn", risk_level="medium", duration=0.03)

        sample_value = exporter.registry.get_sample_value(
            "transaction_scoring_latency_seconds_count",
            {"model": "gnn", "risk_level": "medium"},
        )
        assert sample_value == 5.0

    def test_time_scoring_context_manager(self, exporter: MetricsExporter) -> None:
        with exporter.time_scoring(model="ensemble", risk_level="high"):
            time.sleep(0.01)

        count = exporter.registry.get_sample_value(
            "transaction_scoring_latency_seconds_count",
            {"model": "ensemble", "risk_level": "high"},
        )
        assert count == 1.0
        total = exporter.registry.get_sample_value(
            "transaction_scoring_latency_seconds_sum",
            {"model": "ensemble", "risk_level": "high"},
        )
        assert total is not None and total >= 0.01


# ------------------------------------------------------------------
# Fraud Detection Rate
# ------------------------------------------------------------------


class TestFraudDetectionRate:
    def test_set_fraud_detection_rate(self, exporter: MetricsExporter) -> None:
        exporter.set_fraud_detection_rate(0.032)
        value = exporter.registry.get_sample_value("fraud_detection_rate")
        assert value == pytest.approx(0.032)

    def test_fraud_detection_rate_overwrite(self, exporter: MetricsExporter) -> None:
        exporter.set_fraud_detection_rate(0.1)
        exporter.set_fraud_detection_rate(0.05)
        value = exporter.registry.get_sample_value("fraud_detection_rate")
        assert value == pytest.approx(0.05)


# ------------------------------------------------------------------
# Model Prediction Distribution
# ------------------------------------------------------------------


class TestModelPredictionDistribution:
    def test_observe_prediction(self, exporter: MetricsExporter) -> None:
        exporter.observe_prediction(model="xgboost", score=0.85)
        count = exporter.registry.get_sample_value(
            "model_prediction_distribution_count", {"model": "xgboost"}
        )
        assert count == 1.0

    def test_observe_multiple_predictions(self, exporter: MetricsExporter) -> None:
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        for s in scores:
            exporter.observe_prediction(model="gnn", score=s)
        count = exporter.registry.get_sample_value(
            "model_prediction_distribution_count", {"model": "gnn"}
        )
        assert count == 5.0
        total = exporter.registry.get_sample_value(
            "model_prediction_distribution_sum", {"model": "gnn"}
        )
        assert total == pytest.approx(sum(scores))


# ------------------------------------------------------------------
# Feature Extraction Latency
# ------------------------------------------------------------------


class TestFeatureExtractionLatency:
    def test_observe_feature_extraction_latency(self, exporter: MetricsExporter) -> None:
        exporter.observe_feature_extraction_latency(feature_group="velocity", duration=0.012)
        count = exporter.registry.get_sample_value(
            "feature_extraction_latency_seconds_count", {"feature_group": "velocity"}
        )
        assert count == 1.0

    def test_time_feature_extraction_context_manager(self, exporter: MetricsExporter) -> None:
        with exporter.time_feature_extraction(feature_group="aggregates"):
            time.sleep(0.01)

        count = exporter.registry.get_sample_value(
            "feature_extraction_latency_seconds_count", {"feature_group": "aggregates"}
        )
        assert count == 1.0


# ------------------------------------------------------------------
# Cache Hit / Miss
# ------------------------------------------------------------------


class TestCacheMetrics:
    def test_cache_hit(self, exporter: MetricsExporter) -> None:
        exporter.record_cache_hit("features")
        exporter.record_cache_hit("features")
        value = exporter.registry.get_sample_value(
            "cache_hits_total", {"cache_name": "features"}
        )
        assert value == 2.0

    def test_cache_miss(self, exporter: MetricsExporter) -> None:
        exporter.record_cache_miss("features")
        value = exporter.registry.get_sample_value(
            "cache_misses_total", {"cache_name": "features"}
        )
        assert value == 1.0

    def test_cache_hit_miss_ratio(self, exporter: MetricsExporter) -> None:
        for _ in range(8):
            exporter.record_cache_hit("model")
        for _ in range(2):
            exporter.record_cache_miss("model")

        hits = exporter.registry.get_sample_value("cache_hits_total", {"cache_name": "model"})
        misses = exporter.registry.get_sample_value("cache_misses_total", {"cache_name": "model"})
        assert hits == 8.0
        assert misses == 2.0
        ratio = hits / (hits + misses)
        assert ratio == pytest.approx(0.8)


# ------------------------------------------------------------------
# API Request Rate
# ------------------------------------------------------------------


class TestApiRequestRate:
    def test_record_api_request(self, exporter: MetricsExporter) -> None:
        exporter.record_api_request(endpoint="/score", method="POST", status_code=200)
        value = exporter.registry.get_sample_value(
            "api_request_total",
            {"endpoint": "/score", "method": "POST", "status_code": "200"},
        )
        assert value == 1.0

    def test_record_api_request_multiple_endpoints(self, exporter: MetricsExporter) -> None:
        exporter.record_api_request(endpoint="/score", method="POST", status_code=200)
        exporter.record_api_request(endpoint="/health", method="GET", status_code=200)
        exporter.record_api_request(endpoint="/score", method="POST", status_code=500)

        score_ok = exporter.registry.get_sample_value(
            "api_request_total",
            {"endpoint": "/score", "method": "POST", "status_code": "200"},
        )
        health_ok = exporter.registry.get_sample_value(
            "api_request_total",
            {"endpoint": "/health", "method": "GET", "status_code": "200"},
        )
        score_err = exporter.registry.get_sample_value(
            "api_request_total",
            {"endpoint": "/score", "method": "POST", "status_code": "500"},
        )
        assert score_ok == 1.0
        assert health_ok == 1.0
        assert score_err == 1.0


# ------------------------------------------------------------------
# Active Cases
# ------------------------------------------------------------------


class TestActiveCases:
    def test_set_active_cases(self, exporter: MetricsExporter) -> None:
        exporter.set_active_cases(priority="high", count=12)
        value = exporter.registry.get_sample_value(
            "active_cases_total", {"priority": "high"}
        )
        assert value == 12.0

    def test_active_cases_update(self, exporter: MetricsExporter) -> None:
        exporter.set_active_cases(priority="low", count=50)
        exporter.set_active_cases(priority="low", count=45)
        value = exporter.registry.get_sample_value(
            "active_cases_total", {"priority": "low"}
        )
        assert value == 45.0


# ------------------------------------------------------------------
# Alert Firing Rate
# ------------------------------------------------------------------


class TestAlertFiring:
    def test_record_alert_fired(self, exporter: MetricsExporter) -> None:
        exporter.record_alert_fired(severity="critical", channel="email")
        value = exporter.registry.get_sample_value(
            "alert_firing_total", {"severity": "critical", "channel": "email"}
        )
        assert value == 1.0

    def test_multiple_alerts_different_channels(self, exporter: MetricsExporter) -> None:
        exporter.record_alert_fired(severity="warning", channel="telegram")
        exporter.record_alert_fired(severity="warning", channel="webhook")
        exporter.record_alert_fired(severity="warning", channel="telegram")

        telegram = exporter.registry.get_sample_value(
            "alert_firing_total", {"severity": "warning", "channel": "telegram"}
        )
        webhook = exporter.registry.get_sample_value(
            "alert_firing_total", {"severity": "warning", "channel": "webhook"}
        )
        assert telegram == 2.0
        assert webhook == 1.0


# ------------------------------------------------------------------
# Queue Depth
# ------------------------------------------------------------------


class TestQueueDepth:
    def test_set_queue_depth(self, exporter: MetricsExporter) -> None:
        exporter.set_queue_depth(queue_name="scoring", depth=150)
        value = exporter.registry.get_sample_value("queue_depth", {"queue_name": "scoring"})
        assert value == 150.0

    def test_queue_depth_updates(self, exporter: MetricsExporter) -> None:
        exporter.set_queue_depth(queue_name="alerts", depth=100)
        exporter.set_queue_depth(queue_name="alerts", depth=80)
        value = exporter.registry.get_sample_value("queue_depth", {"queue_name": "alerts"})
        assert value == 80.0


# ------------------------------------------------------------------
# SLO Burn Rate
# ------------------------------------------------------------------


class TestSloBurnRate:
    def test_set_slo_burn_rate(self, exporter: MetricsExporter) -> None:
        exporter.set_slo_burn_rate(slo_name="scoring_latency_p99", window="1h", burn_rate=0.8)
        value = exporter.registry.get_sample_value(
            "slo_burn_rate", {"slo_name": "scoring_latency_p99", "window": "1h"}
        )
        assert value == pytest.approx(0.8)

    def test_multiple_windows(self, exporter: MetricsExporter) -> None:
        exporter.set_slo_burn_rate(slo_name="availability", window="1h", burn_rate=1.2)
        exporter.set_slo_burn_rate(slo_name="availability", window="6h", burn_rate=0.9)

        val_1h = exporter.registry.get_sample_value(
            "slo_burn_rate", {"slo_name": "availability", "window": "1h"}
        )
        val_6h = exporter.registry.get_sample_value(
            "slo_burn_rate", {"slo_name": "availability", "window": "6h"}
        )
        assert val_1h == pytest.approx(1.2)
        assert val_6h == pytest.approx(0.9)


# ------------------------------------------------------------------
# Business Metrics
# ------------------------------------------------------------------


class TestBusinessMetrics:
    def test_record_decision_approved(self, exporter: MetricsExporter) -> None:
        exporter.record_decision("approved")
        value = exporter.registry.get_sample_value("business_approval_total")
        assert value == 1.0

    def test_record_decision_declined(self, exporter: MetricsExporter) -> None:
        exporter.record_decision("declined")
        value = exporter.registry.get_sample_value("business_decline_total")
        assert value == 1.0

    def test_record_decision_review(self, exporter: MetricsExporter) -> None:
        exporter.record_decision("review")
        value = exporter.registry.get_sample_value("business_review_total")
        assert value == 1.0

    def test_record_decision_invalid_raises(self, exporter: MetricsExporter) -> None:
        with pytest.raises(ValueError, match="Unknown decision type"):
            exporter.record_decision("escalate")

    def test_update_business_rates(self, exporter: MetricsExporter) -> None:
        exporter.update_business_rates(approved=70, declined=20, review=10)
        assert exporter.registry.get_sample_value("business_approval_rate") == pytest.approx(0.7)
        assert exporter.registry.get_sample_value("business_decline_rate") == pytest.approx(0.2)
        assert exporter.registry.get_sample_value("business_review_rate") == pytest.approx(0.1)

    def test_update_business_rates_zero_total(self, exporter: MetricsExporter) -> None:
        exporter.update_business_rates(approved=0, declined=0, review=0)
        assert exporter.registry.get_sample_value("business_approval_rate") == 0.0
        assert exporter.registry.get_sample_value("business_decline_rate") == 0.0
        assert exporter.registry.get_sample_value("business_review_rate") == 0.0


# ------------------------------------------------------------------
# Export / Serialization
# ------------------------------------------------------------------


class TestGenerateLatest:
    def test_generate_latest_returns_bytes(self, exporter: MetricsExporter) -> None:
        output = exporter.generate_latest()
        assert isinstance(output, bytes)

    def test_generate_latest_contains_metric_names(self, exporter: MetricsExporter) -> None:
        exporter.observe_scoring_latency(model="xgboost", risk_level="high", duration=0.05)
        exporter.set_fraud_detection_rate(0.02)
        output = exporter.generate_latest().decode()
        assert "transaction_scoring_latency_seconds" in output
        assert "fraud_detection_rate" in output


# ------------------------------------------------------------------
# Isolated Registry
# ------------------------------------------------------------------


class TestRegistryIsolation:
    def test_separate_registries_are_independent(self) -> None:
        a = MetricsExporter(registry=CollectorRegistry())
        b = MetricsExporter(registry=CollectorRegistry())

        a.record_cache_hit("features")
        val_a = a.registry.get_sample_value("cache_hits_total", {"cache_name": "features"})
        val_b = b.registry.get_sample_value("cache_hits_total", {"cache_name": "features"})

        assert val_a == 1.0
        assert val_b is None
