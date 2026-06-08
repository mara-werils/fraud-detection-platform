"""
Comprehensive Prometheus metrics exporter for the fraud detection platform.

Aggregates and exports metrics across all services: scoring, feature store,
alerting, case management, streaming, and API layer. Provides SLO burn-rate
indicators and custom business metrics (approval / decline / review rates).

Usage:
    from observability.metrics_exporter import MetricsExporter

    exporter = MetricsExporter()
    exporter.observe_scoring_latency(model="xgboost", risk_level="high", duration=0.042)
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

F = TypeVar("F", bound=Callable[..., object])

# Default bucket presets used across multiple histograms
_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
_SCORE_BUCKETS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


class MetricsExporter:
    """Central Prometheus metrics exporter for the fraud-detection platform.

    All metrics are registered on a single ``CollectorRegistry`` so that they
    can be exported atomically via ``generate_latest()``.  The class exposes
    typed helper methods that enforce label consistency and make call-sites
    concise.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self._build_metrics()

    # ------------------------------------------------------------------
    # Metric definitions
    # ------------------------------------------------------------------

    def _build_metrics(self) -> None:  # noqa: C901 (single setup method)
        r = self.registry

        # --- Transaction Scoring Latency ---
        self.scoring_latency = Histogram(
            "transaction_scoring_latency_seconds",
            "End-to-end latency of transaction scoring",
            labelnames=["model", "risk_level"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )

        # --- Fraud Detection Rate (real-time) ---
        self.fraud_detection_rate = Gauge(
            "fraud_detection_rate",
            "Real-time ratio of transactions classified as fraud",
            registry=r,
        )

        # --- Model Prediction Distribution ---
        self.model_prediction_distribution = Histogram(
            "model_prediction_distribution",
            "Distribution of raw prediction scores produced by the model",
            labelnames=["model"],
            buckets=_SCORE_BUCKETS,
            registry=r,
        )

        # --- Feature Extraction Latency ---
        self.feature_extraction_latency = Histogram(
            "feature_extraction_latency_seconds",
            "Latency for extracting features from the feature store",
            labelnames=["feature_group"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )

        # --- Cache Hit / Miss ---
        self.cache_hits_total = Counter(
            "cache_hits_total",
            "Total cache hits",
            labelnames=["cache_name"],
            registry=r,
        )
        self.cache_misses_total = Counter(
            "cache_misses_total",
            "Total cache misses",
            labelnames=["cache_name"],
            registry=r,
        )

        # --- API Request Rate ---
        self.api_request_total = Counter(
            "api_request_total",
            "Total API requests by endpoint and method",
            labelnames=["endpoint", "method", "status_code"],
            registry=r,
        )

        # --- Active Cases ---
        self.active_cases = Gauge(
            "active_cases_total",
            "Number of currently active fraud investigation cases",
            labelnames=["priority"],
            registry=r,
        )

        # --- Alert Firing Rate ---
        self.alert_firing_total = Counter(
            "alert_firing_total",
            "Total number of alerts fired",
            labelnames=["severity", "channel"],
            registry=r,
        )

        # --- Queue Depth ---
        self.queue_depth = Gauge(
            "queue_depth",
            "Current depth of processing queues",
            labelnames=["queue_name"],
            registry=r,
        )

        # --- SLO Burn Rate ---
        self.slo_burn_rate = Gauge(
            "slo_burn_rate",
            "SLO error-budget burn rate (>1 means burning faster than budget allows)",
            labelnames=["slo_name", "window"],
            registry=r,
        )

        # --- Business Metrics ---
        self.approval_total = Counter(
            "business_approval_total",
            "Total approved transactions",
            registry=r,
        )
        self.decline_total = Counter(
            "business_decline_total",
            "Total declined transactions",
            registry=r,
        )
        self.review_total = Counter(
            "business_review_total",
            "Total transactions sent to manual review",
            registry=r,
        )
        self.approval_rate = Gauge(
            "business_approval_rate",
            "Current approval rate (approved / total decisions)",
            registry=r,
        )
        self.decline_rate = Gauge(
            "business_decline_rate",
            "Current decline rate (declined / total decisions)",
            registry=r,
        )
        self.review_rate = Gauge(
            "business_review_rate",
            "Current review rate (review / total decisions)",
            registry=r,
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def observe_scoring_latency(
        self, *, model: str, risk_level: str, duration: float
    ) -> None:
        """Record a transaction scoring latency observation."""
        self.scoring_latency.labels(model=model, risk_level=risk_level).observe(duration)

    def set_fraud_detection_rate(self, rate: float) -> None:
        """Set the real-time fraud detection rate (0..1)."""
        self.fraud_detection_rate.set(rate)

    def observe_prediction(self, *, model: str, score: float) -> None:
        """Record a single model prediction score."""
        self.model_prediction_distribution.labels(model=model).observe(score)

    def observe_feature_extraction_latency(
        self, *, feature_group: str, duration: float
    ) -> None:
        """Record feature extraction latency for a given feature group."""
        self.feature_extraction_latency.labels(feature_group=feature_group).observe(duration)

    def record_cache_hit(self, cache_name: str) -> None:
        self.cache_hits_total.labels(cache_name=cache_name).inc()

    def record_cache_miss(self, cache_name: str) -> None:
        self.cache_misses_total.labels(cache_name=cache_name).inc()

    def record_api_request(
        self, *, endpoint: str, method: str, status_code: int
    ) -> None:
        self.api_request_total.labels(
            endpoint=endpoint, method=method, status_code=str(status_code)
        ).inc()

    def set_active_cases(self, *, priority: str, count: int) -> None:
        self.active_cases.labels(priority=priority).set(count)

    def record_alert_fired(self, *, severity: str, channel: str) -> None:
        self.alert_firing_total.labels(severity=severity, channel=channel).inc()

    def set_queue_depth(self, *, queue_name: str, depth: int) -> None:
        self.queue_depth.labels(queue_name=queue_name).set(depth)

    def set_slo_burn_rate(
        self, *, slo_name: str, window: str, burn_rate: float
    ) -> None:
        """Set SLO burn-rate indicator.

        ``window`` is typically ``"1h"``, ``"6h"``, or ``"3d"``.
        A burn_rate > 1.0 means the error budget is being consumed faster than
        the SLO allows.
        """
        self.slo_burn_rate.labels(slo_name=slo_name, window=window).set(burn_rate)

    def record_decision(self, decision: str) -> None:
        """Record a business decision (``approved``, ``declined``, ``review``).

        After recording, the caller should invoke ``update_business_rates``
        to recalculate the rate gauges.
        """
        if decision == "approved":
            self.approval_total.inc()
        elif decision == "declined":
            self.decline_total.inc()
        elif decision == "review":
            self.review_total.inc()
        else:
            raise ValueError(f"Unknown decision type: {decision!r}")

    def update_business_rates(
        self, *, approved: int, declined: int, review: int
    ) -> None:
        """Recalculate approval / decline / review rate gauges from totals."""
        total = approved + declined + review
        if total == 0:
            self.approval_rate.set(0.0)
            self.decline_rate.set(0.0)
            self.review_rate.set(0.0)
            return
        self.approval_rate.set(approved / total)
        self.decline_rate.set(declined / total)
        self.review_rate.set(review / total)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def generate_latest(self) -> bytes:
        """Return the latest Prometheus exposition-format payload."""
        return generate_latest(self.registry)

    def time_scoring(self, *, model: str, risk_level: str):
        """Context manager that times a scoring operation.

        Usage::

            with exporter.time_scoring(model="xgboost", risk_level="medium"):
                score = model.predict(features)
        """
        return _Timer(
            callback=lambda d: self.observe_scoring_latency(
                model=model, risk_level=risk_level, duration=d
            )
        )

    def time_feature_extraction(self, *, feature_group: str):
        """Context manager that times a feature extraction operation."""
        return _Timer(
            callback=lambda d: self.observe_feature_extraction_latency(
                feature_group=feature_group, duration=d
            )
        )


class _Timer:
    """Simple context-manager timer that invokes a callback with elapsed seconds."""

    def __init__(self, callback: Callable[[float], object]) -> None:
        self._callback = callback
        self._start: float = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_exc: object) -> None:
        elapsed = time.monotonic() - self._start
        self._callback(elapsed)
