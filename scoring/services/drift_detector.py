"""Model drift detection using PSI and KS statistics.

Monitors feature distributions and score distributions over time
to detect data drift and concept drift that could degrade model performance.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DriftMetric:
    """A single drift measurement."""

    feature_name: str
    psi: float
    ks_statistic: float
    drift_detected: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class DriftReport:
    """Complete drift report across all monitored features."""

    timestamp: datetime
    overall_drift: bool
    score_psi: float
    score_ks: float
    feature_metrics: list[DriftMetric]
    reference_size: int
    current_size: int
    alert_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_drift": self.overall_drift,
            "score_psi": round(self.score_psi, 6),
            "score_ks": round(self.score_ks, 6),
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "alert_features": self.alert_features,
            "feature_metrics": [
                {
                    "feature": m.feature_name,
                    "psi": round(m.psi, 6),
                    "ks_statistic": round(m.ks_statistic, 6),
                    "drift_detected": m.drift_detected,
                }
                for m in self.feature_metrics
            ],
        }


class DriftDetector:
    """Detects model and data drift using statistical tests.

    Maintains a reference distribution (baseline) and compares incoming
    data against it using Population Stability Index (PSI) and
    Kolmogorov-Smirnov (KS) statistics.

    Args:
        reference_window: Number of samples in the reference window.
        current_window: Number of samples in the current window.
        psi_threshold: PSI value above which drift is flagged.
        ks_threshold: KS statistic above which drift is flagged.
        monitored_features: List of feature names to monitor.
    """

    DEFAULT_FEATURES = [
        "fraud_score",
        "amount_zscore",
        "txn_count_1h",
        "txn_count_24h",
        "distance_from_last_txn_km",
        "amount_to_avg_ratio",
        "unique_merchants_24h",
        "time_since_last_txn_seconds",
    ]

    def __init__(
        self,
        reference_window: int = 10_000,
        current_window: int = 1_000,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1,
        monitored_features: list[str] | None = None,
    ) -> None:
        self._reference_window = reference_window
        self._current_window = current_window
        self._psi_threshold = psi_threshold
        self._ks_threshold = ks_threshold
        self._features = monitored_features or self.DEFAULT_FEATURES

        self._lock = Lock()
        self._reference: dict[str, deque[float]] = {
            f: deque(maxlen=reference_window) for f in self._features
        }
        self._current: dict[str, deque[float]] = {
            f: deque(maxlen=current_window) for f in self._features
        }
        self._reports: deque[DriftReport] = deque(maxlen=100)
        self._samples_added = 0

    def add_sample(self, features: dict[str, float]) -> None:
        """Add a new sample to the current window.

        Once the reference window is full, new samples go to the current
        window for comparison. Before that, samples populate the reference.

        Args:
            features: Dictionary mapping feature name to value.
        """
        with self._lock:
            self._samples_added += 1

            for feat_name in self._features:
                value = features.get(feat_name)
                if value is None:
                    continue

                if len(self._reference[feat_name]) < self._reference_window:
                    self._reference[feat_name].append(value)
                else:
                    self._current[feat_name].append(value)

    def check_drift(self) -> DriftReport | None:
        """Run drift detection on all monitored features.

        Returns:
            DriftReport if enough data is available, None otherwise.
        """
        with self._lock:
            # Need enough data in both windows
            min_ref = min(len(self._reference[f]) for f in self._features)
            min_cur = min(len(self._current[f]) for f in self._features)

            if min_ref < 100 or min_cur < 50:
                return None

            feature_metrics: list[DriftMetric] = []
            alert_features: list[str] = []

            for feat_name in self._features:
                ref_data = list(self._reference[feat_name])
                cur_data = list(self._current[feat_name])

                psi = self._compute_psi(ref_data, cur_data)
                ks = self._compute_ks(ref_data, cur_data)
                drifted = psi > self._psi_threshold or ks > self._ks_threshold

                metric = DriftMetric(
                    feature_name=feat_name,
                    psi=psi,
                    ks_statistic=ks,
                    drift_detected=drifted,
                )
                feature_metrics.append(metric)

                if drifted:
                    alert_features.append(feat_name)

            # Score-level drift
            score_ref = list(self._reference.get("fraud_score", []))
            score_cur = list(self._current.get("fraud_score", []))
            score_psi = self._compute_psi(score_ref, score_cur) if score_ref and score_cur else 0.0
            score_ks = self._compute_ks(score_ref, score_cur) if score_ref and score_cur else 0.0

            overall_drift = len(alert_features) > 0

            report = DriftReport(
                timestamp=datetime.now(UTC),
                overall_drift=overall_drift,
                score_psi=score_psi,
                score_ks=score_ks,
                feature_metrics=feature_metrics,
                reference_size=min_ref,
                current_size=min_cur,
                alert_features=alert_features,
            )

            self._reports.append(report)

            if overall_drift:
                logger.warning(
                    "model_drift_detected",
                    alert_features=alert_features,
                    score_psi=round(score_psi, 4),
                )

            return report

    def get_latest_report(self) -> DriftReport | None:
        """Return the most recent drift report."""
        return self._reports[-1] if self._reports else None

    def get_reports(self, limit: int = 10) -> list[DriftReport]:
        """Return recent drift reports."""
        return list(self._reports)[-limit:]

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary of drift detection state."""
        latest = self.get_latest_report()
        return {
            "samples_added": self._samples_added,
            "reports_generated": len(self._reports),
            "latest_drift_detected": latest.overall_drift if latest else None,
            "latest_alert_features": latest.alert_features if latest else [],
            "latest_score_psi": round(latest.score_psi, 6) if latest else None,
            "monitored_features": self._features,
            "thresholds": {
                "psi": self._psi_threshold,
                "ks": self._ks_threshold,
            },
        }

    @staticmethod
    def _compute_psi(reference: list[float], current: list[float], bins: int = 10) -> float:
        """Compute Population Stability Index (PSI).

        PSI measures how much the distribution has shifted.
        PSI < 0.1: no significant shift
        PSI 0.1-0.2: moderate shift
        PSI > 0.2: significant shift

        Args:
            reference: Reference distribution values.
            current: Current distribution values.
            bins: Number of bins for histogram.

        Returns:
            PSI value.
        """
        if not reference or not current:
            return 0.0

        # Compute bin edges from reference
        all_values = reference + current
        min_val = min(all_values)
        max_val = max(all_values)

        if min_val == max_val:
            return 0.0

        bin_width = (max_val - min_val) / bins
        edges = [min_val + i * bin_width for i in range(bins + 1)]
        edges[-1] = max_val + 1e-10  # Include max value

        # Compute proportions using bisect for O(n log bins) instead of O(n * bins)
        ref_counts = [0] * bins
        cur_counts = [0] * bins

        for v in reference:
            idx = bisect.bisect_right(edges, v) - 1
            idx = max(0, min(idx, bins - 1))
            ref_counts[idx] += 1

        for v in current:
            idx = bisect.bisect_right(edges, v) - 1
            idx = max(0, min(idx, bins - 1))
            cur_counts[idx] += 1

        ref_total = sum(ref_counts) or 1
        cur_total = sum(cur_counts) or 1

        psi = 0.0
        for i in range(bins):
            ref_prop = max(ref_counts[i] / ref_total, 1e-6)
            cur_prop = max(cur_counts[i] / cur_total, 1e-6)
            psi += (cur_prop - ref_prop) * math.log(cur_prop / ref_prop)

        return psi

    @staticmethod
    def _compute_ks(reference: list[float], current: list[float]) -> float:
        """Compute Kolmogorov-Smirnov statistic.

        Measures the maximum distance between two CDFs.

        Args:
            reference: Reference distribution values.
            current: Current distribution values.

        Returns:
            KS statistic in [0, 1].
        """
        if not reference or not current:
            return 0.0

        ref_sorted = sorted(reference)
        cur_sorted = sorted(current)

        n_ref = len(ref_sorted)
        n_cur = len(cur_sorted)

        # Merge and compare CDFs
        all_values = sorted(set(ref_sorted + cur_sorted))

        max_diff = 0.0
        ref_idx = 0
        cur_idx = 0

        for val in all_values:
            while ref_idx < n_ref and ref_sorted[ref_idx] <= val:
                ref_idx += 1
            while cur_idx < n_cur and cur_sorted[cur_idx] <= val:
                cur_idx += 1

            ref_cdf = ref_idx / n_ref
            cur_cdf = cur_idx / n_cur
            max_diff = max(max_diff, abs(ref_cdf - cur_cdf))

        return max_diff
