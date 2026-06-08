"""Data quality monitoring for incoming transaction data.

Provides schema validation, null/missing field detection, data freshness
monitoring, statistical outlier detection, data drift detection (KL divergence
and PSI), and per-source quality scoring.
"""

from __future__ import annotations

import bisect
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: dict[str, type] = {
    "transaction_id": str,
    "user_id": str,
    "amount": (int, float),  # type: ignore[assignment]
    "currency": str,
    "transaction_type": str,
    "timestamp": str,
}

OPTIONAL_FIELDS: dict[str, type] = {
    "merchant_id": str,
    "merchant_category": str,
    "card_number_hash": str,
    "ip_address": str,
    "device_id": str,
    "latitude": (int, float),  # type: ignore[assignment]
    "longitude": (int, float),  # type: ignore[assignment]
}

ALL_FIELDS: dict[str, type] = {**REQUIRED_FIELDS, **OPTIONAL_FIELDS}

VALID_TRANSACTION_TYPES = {"purchase", "transfer", "withdrawal", "deposit", "refund"}
VALID_CURRENCIES_PATTERN_LEN = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SchemaViolation:
    """A single schema violation."""

    field: str
    violation_type: str  # "missing", "wrong_type", "invalid_value"
    detail: str


@dataclass
class ValidationResult:
    """Result of validating a single transaction."""

    valid: bool
    violations: list[SchemaViolation] = field(default_factory=list)
    null_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": [
                {"field": v.field, "type": v.violation_type, "detail": v.detail}
                for v in self.violations
            ],
            "null_fields": self.null_fields,
        }


@dataclass
class QualityScore:
    """Quality score for a data source."""

    source: str
    score: float  # 0.0 – 1.0
    total_records: int
    valid_records: int
    null_field_rates: dict[str, float]
    freshness_seconds: float | None
    outlier_rate: float
    drift_detected: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "score": round(self.score, 4),
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "null_field_rates": {k: round(v, 4) for k, v in self.null_field_rates.items()},
            "freshness_seconds": round(self.freshness_seconds, 2) if self.freshness_seconds is not None else None,
            "outlier_rate": round(self.outlier_rate, 4),
            "drift_detected": self.drift_detected,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""

    feature: str
    kl_divergence: float
    psi: float
    drift_detected: bool


# ---------------------------------------------------------------------------
# Schema validator
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Validates incoming transaction dictionaries against the expected schema."""

    def validate(self, record: dict[str, Any]) -> ValidationResult:
        violations: list[SchemaViolation] = []
        null_fields: list[str] = []

        # Check required fields
        for fname, ftype in REQUIRED_FIELDS.items():
            if fname not in record or record[fname] is None:
                violations.append(
                    SchemaViolation(field=fname, violation_type="missing", detail=f"Required field '{fname}' is missing or null")
                )
                null_fields.append(fname)
                continue
            if not isinstance(record[fname], ftype):
                violations.append(
                    SchemaViolation(
                        field=fname,
                        violation_type="wrong_type",
                        detail=f"Expected {ftype}, got {type(record[fname]).__name__}",
                    )
                )

        # Check optional fields — only validate type when present and non-null
        for fname, ftype in OPTIONAL_FIELDS.items():
            if fname in record:
                if record[fname] is None:
                    null_fields.append(fname)
                elif not isinstance(record[fname], ftype):
                    violations.append(
                        SchemaViolation(
                            field=fname,
                            violation_type="wrong_type",
                            detail=f"Expected {ftype}, got {type(record[fname]).__name__}",
                        )
                    )

        # Validate specific values
        if "transaction_type" in record and record.get("transaction_type") is not None:
            if record["transaction_type"] not in VALID_TRANSACTION_TYPES:
                violations.append(
                    SchemaViolation(
                        field="transaction_type",
                        violation_type="invalid_value",
                        detail=f"Invalid transaction type: {record['transaction_type']}",
                    )
                )

        if "currency" in record and record.get("currency") is not None:
            cur = record["currency"]
            if not (isinstance(cur, str) and len(cur) == VALID_CURRENCIES_PATTERN_LEN and cur.isalpha() and cur.isupper()):
                violations.append(
                    SchemaViolation(
                        field="currency",
                        violation_type="invalid_value",
                        detail=f"Currency must be a 3-letter uppercase ISO code, got '{cur}'",
                    )
                )

        if "amount" in record and record.get("amount") is not None:
            if isinstance(record["amount"], (int, float)) and record["amount"] < 0:
                violations.append(
                    SchemaViolation(
                        field="amount",
                        violation_type="invalid_value",
                        detail="Amount must be non-negative",
                    )
                )

        return ValidationResult(valid=len(violations) == 0, violations=violations, null_fields=null_fields)


# ---------------------------------------------------------------------------
# Null / missing field monitor
# ---------------------------------------------------------------------------


class NullFieldMonitor:
    """Tracks null/missing field rates and alerts when thresholds are exceeded.

    Args:
        thresholds: Mapping of field name to maximum allowed null rate (0.0–1.0).
        default_threshold: Default null-rate threshold for fields not in *thresholds*.
        window_size: Number of records to track in the sliding window.
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        default_threshold: float = 0.1,
        window_size: int = 1000,
    ) -> None:
        self._thresholds = thresholds or {}
        self._default_threshold = default_threshold
        self._window_size = window_size
        self._lock = Lock()

        # Each field tracks a deque of booleans (True = null/missing)
        self._observations: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=window_size))
        self._total_records = 0

    def observe(self, record: dict[str, Any]) -> None:
        """Record the presence/absence of all known fields."""
        with self._lock:
            self._total_records += 1
            for fname in ALL_FIELDS:
                is_null = fname not in record or record[fname] is None
                self._observations[fname].append(is_null)

    def null_rates(self) -> dict[str, float]:
        """Return the current null rate for every tracked field."""
        with self._lock:
            rates: dict[str, float] = {}
            for fname, obs in self._observations.items():
                if obs:
                    rates[fname] = sum(obs) / len(obs)
                else:
                    rates[fname] = 0.0
            return rates

    def breached_fields(self) -> dict[str, float]:
        """Return fields whose null rates exceed their threshold."""
        rates = self.null_rates()
        breached: dict[str, float] = {}
        for fname, rate in rates.items():
            threshold = self._thresholds.get(fname, self._default_threshold)
            if rate > threshold:
                breached[fname] = rate
        return breached


# ---------------------------------------------------------------------------
# Data freshness monitor
# ---------------------------------------------------------------------------


class FreshnessMonitor:
    """Monitors the freshness of incoming data per source.

    Args:
        max_staleness_seconds: Maximum acceptable age (in seconds) for a record.
    """

    def __init__(self, max_staleness_seconds: float = 300.0) -> None:
        self._max_staleness = max_staleness_seconds
        self._lock = Lock()
        self._last_seen: dict[str, float] = {}  # source -> epoch timestamp
        self._delays: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=500))

    def record_arrival(self, source: str, record_timestamp: float | None = None) -> float:
        """Record the arrival of a new record from *source*.

        Args:
            source: Data source identifier.
            record_timestamp: The record's own timestamp as Unix epoch seconds.
                If ``None``, the current time is used (zero delay).

        Returns:
            The delay in seconds (current time minus record timestamp).
        """
        now = time.time()
        delay = (now - record_timestamp) if record_timestamp is not None else 0.0
        with self._lock:
            self._last_seen[source] = now
            self._delays[source].append(delay)
        return delay

    def is_stale(self, source: str) -> bool:
        """Return True if no data has arrived from *source* within the staleness window."""
        with self._lock:
            last = self._last_seen.get(source)
            if last is None:
                return True
            return (time.time() - last) > self._max_staleness

    def freshness_seconds(self, source: str) -> float | None:
        """Return seconds since the last record from *source*, or None if never seen."""
        with self._lock:
            last = self._last_seen.get(source)
            if last is None:
                return None
            return time.time() - last

    def avg_delay(self, source: str) -> float | None:
        """Average ingestion delay for *source*."""
        with self._lock:
            delays = self._delays.get(source)
            if not delays:
                return None
            return sum(delays) / len(delays)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {}
            now = time.time()
            for source, last in self._last_seen.items():
                delays = self._delays.get(source, deque())
                result[source] = {
                    "last_seen_seconds_ago": round(now - last, 2),
                    "stale": (now - last) > self._max_staleness,
                    "avg_delay_seconds": round(sum(delays) / len(delays), 2) if delays else None,
                }
            return result


# ---------------------------------------------------------------------------
# Statistical outlier detector
# ---------------------------------------------------------------------------


class OutlierDetector:
    """Detects statistical outliers in feature values using z-score.

    Args:
        zscore_threshold: Absolute z-score above which a value is an outlier.
        min_samples: Minimum observations before outlier detection activates.
    """

    def __init__(self, zscore_threshold: float = 3.0, min_samples: int = 30) -> None:
        self._zscore_threshold = zscore_threshold
        self._min_samples = min_samples
        self._lock = Lock()

        # Running statistics per feature (Welford's online algorithm)
        self._count: dict[str, int] = defaultdict(int)
        self._mean: dict[str, float] = defaultdict(float)
        self._m2: dict[str, float] = defaultdict(float)

    def update(self, features: dict[str, float]) -> dict[str, float]:
        """Update running stats and return z-scores for each feature.

        Args:
            features: Mapping of feature name to numeric value.

        Returns:
            Dictionary of feature name to z-score.  Only features with enough
            samples to compute statistics are included.
        """
        zscores: dict[str, float] = {}
        with self._lock:
            for fname, value in features.items():
                # Welford's online update
                self._count[fname] += 1
                n = self._count[fname]
                delta = value - self._mean[fname]
                self._mean[fname] += delta / n
                delta2 = value - self._mean[fname]
                self._m2[fname] += delta * delta2

                if n >= self._min_samples:
                    variance = self._m2[fname] / n
                    std = math.sqrt(variance) if variance > 0 else 0.0
                    if std > 0:
                        zscores[fname] = (value - self._mean[fname]) / std
                    else:
                        zscores[fname] = 0.0
        return zscores

    def is_outlier(self, features: dict[str, float]) -> dict[str, bool]:
        """Return a mapping indicating which features are outliers."""
        zscores = self.update(features)
        return {fname: abs(z) > self._zscore_threshold for fname, z in zscores.items()}

    def stats(self) -> dict[str, dict[str, float]]:
        """Return current running statistics."""
        with self._lock:
            result: dict[str, dict[str, float]] = {}
            for fname in self._count:
                n = self._count[fname]
                variance = self._m2[fname] / n if n > 0 else 0.0
                result[fname] = {
                    "count": n,
                    "mean": self._mean[fname],
                    "std": math.sqrt(variance) if variance > 0 else 0.0,
                }
            return result


# ---------------------------------------------------------------------------
# Drift detector (KL divergence + PSI)
# ---------------------------------------------------------------------------


class DataDriftDetector:
    """Detects data drift using KL divergence and Population Stability Index.

    Maintains a reference (baseline) distribution per feature and compares
    it against a sliding current window.

    Args:
        reference_window: Number of samples for the reference distribution.
        current_window: Number of samples for the current distribution.
        psi_threshold: PSI above which drift is flagged.
        kl_threshold: KL divergence above which drift is flagged.
        num_bins: Number of histogram bins.
        monitored_features: Feature names to monitor.
    """

    DEFAULT_FEATURES = [
        "amount",
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
        kl_threshold: float = 0.5,
        num_bins: int = 10,
        monitored_features: list[str] | None = None,
    ) -> None:
        self._reference_window = reference_window
        self._current_window = current_window
        self._psi_threshold = psi_threshold
        self._kl_threshold = kl_threshold
        self._num_bins = num_bins
        self._features = monitored_features or self.DEFAULT_FEATURES

        self._lock = Lock()
        self._reference: dict[str, deque[float]] = {f: deque(maxlen=reference_window) for f in self._features}
        self._current: dict[str, deque[float]] = {f: deque(maxlen=current_window) for f in self._features}
        self._samples_added = 0
        self._results: deque[list[DriftResult]] = deque(maxlen=50)

    def add_sample(self, features: dict[str, float]) -> None:
        """Add an observation. Fills reference first, then current."""
        with self._lock:
            self._samples_added += 1
            for fname in self._features:
                val = features.get(fname)
                if val is None:
                    continue
                if len(self._reference[fname]) < self._reference_window:
                    self._reference[fname].append(val)
                else:
                    self._current[fname].append(val)

    def check_drift(self) -> list[DriftResult] | None:
        """Run drift detection. Returns results or None if insufficient data."""
        with self._lock:
            min_ref = min((len(self._reference[f]) for f in self._features), default=0)
            min_cur = min((len(self._current[f]) for f in self._features), default=0)

            if min_ref < 50 or min_cur < 30:
                return None

            results: list[DriftResult] = []
            for fname in self._features:
                ref = list(self._reference[fname])
                cur = list(self._current[fname])
                psi = self._compute_psi(ref, cur, self._num_bins)
                kl = self._compute_kl_divergence(ref, cur, self._num_bins)
                drifted = psi > self._psi_threshold or kl > self._kl_threshold
                results.append(DriftResult(feature=fname, kl_divergence=kl, psi=psi, drift_detected=drifted))

                if drifted:
                    logger.warning("data_drift_detected", feature=fname, psi=round(psi, 4), kl=round(kl, 4))

            self._results.append(results)
            return results

    def latest_results(self) -> list[DriftResult] | None:
        return self._results[-1] if self._results else None

    # ---- Static computational helpers ----

    @staticmethod
    def _histogram(data: list[float], edges: list[float]) -> list[int]:
        """Bucket *data* into bins defined by *edges*."""
        n_bins = len(edges) - 1
        counts = [0] * n_bins
        for v in data:
            idx = bisect.bisect_right(edges, v) - 1
            idx = max(0, min(idx, n_bins - 1))
            counts[idx] += 1
        return counts

    @staticmethod
    def _compute_psi(reference: list[float], current: list[float], bins: int = 10) -> float:
        """Compute Population Stability Index."""
        if not reference or not current:
            return 0.0

        all_vals = reference + current
        min_val, max_val = min(all_vals), max(all_vals)
        if min_val == max_val:
            return 0.0

        bin_width = (max_val - min_val) / bins
        edges = [min_val + i * bin_width for i in range(bins + 1)]
        edges[-1] = max_val + 1e-10

        ref_counts = DataDriftDetector._histogram(reference, edges)
        cur_counts = DataDriftDetector._histogram(current, edges)

        ref_total = sum(ref_counts) or 1
        cur_total = sum(cur_counts) or 1

        psi = 0.0
        for i in range(bins):
            ref_prop = max(ref_counts[i] / ref_total, 1e-6)
            cur_prop = max(cur_counts[i] / cur_total, 1e-6)
            psi += (cur_prop - ref_prop) * math.log(cur_prop / ref_prop)
        return psi

    @staticmethod
    def _compute_kl_divergence(reference: list[float], current: list[float], bins: int = 10) -> float:
        """Compute KL divergence D_KL(current || reference).

        Uses histogram binning and smoothing to avoid zero probabilities.
        """
        if not reference or not current:
            return 0.0

        all_vals = reference + current
        min_val, max_val = min(all_vals), max(all_vals)
        if min_val == max_val:
            return 0.0

        bin_width = (max_val - min_val) / bins
        edges = [min_val + i * bin_width for i in range(bins + 1)]
        edges[-1] = max_val + 1e-10

        ref_counts = DataDriftDetector._histogram(reference, edges)
        cur_counts = DataDriftDetector._histogram(current, edges)

        ref_total = sum(ref_counts) or 1
        cur_total = sum(cur_counts) or 1

        kl = 0.0
        for i in range(bins):
            p = max(cur_counts[i] / cur_total, 1e-10)  # current (P)
            q = max(ref_counts[i] / ref_total, 1e-10)  # reference (Q)
            kl += p * math.log(p / q)
        return kl


# ---------------------------------------------------------------------------
# Composite quality scorer
# ---------------------------------------------------------------------------


class DataQualityMonitor:
    """Orchestrates all data quality checks and produces per-source quality scores.

    Args:
        null_thresholds: Per-field null rate thresholds.
        default_null_threshold: Default null rate threshold.
        max_staleness_seconds: Maximum acceptable freshness.
        zscore_threshold: Z-score threshold for outlier detection.
        psi_threshold: PSI threshold for drift detection.
        kl_threshold: KL divergence threshold for drift detection.
        reference_window: Drift detector reference window size.
        current_window: Drift detector current window size.
        monitored_features: Features to monitor for drift/outliers.
    """

    def __init__(
        self,
        null_thresholds: dict[str, float] | None = None,
        default_null_threshold: float = 0.1,
        max_staleness_seconds: float = 300.0,
        zscore_threshold: float = 3.0,
        psi_threshold: float = 0.2,
        kl_threshold: float = 0.5,
        reference_window: int = 10_000,
        current_window: int = 1_000,
        monitored_features: list[str] | None = None,
    ) -> None:
        self.schema_validator = SchemaValidator()
        self.null_monitor = NullFieldMonitor(
            thresholds=null_thresholds,
            default_threshold=default_null_threshold,
        )
        self.freshness_monitor = FreshnessMonitor(max_staleness_seconds=max_staleness_seconds)
        self.outlier_detector = OutlierDetector(zscore_threshold=zscore_threshold)
        self.drift_detector = DataDriftDetector(
            reference_window=reference_window,
            current_window=current_window,
            psi_threshold=psi_threshold,
            kl_threshold=kl_threshold,
            monitored_features=monitored_features,
        )

        self._lock = Lock()
        self._source_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "valid": 0, "outliers": 0})
        self._scores: dict[str, QualityScore] = {}

    def process_record(self, record: dict[str, Any], source: str = "default") -> ValidationResult:
        """Validate a record and update all quality monitors.

        Args:
            record: Raw transaction dictionary.
            source: Data source identifier.

        Returns:
            ValidationResult with schema violations and null fields.
        """
        # Schema validation
        result = self.schema_validator.validate(record)

        # Null monitoring
        self.null_monitor.observe(record)

        # Freshness
        record_ts = None
        if "timestamp" in record and record["timestamp"] is not None:
            try:
                if isinstance(record["timestamp"], (int, float)):
                    record_ts = float(record["timestamp"])
                elif isinstance(record["timestamp"], str):
                    dt = datetime.fromisoformat(record["timestamp"])
                    record_ts = dt.timestamp()
            except (ValueError, TypeError, OSError):
                pass
        self.freshness_monitor.record_arrival(source, record_ts)

        # Extract numeric features for outlier/drift
        numeric_features: dict[str, float] = {}
        for key, val in record.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_features[key] = float(val)

        # Outlier detection
        outlier_flags = self.outlier_detector.is_outlier(numeric_features)
        has_outlier = any(outlier_flags.values())

        # Drift detection
        self.drift_detector.add_sample(numeric_features)

        # Update source stats
        with self._lock:
            self._source_stats[source]["total"] += 1
            if result.valid:
                self._source_stats[source]["valid"] += 1
            if has_outlier:
                self._source_stats[source]["outliers"] += 1

        return result

    def compute_quality_score(self, source: str = "default") -> QualityScore:
        """Compute a composite quality score for the given data source.

        The score is a weighted average:
        - 40% schema validity rate
        - 20% null-field health (1 - fraction of breached fields)
        - 20% freshness health (1 if fresh, 0 if stale)
        - 10% outlier health (1 - outlier rate)
        - 10% drift health (1 if no drift, 0 if drift)
        """
        with self._lock:
            stats = self._source_stats[source]
            total = stats["total"]
            valid = stats["valid"]
            outliers = stats["outliers"]

        if total == 0:
            score = QualityScore(
                source=source,
                score=0.0,
                total_records=0,
                valid_records=0,
                null_field_rates={},
                freshness_seconds=None,
                outlier_rate=0.0,
                drift_detected=False,
            )
            self._scores[source] = score
            return score

        # Validity component
        validity_rate = valid / total

        # Null field component
        null_rates = self.null_monitor.null_rates()
        breached = self.null_monitor.breached_fields()
        tracked_field_count = len(null_rates) or 1
        null_health = 1.0 - (len(breached) / tracked_field_count)

        # Freshness component
        freshness_sec = self.freshness_monitor.freshness_seconds(source)
        stale = self.freshness_monitor.is_stale(source)
        freshness_health = 0.0 if stale else 1.0

        # Outlier component
        outlier_rate = outliers / total
        outlier_health = 1.0 - min(outlier_rate, 1.0)

        # Drift component
        drift_results = self.drift_detector.check_drift()
        drift_detected = False
        if drift_results:
            drift_detected = any(r.drift_detected for r in drift_results)
        drift_health = 0.0 if drift_detected else 1.0

        composite = (
            0.40 * validity_rate
            + 0.20 * null_health
            + 0.20 * freshness_health
            + 0.10 * outlier_health
            + 0.10 * drift_health
        )

        score = QualityScore(
            source=source,
            score=composite,
            total_records=total,
            valid_records=valid,
            null_field_rates=null_rates,
            freshness_seconds=freshness_sec,
            outlier_rate=outlier_rate,
            drift_detected=drift_detected,
        )
        self._scores[source] = score
        return score

    def get_latest_score(self, source: str = "default") -> QualityScore | None:
        """Return the most recently computed quality score for *source*."""
        return self._scores.get(source)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all monitored sources."""
        return {
            "sources": {
                source: score.to_dict() for source, score in self._scores.items()
            },
            "freshness": self.freshness_monitor.summary(),
            "null_breaches": self.null_monitor.breached_fields(),
            "outlier_stats": self.outlier_detector.stats(),
        }
