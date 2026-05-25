"""Analytics engine for real-time fraud detection metrics and reporting.

Provides time-series aggregation, dashboard metrics, geographic heatmaps,
model performance statistics, and period-over-period comparisons — all
computed in-memory from the scored transaction stream.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from threading import Lock
from typing import Any

import structlog
from prometheus_client import Counter as PCounter
from prometheus_client import Gauge, Histogram
from pydantic import BaseModel, Field

from shared.schemas import ScoredTransaction

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

ANALYTICS_TRANSACTIONS_TOTAL = PCounter(
    "analytics_transactions_recorded_total",
    "Total transactions ingested by the analytics engine",
    ["decision"],
)

ANALYTICS_FRAUD_RATE = Gauge(
    "analytics_fraud_rate_24h",
    "Rolling 24-hour fraud rate (blocked transactions / total)",
)

ANALYTICS_AVG_LATENCY_MS = Gauge(
    "analytics_avg_scoring_latency_ms",
    "Rolling average scoring latency in milliseconds",
)

ANALYTICS_THROUGHPUT = Gauge(
    "analytics_throughput_tps",
    "Estimated transactions per second (rolling 60-second window)",
)

ANALYTICS_SCORE_HISTOGRAM = Histogram(
    "analytics_fraud_score",
    "Distribution of fraud scores across all scored transactions",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ── Enums ─────────────────────────────────────────────────────────────────────


class TimeGranularity(StrEnum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class MetricType(StrEnum):
    FRAUD_RATE = "fraud_rate"
    BLOCK_RATE = "block_rate"
    REVIEW_RATE = "review_rate"
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    THROUGHPUT = "throughput"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    MODEL_ACCURACY = "model_accuracy"


# ── Pydantic models ───────────────────────────────────────────────────────────


class TimeSeriesPoint(BaseModel):
    """A single data point in a time series."""

    timestamp: datetime = Field(..., description="Bucket start timestamp (UTC)")
    value: float = Field(..., description="Metric value for this bucket")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional per-point metadata (e.g. sample count)",
    )


class TimeSeriesData(BaseModel):
    """A complete time series response for a given metric."""

    metric_type: MetricType
    granularity: TimeGranularity
    points: list[TimeSeriesPoint] = Field(..., description="Ordered data points")
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate summary (min, max, mean, trend_pct)",
    )


class DashboardMetrics(BaseModel):
    """Real-time platform overview for the operations dashboard."""

    # Volume counters
    total_transactions_today: int = Field(..., description="Transactions scored since midnight UTC")
    total_blocked: int = Field(..., description="Transactions blocked today")
    total_reviewed: int = Field(..., description="Transactions sent to review today")

    # Rate / quality
    fraud_rate_24h: float = Field(..., ge=0.0, le=1.0, description="24-hour fraud rate")
    avg_score: float = Field(..., ge=0.0, le=1.0, description="Average fraud score (24h)")
    avg_latency_ms: float = Field(..., ge=0.0, description="Average scoring latency (ms, 24h)")

    # Categorical breakdowns
    top_fraud_categories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top merchant categories by fraud rate",
    )
    top_fraud_countries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top source countries by transaction count flagged",
    )

    # Score distribution histogram
    score_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Score histogram: bucket label -> count",
    )

    # Intra-day trend
    hourly_trend: list[TimeSeriesPoint] = Field(
        default_factory=list,
        description="Hourly fraud-rate trend for the last 24 hours",
    )

    # Computed at
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When these metrics were computed",
    )


class FraudPattern(BaseModel):
    """A recurring fraud signal or pattern identified in recent data."""

    pattern_id: str
    description: str
    occurrence_count: int
    fraud_rate: float = Field(ge=0.0, le=1.0)
    avg_amount: float
    example_categories: list[str] = Field(default_factory=list)
    first_seen: datetime
    last_seen: datetime


class ModelPerformanceMetrics(BaseModel):
    """Offline-style model quality metrics derived from analyst feedback."""

    model_version: str
    total_evaluated: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    auc_roc_estimate: float = Field(ge=0.0, le=1.0)
    avg_score_positive: float
    avg_score_negative: float
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GeoPoint(BaseModel):
    """Aggregated geographic data point for the heatmap."""

    country_code: str = Field(..., min_length=2, max_length=2)
    country_name: str
    transaction_count: int
    flagged_count: int
    fraud_rate: float = Field(ge=0.0, le=1.0)
    avg_amount: float
    latitude: float
    longitude: float


class PeriodComparison(BaseModel):
    """Statistical comparison between two time periods for a given metric."""

    metric_type: MetricType
    period_1_label: str
    period_2_label: str
    period_1_value: float
    period_2_value: float
    absolute_change: float
    relative_change_pct: float
    is_improvement: bool
    period_1_points: list[TimeSeriesPoint]
    period_2_points: list[TimeSeriesPoint]


# ── Internal dataclass ────────────────────────────────────────────────────────


@dataclass
class _TxnRecord:
    """Lightweight record stored inside the analytics engine."""

    timestamp: datetime
    scored_at: datetime
    fraud_score: float
    latency_ms: float
    is_flagged: bool
    decision: str  # BLOCK / REVIEW / ALLOW
    merchant_category: str | None
    country_code: str | None
    amount: float
    model_version: str
    # Feedback fields — populated later when analysts confirm/deny
    confirmed_fraud: bool | None = None  # None = not yet evaluated


# ── Analytics engine ──────────────────────────────────────────────────────────

_SCORE_BUCKETS = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
                  "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]

_COUNTRY_COORDS: dict[str, tuple[str, float, float]] = {
    "US": ("United States", 37.09, -95.71),
    "GB": ("United Kingdom", 55.38, -3.44),
    "DE": ("Germany", 51.17, 10.45),
    "FR": ("France", 46.23, 2.21),
    "CN": ("China", 35.86, 104.20),
    "RU": ("Russia", 61.52, 105.32),
    "BR": ("Brazil", -14.24, -51.93),
    "IN": ("India", 20.59, 78.96),
    "NG": ("Nigeria", 9.08, 8.68),
    "PH": ("Philippines", 12.88, 121.77),
    "UA": ("Ukraine", 48.38, 31.17),
    "RO": ("Romania", 45.94, 24.97),
    "VN": ("Vietnam", 14.06, 108.28),
    "ID": ("Indonesia", -0.79, 113.92),
    "MX": ("Mexico", 23.63, -102.55),
}

_MAX_RECORDS = 500_000
_WINDOW_24H = timedelta(hours=24)
_WINDOW_1H = timedelta(hours=1)


class AnalyticsEngine:
    """Thread-safe in-memory analytics engine.

    Ingests scored transactions and computes metrics on demand.
    Designed to be registered on ``app.state`` during lifespan startup.
    """

    def __init__(self, max_records: int = _MAX_RECORDS) -> None:
        self._max_records = max_records
        self._lock = Lock()
        self._records: list[_TxnRecord] = []
        # Feedback index: transaction_id -> record index (not stored here,
        # we use a separate dict for fast lookup)
        self._feedback: dict[str, bool] = {}  # txn_id -> confirmed_fraud

        logger.info("analytics_engine_initialized", max_records=max_records)

    # ── Ingest ────────────────────────────────────────────────────────────────

    def record_transaction(self, txn: ScoredTransaction) -> None:
        """Ingest a newly scored transaction into the analytics engine."""
        score = txn.fraud_score
        latency = txn.scoring_latency_ms

        # Derive a simple decision label from is_flagged + score thresholds
        if txn.is_flagged and score >= 0.8:
            decision = "BLOCK"
        elif txn.is_flagged:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        country: str | None = None
        if txn.features and txn.features.unique_countries_24h:
            # Best-effort: extract country from metadata if present
            country = str(txn.metadata.get("country_code", "")) or None

        record = _TxnRecord(
            timestamp=txn.timestamp,
            scored_at=txn.scored_at,
            fraud_score=score,
            latency_ms=latency,
            is_flagged=txn.is_flagged,
            decision=decision,
            merchant_category=txn.metadata.get("merchant_category"),
            country_code=country,
            amount=float(txn.amount),
            model_version=txn.model_version,
        )

        with self._lock:
            if len(self._records) >= self._max_records:
                # Evict oldest 10 %
                trim = self._max_records // 10
                self._records = self._records[trim:]

            self._records.append(record)

        # Prometheus
        ANALYTICS_TRANSACTIONS_TOTAL.labels(decision=decision).inc()
        ANALYTICS_SCORE_HISTOGRAM.observe(score)

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def get_dashboard_metrics(self) -> DashboardMetrics:
        """Return a snapshot of platform-wide real-time metrics."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_24h = now - _WINDOW_24H

        with self._lock:
            records_copy = list(self._records)

        today = [r for r in records_copy if _aware(r.scored_at) >= today_start]
        last_24h = [r for r in records_copy if _aware(r.scored_at) >= cutoff_24h]

        total_today = len(today)
        total_blocked = sum(1 for r in today if r.decision == "BLOCK")
        total_reviewed = sum(1 for r in today if r.decision == "REVIEW")

        fraud_rate_24h = (
            sum(1 for r in last_24h if r.is_flagged) / len(last_24h)
            if last_24h else 0.0
        )
        avg_score = statistics.mean(r.fraud_score for r in last_24h) if last_24h else 0.0
        avg_latency = statistics.mean(r.latency_ms for r in last_24h) if last_24h else 0.0

        # Update Prometheus gauges
        ANALYTICS_FRAUD_RATE.set(fraud_rate_24h)
        ANALYTICS_AVG_LATENCY_MS.set(avg_latency)

        # Top fraud categories
        cat_total: Counter[str] = Counter()
        cat_fraud: Counter[str] = Counter()
        for r in last_24h:
            cat = r.merchant_category or "unknown"
            cat_total[cat] += 1
            if r.is_flagged:
                cat_fraud[cat] += 1

        top_categories = [
            {"category": cat, "count": cat_total[cat],
             "fraud_rate": round(cat_fraud[cat] / cat_total[cat], 4)}
            for cat in sorted(cat_total, key=lambda c: cat_fraud[c], reverse=True)[:10]
        ]

        # Top fraud countries
        country_total: Counter[str] = Counter()
        country_fraud: Counter[str] = Counter()
        for r in last_24h:
            cc = r.country_code or "XX"
            country_total[cc] += 1
            if r.is_flagged:
                country_fraud[cc] += 1

        top_countries = [
            {"country_code": cc, "count": country_total[cc],
             "flagged": country_fraud[cc],
             "fraud_rate": round(country_fraud[cc] / country_total[cc], 4)}
            for cc in sorted(country_total, key=lambda c: country_fraud[c], reverse=True)[:10]
        ]

        # Score distribution
        score_dist: dict[str, int] = {b: 0 for b in _SCORE_BUCKETS}
        for r in last_24h:
            bucket_idx = min(int(r.fraud_score * 10), 9)
            score_dist[_SCORE_BUCKETS[bucket_idx]] += 1

        # Hourly trend (last 24 h)
        hourly_points = self._build_hourly_trend(last_24h, now)

        return DashboardMetrics(
            total_transactions_today=total_today,
            total_blocked=total_blocked,
            total_reviewed=total_reviewed,
            fraud_rate_24h=round(fraud_rate_24h, 6),
            avg_score=round(avg_score, 6),
            avg_latency_ms=round(avg_latency, 3),
            top_fraud_categories=top_categories,
            top_fraud_countries=top_countries,
            score_distribution=score_dist,
            hourly_trend=hourly_points,
        )

    # ── Time series ───────────────────────────────────────────────────────────

    def get_time_series(
        self,
        metric_type: MetricType,
        granularity: TimeGranularity,
        from_ts: datetime,
        to_ts: datetime,
    ) -> TimeSeriesData:
        """Aggregate a metric into time-series buckets over a date range."""
        with self._lock:
            records_copy = list(self._records)

        window = [
            r for r in records_copy
            if from_ts <= _aware(r.scored_at) <= to_ts
        ]

        delta = _granularity_delta(granularity)
        buckets = _build_buckets(from_ts, to_ts, delta)

        points: list[TimeSeriesPoint] = []
        for bucket_start, bucket_end in buckets:
            bucket_recs = [r for r in window if bucket_start <= _aware(r.scored_at) < bucket_end]
            value = _compute_metric(metric_type, bucket_recs)
            points.append(TimeSeriesPoint(
                timestamp=bucket_start,
                value=round(value, 6),
                metadata={"sample_count": len(bucket_recs)},
            ))

        all_values = [p.value for p in points if p.metadata.get("sample_count", 0) > 0]
        summary: dict[str, Any] = {}
        if all_values:
            summary = {
                "min": round(min(all_values), 6),
                "max": round(max(all_values), 6),
                "mean": round(statistics.mean(all_values), 6),
                "trend_pct": _trend_pct(all_values),
            }

        return TimeSeriesData(
            metric_type=metric_type,
            granularity=granularity,
            points=points,
            summary=summary,
        )

    # ── Score distribution ────────────────────────────────────────────────────

    def get_score_distribution(
        self,
        from_ts: datetime,
        to_ts: datetime,
    ) -> dict[str, Any]:
        """Return a detailed score histogram for a given time window."""
        with self._lock:
            records_copy = list(self._records)

        window = [
            r for r in records_copy
            if from_ts <= _aware(r.scored_at) <= to_ts
        ]

        distribution: dict[str, int] = {b: 0 for b in _SCORE_BUCKETS}
        for r in window:
            bucket_idx = min(int(r.fraud_score * 10), 9)
            distribution[_SCORE_BUCKETS[bucket_idx]] += 1

        scores = [r.fraud_score for r in window]
        percentiles: dict[str, float] = {}
        if scores:
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            percentiles = {
                "p50": sorted_scores[int(n * 0.50)],
                "p75": sorted_scores[int(n * 0.75)],
                "p90": sorted_scores[int(n * 0.90)],
                "p95": sorted_scores[int(n * 0.95)],
                "p99": sorted_scores[min(int(n * 0.99), n - 1)],
            }

        return {
            "distribution": distribution,
            "percentiles": percentiles,
            "total_samples": len(window),
            "mean": round(statistics.mean(scores), 6) if scores else 0.0,
            "stddev": round(statistics.stdev(scores), 6) if len(scores) > 1 else 0.0,
        }

    # ── Fraud patterns ────────────────────────────────────────────────────────

    def get_top_fraud_patterns(self, limit: int = 10) -> list[FraudPattern]:
        """Identify recurring fraud signals from recent transactions."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=72)

        with self._lock:
            records_copy = list(self._records)

        recent = [r for r in records_copy if _aware(r.scored_at) >= cutoff]

        # Group flagged transactions by merchant category to surface patterns
        pattern_map: dict[str, list[_TxnRecord]] = defaultdict(list)
        for r in recent:
            if r.is_flagged:
                key = r.merchant_category or "unknown"
                pattern_map[key].append(r)

        patterns: list[FraudPattern] = []
        for cat, recs in sorted(pattern_map.items(), key=lambda kv: len(kv[1]), reverse=True)[:limit]:
            total_in_cat = sum(1 for r in recent if (r.merchant_category or "unknown") == cat)
            fraud_rate = len(recs) / total_in_cat if total_in_cat else 0.0
            timestamps = [_aware(r.scored_at) for r in recs]

            patterns.append(FraudPattern(
                pattern_id=f"pattern_cat_{cat.lower().replace(' ', '_')}",
                description=f"Elevated fraud in category: {cat}",
                occurrence_count=len(recs),
                fraud_rate=round(fraud_rate, 4),
                avg_amount=round(statistics.mean(r.amount for r in recs), 2),
                example_categories=[cat],
                first_seen=min(timestamps),
                last_seen=max(timestamps),
            ))

        return patterns

    # ── Model performance ─────────────────────────────────────────────────────

    def get_model_performance_metrics(self) -> ModelPerformanceMetrics:
        """Compute precision/recall from analyst-confirmed feedback."""
        with self._lock:
            records_copy = list(self._records)
            feedback_copy = dict(self._feedback)

        model_versions: Counter[str] = Counter(r.model_version for r in records_copy)
        dominant_version = model_versions.most_common(1)[0][0] if model_versions else "unknown"

        evaluated = [r for r in records_copy if r.confirmed_fraud is not None]

        tp = sum(1 for r in evaluated if r.is_flagged and r.confirmed_fraud is True)
        fp = sum(1 for r in evaluated if r.is_flagged and r.confirmed_fraud is False)
        tn = sum(1 for r in evaluated if not r.is_flagged and r.confirmed_fraud is False)
        fn = sum(1 for r in evaluated if not r.is_flagged and r.confirmed_fraud is True)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        pos_scores = [r.fraud_score for r in evaluated if r.confirmed_fraud is True]
        neg_scores = [r.fraud_score for r in evaluated if r.confirmed_fraud is False]

        # Rough AUC-ROC estimate using mean score separation
        avg_pos = statistics.mean(pos_scores) if pos_scores else 0.0
        avg_neg = statistics.mean(neg_scores) if neg_scores else 0.0
        auc_estimate = min(1.0, max(0.0, 0.5 + (avg_pos - avg_neg) / 2))

        return ModelPerformanceMetrics(
            model_version=dominant_version,
            total_evaluated=len(evaluated),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            auc_roc_estimate=round(auc_estimate, 4),
            avg_score_positive=round(avg_pos, 4),
            avg_score_negative=round(avg_neg, 4),
        )

    # ── Geographic heatmap ────────────────────────────────────────────────────

    def get_geographic_heatmap(self) -> list[GeoPoint]:
        """Return per-country aggregated fraud statistics for a heatmap."""
        now = datetime.now(timezone.utc)
        cutoff = now - _WINDOW_24H

        with self._lock:
            records_copy = list(self._records)

        recent = [r for r in records_copy if _aware(r.scored_at) >= cutoff]

        country_totals: dict[str, list[_TxnRecord]] = defaultdict(list)
        country_fraud: dict[str, int] = defaultdict(int)

        for r in recent:
            cc = r.country_code or "XX"
            country_totals[cc].append(r)
            if r.is_flagged:
                country_fraud[cc] += 1

        geo_points: list[GeoPoint] = []
        for cc, recs in country_totals.items():
            coords = _COUNTRY_COORDS.get(cc, (cc, 0.0, 0.0))
            geo_points.append(GeoPoint(
                country_code=cc[:2].upper(),
                country_name=coords[0],
                transaction_count=len(recs),
                flagged_count=country_fraud[cc],
                fraud_rate=round(country_fraud[cc] / len(recs), 4),
                avg_amount=round(statistics.mean(r.amount for r in recs), 2),
                latitude=coords[1],
                longitude=coords[2],
            ))

        return sorted(geo_points, key=lambda p: p.flagged_count, reverse=True)

    # ── Period comparison ─────────────────────────────────────────────────────

    def compare_periods(
        self,
        metric: MetricType,
        period_1: tuple[datetime, datetime],
        period_2: tuple[datetime, datetime],
    ) -> PeriodComparison:
        """Compare a metric across two arbitrary time periods."""
        label_1 = f"{period_1[0].date()} – {period_1[1].date()}"
        label_2 = f"{period_2[0].date()} – {period_2[1].date()}"

        # Determine sensible granularity from period length
        span_1 = period_1[1] - period_1[0]
        granularity = TimeGranularity.HOUR if span_1 <= timedelta(days=2) else TimeGranularity.DAY

        ts1 = self.get_time_series(metric, granularity, period_1[0], period_1[1])
        ts2 = self.get_time_series(metric, granularity, period_2[0], period_2[1])

        val1 = ts1.summary.get("mean", 0.0)
        val2 = ts2.summary.get("mean", 0.0)

        abs_change = val2 - val1
        rel_change = ((val2 - val1) / val1 * 100.0) if val1 != 0 else 0.0

        # For fraud-style metrics a decrease is an improvement
        improvement_metrics = {
            MetricType.FRAUD_RATE, MetricType.BLOCK_RATE, MetricType.REVIEW_RATE,
            MetricType.LATENCY_P50, MetricType.LATENCY_P95, MetricType.FALSE_POSITIVE_RATE,
        }
        is_improvement = abs_change < 0 if metric in improvement_metrics else abs_change > 0

        return PeriodComparison(
            metric_type=metric,
            period_1_label=label_1,
            period_2_label=label_2,
            period_1_value=round(val1, 6),
            period_2_value=round(val2, 6),
            absolute_change=round(abs_change, 6),
            relative_change_pct=round(rel_change, 2),
            is_improvement=is_improvement,
            period_1_points=ts1.points,
            period_2_points=ts2.points,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_hourly_trend(
        self,
        records: list[_TxnRecord],
        now: datetime,
    ) -> list[TimeSeriesPoint]:
        points: list[TimeSeriesPoint] = []
        for h in range(23, -1, -1):
            bucket_start = (now - timedelta(hours=h + 1)).replace(minute=0, second=0, microsecond=0)
            bucket_end = bucket_start + timedelta(hours=1)
            bucket = [r for r in records if bucket_start <= _aware(r.scored_at) < bucket_end]
            rate = sum(1 for r in bucket if r.is_flagged) / len(bucket) if bucket else 0.0
            points.append(TimeSeriesPoint(
                timestamp=bucket_start,
                value=round(rate, 6),
                metadata={"sample_count": len(bucket)},
            ))
        return points


# ── Internal helpers ──────────────────────────────────────────────────────────


def _aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (assume UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _granularity_delta(granularity: TimeGranularity) -> timedelta:
    return {
        TimeGranularity.MINUTE: timedelta(minutes=1),
        TimeGranularity.HOUR: timedelta(hours=1),
        TimeGranularity.DAY: timedelta(days=1),
        TimeGranularity.WEEK: timedelta(weeks=1),
        TimeGranularity.MONTH: timedelta(days=30),
    }[granularity]


def _build_buckets(
    from_ts: datetime,
    to_ts: datetime,
    delta: timedelta,
) -> list[tuple[datetime, datetime]]:
    buckets: list[tuple[datetime, datetime]] = []
    current = from_ts
    while current < to_ts:
        end = min(current + delta, to_ts)
        buckets.append((current, end))
        current += delta
    return buckets


def _compute_metric(metric_type: MetricType, records: list[_TxnRecord]) -> float:
    if not records:
        return 0.0

    if metric_type == MetricType.FRAUD_RATE:
        return sum(1 for r in records if r.is_flagged) / len(records)

    if metric_type == MetricType.BLOCK_RATE:
        return sum(1 for r in records if r.decision == "BLOCK") / len(records)

    if metric_type == MetricType.REVIEW_RATE:
        return sum(1 for r in records if r.decision == "REVIEW") / len(records)

    if metric_type == MetricType.THROUGHPUT:
        return float(len(records))

    latencies = sorted(r.latency_ms for r in records)
    n = len(latencies)

    if metric_type == MetricType.LATENCY_P50:
        return latencies[int(n * 0.50)]

    if metric_type == MetricType.LATENCY_P95:
        return latencies[min(int(n * 0.95), n - 1)]

    if metric_type == MetricType.FALSE_POSITIVE_RATE:
        evaluated = [r for r in records if r.confirmed_fraud is not None]
        if not evaluated:
            return 0.0
        fp = sum(1 for r in evaluated if r.is_flagged and r.confirmed_fraud is False)
        negatives = sum(1 for r in evaluated if r.confirmed_fraud is False)
        return fp / negatives if negatives else 0.0

    if metric_type == MetricType.MODEL_ACCURACY:
        evaluated = [r for r in records if r.confirmed_fraud is not None]
        if not evaluated:
            return 0.0
        correct = sum(
            1 for r in evaluated
            if (r.is_flagged and r.confirmed_fraud is True)
            or (not r.is_flagged and r.confirmed_fraud is False)
        )
        return correct / len(evaluated)

    return 0.0


def _trend_pct(values: list[float]) -> float:
    """Simple first-half vs second-half trend percentage."""
    if len(values) < 2:
        return 0.0
    mid = len(values) // 2
    first_avg = statistics.mean(values[:mid]) or 0.0
    second_avg = statistics.mean(values[mid:]) or 0.0
    if first_avg == 0:
        return 0.0
    return round((second_avg - first_avg) / first_avg * 100.0, 2)
