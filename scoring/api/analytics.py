"""Real-time metrics and analytics API.

Provides endpoints for the operations dashboard, time-series queries,
score distributions, fraud pattern discovery, model performance reporting,
and geographic heatmap data.

All routes read from the AnalyticsEngine instance registered on
``app.state.analytics_engine`` during application startup.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from scoring.services.analytics_engine import (
    AnalyticsEngine,
    DashboardMetrics,
    FraudPattern,
    GeoPoint,
    MetricType,
    ModelPerformanceMetrics,
    PeriodComparison,
    TimeGranularity,
    TimeSeriesData,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_engine(request: Request) -> AnalyticsEngine:
    engine: AnalyticsEngine | None = getattr(request.app.state, "analytics_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    return engine


def _parse_ts(value: str | None, default: datetime) -> datetime:
    """Parse an ISO 8601 timestamp query parameter, falling back to *default*."""
    if value is None:
        return default
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid timestamp format '{value}'. Use ISO 8601 (e.g. 2026-05-25T00:00:00Z).",
        ) from exc


# ── Response wrappers ─────────────────────────────────────────────────────────


class PatternsResponse(BaseModel):
    patterns: list[FraudPattern]
    total: int


class GeoResponse(BaseModel):
    points: list[GeoPoint]
    total_countries: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DistributionResponse(BaseModel):
    distribution: dict[str, int]
    percentiles: dict[str, float]
    total_samples: int
    mean: float
    stddev: float
    from_ts: datetime
    to_ts: datetime


class CompareRequest(BaseModel):
    """Request body for period-over-period comparison."""

    metric_type: MetricType
    period_1_from: datetime
    period_1_to: datetime
    period_2_from: datetime
    period_2_to: datetime


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "/dashboard",
    response_model=DashboardMetrics,
    summary="Real-time operations dashboard",
    description=(
        "Returns a snapshot of platform-wide metrics for the operations dashboard: "
        "transaction volumes, fraud rates, top fraud categories and countries, "
        "score distribution histogram, and hourly fraud-rate trend."
    ),
)
async def get_dashboard(request: Request) -> DashboardMetrics:
    """Compute and return the real-time dashboard metrics snapshot."""
    engine = _get_engine(request)

    try:
        metrics = engine.get_dashboard_metrics()
    except Exception as exc:
        logger.exception("analytics_dashboard_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute dashboard metrics") from exc

    await logger.ainfo(
        "analytics_dashboard_served",
        total_today=metrics.total_transactions_today,
        fraud_rate_24h=metrics.fraud_rate_24h,
    )
    return metrics


@router.get(
    "/timeseries/{metric_type}",
    response_model=TimeSeriesData,
    summary="Time-series metric query",
    description=(
        "Aggregate any supported metric over an arbitrary date range at a chosen granularity. "
        "Use ``from`` / ``to`` query params (ISO 8601) and ``granularity`` "
        "(minute | hour | day | week | month)."
    ),
)
async def get_timeseries(
    metric_type: MetricType,
    request: Request,
    granularity: TimeGranularity = Query(
        TimeGranularity.HOUR,
        description="Bucket granularity",
    ),
    from_ts: str | None = Query(
        None,
        alias="from",
        description="Start of range (ISO 8601, default: 24 h ago)",
    ),
    to_ts: str | None = Query(
        None,
        alias="to",
        description="End of range (ISO 8601, default: now)",
    ),
) -> TimeSeriesData:
    """Return a bucketed time series for the requested metric."""
    engine = _get_engine(request)

    now = datetime.now(UTC)
    parsed_from = _parse_ts(from_ts, now - timedelta(hours=24))
    parsed_to = _parse_ts(to_ts, now)

    if parsed_from >= parsed_to:
        raise HTTPException(
            status_code=422,
            detail="'from' must be strictly before 'to'.",
        )

    max_span = timedelta(days=366)
    if (parsed_to - parsed_from) > max_span:
        raise HTTPException(
            status_code=422,
            detail="Requested time range exceeds the maximum of 366 days.",
        )

    try:
        result = engine.get_time_series(metric_type, granularity, parsed_from, parsed_to)
    except Exception as exc:
        logger.exception("analytics_timeseries_error", metric=metric_type, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute time series") from exc

    await logger.ainfo(
        "analytics_timeseries_served",
        metric=metric_type,
        granularity=granularity,
        buckets=len(result.points),
    )
    return result


@router.get(
    "/distribution",
    response_model=DistributionResponse,
    summary="Fraud score distribution",
    description=(
        "Returns a histogram of fraud scores across 10 equal-width buckets (0.0–1.0), "
        "along with key percentiles and summary statistics."
    ),
)
async def get_distribution(
    request: Request,
    from_ts: str | None = Query(
        None,
        alias="from",
        description="Start of window (ISO 8601, default: 24 h ago)",
    ),
    to_ts: str | None = Query(
        None,
        alias="to",
        description="End of window (ISO 8601, default: now)",
    ),
) -> DistributionResponse:
    """Return the fraud score distribution for a given time window."""
    engine = _get_engine(request)

    now = datetime.now(UTC)
    parsed_from = _parse_ts(from_ts, now - timedelta(hours=24))
    parsed_to = _parse_ts(to_ts, now)

    if parsed_from >= parsed_to:
        raise HTTPException(status_code=422, detail="'from' must be strictly before 'to'.")

    try:
        result = engine.get_score_distribution(parsed_from, parsed_to)
    except Exception as exc:
        logger.exception("analytics_distribution_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute score distribution") from exc

    return DistributionResponse(
        distribution=result["distribution"],
        percentiles=result["percentiles"],
        total_samples=result["total_samples"],
        mean=result["mean"],
        stddev=result["stddev"],
        from_ts=parsed_from,
        to_ts=parsed_to,
    )


@router.get(
    "/patterns",
    response_model=PatternsResponse,
    summary="Top fraud patterns",
    description=(
        "Identifies the most prevalent recurring fraud signals from the last 72 hours, "
        "grouped by merchant category. Returns occurrence count, fraud rate, and time range."
    ),
)
async def get_patterns(
    request: Request,
    limit: int = Query(10, ge=1, le=50, description="Maximum patterns to return"),
) -> PatternsResponse:
    """Return the top-N recurring fraud patterns detected in recent data."""
    engine = _get_engine(request)

    try:
        patterns = engine.get_top_fraud_patterns(limit=limit)
    except Exception as exc:
        logger.exception("analytics_patterns_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute fraud patterns") from exc

    await logger.ainfo("analytics_patterns_served", count=len(patterns))
    return PatternsResponse(patterns=patterns, total=len(patterns))


@router.get(
    "/performance",
    response_model=ModelPerformanceMetrics,
    summary="Model performance metrics",
    description=(
        "Computes precision, recall, F1, and an estimated AUC-ROC from analyst feedback "
        "on flagged and passed transactions. Reflects the currently dominant model version."
    ),
)
async def get_performance(request: Request) -> ModelPerformanceMetrics:
    """Return model quality metrics derived from analyst-confirmed feedback."""
    engine = _get_engine(request)

    try:
        perf = engine.get_model_performance_metrics()
    except Exception as exc:
        logger.exception("analytics_performance_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute model performance") from exc

    await logger.ainfo(
        "analytics_performance_served",
        model_version=perf.model_version,
        precision=perf.precision,
        recall=perf.recall,
        f1=perf.f1_score,
    )
    return perf


@router.get(
    "/geo",
    response_model=GeoResponse,
    summary="Geographic fraud heatmap",
    description=(
        "Returns per-country aggregated fraud statistics (count, flagged count, fraud rate, "
        "average amount) over the last 24 hours, suitable for rendering a heatmap."
    ),
)
async def get_geo_heatmap(request: Request) -> GeoResponse:
    """Return geographic heatmap data for the last 24 hours."""
    engine = _get_engine(request)

    try:
        points = engine.get_geographic_heatmap()
    except Exception as exc:
        logger.exception("analytics_geo_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute geographic heatmap") from exc

    await logger.ainfo("analytics_geo_served", countries=len(points))
    return GeoResponse(points=points, total_countries=len(points))


@router.post(
    "/compare",
    response_model=PeriodComparison,
    summary="Period-over-period metric comparison",
    description=(
        "Compares a single metric between two arbitrary time periods and returns "
        "the absolute and relative change, as well as the per-bucket time series "
        "for both periods."
    ),
)
async def compare_periods(
    body: CompareRequest,
    request: Request,
) -> PeriodComparison:
    """Compare a metric between two time periods and return the delta."""
    engine = _get_engine(request)

    if body.period_1_from >= body.period_1_to:
        raise HTTPException(status_code=422, detail="period_1_from must be before period_1_to.")
    if body.period_2_from >= body.period_2_to:
        raise HTTPException(status_code=422, detail="period_2_from must be before period_2_to.")

    try:
        result = engine.compare_periods(
            metric=body.metric_type,
            period_1=(body.period_1_from, body.period_1_to),
            period_2=(body.period_2_from, body.period_2_to),
        )
    except Exception as exc:
        logger.exception("analytics_compare_error", metric=body.metric_type, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compare periods") from exc

    await logger.ainfo(
        "analytics_compare_served",
        metric=body.metric_type,
        rel_change_pct=result.relative_change_pct,
        is_improvement=result.is_improvement,
    )
    return result
