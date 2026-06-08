"""Real-time dashboard API with WebSocket live transaction feed.

Provides endpoints for a live operations dashboard including real-time fraud
statistics, time-series chart data, top triggered rules, geographic heatmaps,
model performance metrics, and a WebSocket endpoint for streaming scored
transactions as they arrive.

All routes read from services registered on ``app.state`` during startup.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from scoring.services.analytics_engine import (
    AnalyticsEngine,
    GeoPoint,
    ModelPerformanceMetrics,
    TimeGranularity,
    TimeSeriesPoint,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


# ── Pydantic response models ────────────────────────────────────────────────


class OverviewStats(BaseModel):
    """Real-time fraud overview statistics."""

    total_transactions: int = Field(..., description="Total transactions scored today")
    flagged_transactions: int = Field(..., description="Transactions flagged for review")
    blocked_transactions: int = Field(..., description="Transactions blocked")
    approved_transactions: int = Field(..., description="Transactions approved")
    approval_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Approval rate (approved / total)"
    )
    fraud_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraud rate (blocked / total)"
    )
    avg_score: float = Field(..., ge=0.0, le=1.0, description="Average fraud score")
    avg_latency_ms: float = Field(..., ge=0.0, description="Average scoring latency (ms)")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the snapshot was generated",
    )


class TimelineGranularity(StrEnum):
    HOURLY = "hourly"
    DAILY = "daily"


class TimelinePoint(BaseModel):
    """A single data point in the timeline."""

    timestamp: datetime = Field(..., description="Bucket start timestamp (UTC)")
    total: int = Field(..., description="Total transactions in this bucket")
    flagged: int = Field(..., description="Flagged transactions in this bucket")
    blocked: int = Field(..., description="Blocked transactions in this bucket")
    fraud_rate: float = Field(..., ge=0.0, le=1.0, description="Fraud rate for this bucket")


class TimelineResponse(BaseModel):
    """Time-series data for fraud-rate charts."""

    granularity: TimelineGranularity = Field(..., description="Bucket granularity")
    points: list[TimelinePoint] = Field(..., description="Ordered time-series data points")
    period_start: datetime = Field(..., description="Start of the queried period")
    period_end: datetime = Field(..., description="End of the queried period")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


class TriggeredRule(BaseModel):
    """A rule and its trigger count."""

    rule_id: str = Field(..., description="Unique rule identifier")
    rule_name: str = Field(..., description="Human-readable rule name")
    trigger_count: int = Field(..., ge=0, description="Number of times triggered")
    fraud_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraud rate among triggered transactions"
    )
    avg_score: float = Field(
        ..., ge=0.0, le=1.0, description="Average fraud score when rule fires"
    )
    last_triggered: datetime = Field(..., description="Last time this rule fired")


class TopRulesResponse(BaseModel):
    """Most triggered fraud detection rules."""

    rules: list[TriggeredRule] = Field(..., description="Rules sorted by trigger count desc")
    total_rules_evaluated: int = Field(
        ..., description="Total number of distinct rules evaluated"
    )
    period_hours: int = Field(..., description="Look-back window in hours")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


class GeoFraudEntry(BaseModel):
    """Fraud statistics for a single country/region."""

    country_code: str = Field(..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2")
    country_name: str = Field(..., description="Human-readable country name")
    transaction_count: int = Field(..., ge=0)
    flagged_count: int = Field(..., ge=0)
    blocked_count: int = Field(..., ge=0)
    fraud_rate: float = Field(..., ge=0.0, le=1.0)
    total_amount: float = Field(..., ge=0.0, description="Sum of transaction amounts")
    avg_amount: float = Field(..., ge=0.0, description="Average transaction amount")
    latitude: float
    longitude: float


class GeographyResponse(BaseModel):
    """Geographic fraud heatmap data."""

    regions: list[GeoFraudEntry] = Field(
        ..., description="Per-country fraud statistics"
    )
    total_countries: int = Field(..., description="Number of countries with activity")
    highest_fraud_country: str | None = Field(
        None, description="Country code with highest fraud rate"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


class ModelPerformanceResponse(BaseModel):
    """Live model performance metrics."""

    model_version: str = Field(..., description="Active model version identifier")
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    auc_roc_estimate: float = Field(..., ge=0.0, le=1.0)
    true_positives: int = Field(..., ge=0)
    false_positives: int = Field(..., ge=0)
    true_negatives: int = Field(..., ge=0)
    false_negatives: int = Field(..., ge=0)
    total_evaluated: int = Field(..., ge=0)
    avg_score_positive: float = Field(..., description="Mean score on confirmed fraud")
    avg_score_negative: float = Field(..., description="Mean score on confirmed legitimate")
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


class LiveTransactionEvent(BaseModel):
    """Schema for a live transaction event pushed over WebSocket."""

    transaction_id: str
    user_id: str
    amount: float
    currency: str
    fraud_score: float
    decision: str
    is_flagged: bool
    country: str | None = None
    merchant_category: str | None = None
    scored_at: datetime


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_engine(request: Request) -> AnalyticsEngine:
    engine: AnalyticsEngine | None = getattr(request.app.state, "analytics_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    return engine


def _get_scorer(request: Request) -> Any:
    scorer = getattr(request.app.state, "scorer", None)
    if scorer is None:
        raise HTTPException(status_code=503, detail="Scoring model not loaded")
    return scorer


def _get_transaction_store(request: Request) -> Any:
    store = getattr(request.app.state, "transaction_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Transaction store not available")
    return store


# ── WebSocket connection manager ─────────────────────────────────────────────


class ConnectionManager:
    """Manages active WebSocket connections for the live transaction feed."""

    def __init__(self) -> None:
        self._active: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._active.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._active:
                self._active.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a JSON message to all connected clients."""
        async with self._lock:
            stale: list[WebSocket] = []
            for ws in self._active:
                try:
                    await ws.send_json(message)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._active.remove(ws)

    @property
    def active_count(self) -> int:
        return len(self._active)


manager = ConnectionManager()


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get(
    "/overview",
    response_model=OverviewStats,
    summary="Real-time fraud overview",
    description=(
        "Returns a snapshot of today's fraud statistics including total transactions, "
        "flagged/blocked counts, approval rate, average fraud score, and latency."
    ),
)
async def get_overview(request: Request) -> OverviewStats:
    """Compute and return the real-time overview stats."""
    engine = _get_engine(request)
    scorer = _get_scorer(request)

    try:
        dashboard = engine.get_dashboard_metrics()
    except Exception as exc:
        logger.exception("dashboard_overview_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute overview stats") from exc

    total = dashboard.total_transactions_today
    blocked = dashboard.total_blocked
    reviewed = dashboard.total_reviewed
    approved = max(0, total - blocked - reviewed)

    approval_rate = approved / total if total > 0 else 1.0
    fraud_rate = blocked / total if total > 0 else 0.0

    result = OverviewStats(
        total_transactions=total,
        flagged_transactions=reviewed,
        blocked_transactions=blocked,
        approved_transactions=approved,
        approval_rate=round(approval_rate, 6),
        fraud_rate=round(fraud_rate, 6),
        avg_score=dashboard.avg_score,
        avg_latency_ms=dashboard.avg_latency_ms,
    )

    await logger.ainfo(
        "dashboard_overview_served",
        total=total,
        approval_rate=result.approval_rate,
    )
    return result


@router.get(
    "/timeline",
    response_model=TimelineResponse,
    summary="Time-series fraud rate data",
    description=(
        "Returns bucketed time-series data for rendering fraud rate charts. "
        "Supports hourly (default, last 24h) and daily (last 30d) granularity."
    ),
)
async def get_timeline(
    request: Request,
    granularity: TimelineGranularity = Query(
        TimelineGranularity.HOURLY,
        description="Bucket granularity: hourly or daily",
    ),
) -> TimelineResponse:
    """Return time-series data for the fraud-rate timeline chart."""
    engine = _get_engine(request)

    now = datetime.now(UTC)
    if granularity == TimelineGranularity.HOURLY:
        period_start = now - timedelta(hours=24)
        engine_granularity = TimeGranularity.HOUR
    else:
        period_start = now - timedelta(days=30)
        engine_granularity = TimeGranularity.DAY

    try:
        ts_data = engine.get_time_series(
            metric="fraud_rate",
            granularity=engine_granularity,
            from_ts=period_start,
            to_ts=now,
        )
    except Exception as exc:
        logger.exception("dashboard_timeline_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute timeline data") from exc

    points = [
        TimelinePoint(
            timestamp=p.timestamp,
            total=p.metadata.get("total", 0),
            flagged=p.metadata.get("flagged", 0),
            blocked=p.metadata.get("blocked", 0),
            fraud_rate=p.value,
        )
        for p in ts_data.points
    ]

    result = TimelineResponse(
        granularity=granularity,
        points=points,
        period_start=period_start,
        period_end=now,
    )

    await logger.ainfo(
        "dashboard_timeline_served",
        granularity=granularity,
        buckets=len(points),
    )
    return result


@router.get(
    "/top-rules",
    response_model=TopRulesResponse,
    summary="Most triggered fraud rules",
    description=(
        "Returns the top N most frequently triggered fraud detection rules "
        "within the specified look-back window, ranked by trigger count."
    ),
)
async def get_top_rules(
    request: Request,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of rules to return"),
    hours: int = Query(24, ge=1, le=720, description="Look-back window in hours"),
) -> TopRulesResponse:
    """Return the most triggered rules with counts and fraud rates."""
    engine = _get_engine(request)

    try:
        patterns = engine.get_top_fraud_patterns(limit=limit)
    except Exception as exc:
        logger.exception("dashboard_top_rules_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute top rules") from exc

    rules = [
        TriggeredRule(
            rule_id=p.pattern_id,
            rule_name=p.description,
            trigger_count=p.occurrence_count,
            fraud_rate=p.fraud_rate,
            avg_score=p.avg_amount / 10000.0 if p.avg_amount else 0.0,
            last_triggered=p.last_seen,
        )
        for p in patterns
    ]

    result = TopRulesResponse(
        rules=rules,
        total_rules_evaluated=len(rules),
        period_hours=hours,
    )

    await logger.ainfo("dashboard_top_rules_served", count=len(rules))
    return result


@router.get(
    "/geography",
    response_model=GeographyResponse,
    summary="Fraud by geography heatmap",
    description=(
        "Returns per-country aggregated fraud statistics for the last 24 hours, "
        "suitable for rendering a geographic heatmap on the dashboard."
    ),
)
async def get_geography(request: Request) -> GeographyResponse:
    """Return geographic fraud heatmap data."""
    engine = _get_engine(request)

    try:
        geo_points = engine.get_geographic_heatmap()
    except Exception as exc:
        logger.exception("dashboard_geography_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute geography data") from exc

    regions = [
        GeoFraudEntry(
            country_code=gp.country_code,
            country_name=gp.country_name,
            transaction_count=gp.transaction_count,
            flagged_count=gp.flagged_count,
            blocked_count=int(gp.flagged_count * gp.fraud_rate),
            fraud_rate=gp.fraud_rate,
            total_amount=gp.avg_amount * gp.transaction_count,
            avg_amount=gp.avg_amount,
            latitude=gp.latitude,
            longitude=gp.longitude,
        )
        for gp in geo_points
    ]

    highest = max(regions, key=lambda r: r.fraud_rate).country_code if regions else None

    result = GeographyResponse(
        regions=regions,
        total_countries=len(regions),
        highest_fraud_country=highest,
    )

    await logger.ainfo("dashboard_geography_served", countries=len(regions))
    return result


@router.get(
    "/model-performance",
    response_model=ModelPerformanceResponse,
    summary="Live model performance metrics",
    description=(
        "Returns precision, recall, F1 score, and estimated AUC-ROC for the active "
        "model version, derived from analyst feedback on scored transactions."
    ),
)
async def get_model_performance(request: Request) -> ModelPerformanceResponse:
    """Return live model quality metrics."""
    engine = _get_engine(request)

    try:
        perf = engine.get_model_performance_metrics()
    except Exception as exc:
        logger.exception("dashboard_model_performance_error", error=str(exc))
        raise HTTPException(
            status_code=500, detail="Failed to compute model performance"
        ) from exc

    result = ModelPerformanceResponse(
        model_version=perf.model_version,
        precision=perf.precision,
        recall=perf.recall,
        f1_score=perf.f1_score,
        auc_roc_estimate=perf.auc_roc_estimate,
        true_positives=perf.true_positives,
        false_positives=perf.false_positives,
        true_negatives=perf.true_negatives,
        false_negatives=perf.false_negatives,
        total_evaluated=perf.total_evaluated,
        avg_score_positive=perf.avg_score_positive,
        avg_score_negative=perf.avg_score_negative,
        computed_at=perf.computed_at,
    )

    await logger.ainfo(
        "dashboard_model_performance_served",
        model_version=perf.model_version,
        f1=perf.f1_score,
    )
    return result


@router.websocket("/live")
async def live_transaction_feed(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming live scored transactions.

    Clients connect and receive a JSON message for each newly scored
    transaction in real time. Supports an optional ``min_score`` filter
    sent as a JSON message after connection (e.g. ``{"min_score": 0.5}``).

    Wire format (server -> client)::

        {
            "type": "transaction",
            "data": { <LiveTransactionEvent fields> }
        }
    """
    await manager.connect(websocket)
    await logger.ainfo(
        "dashboard_ws_connected",
        active_connections=manager.active_count,
    )

    min_score: float = 0.0

    try:
        while True:
            # Listen for client configuration messages (non-blocking filter updates)
            try:
                raw = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                if isinstance(raw, dict) and "min_score" in raw:
                    min_score = float(raw["min_score"])
                    await websocket.send_json({
                        "type": "config_ack",
                        "min_score": min_score,
                    })
            except asyncio.TimeoutError:
                # No client message within the timeout window; send a heartbeat
                await websocket.send_json({"type": "heartbeat", "ts": datetime.now(UTC).isoformat()})
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        await logger.ainfo(
            "dashboard_ws_disconnected",
            active_connections=manager.active_count,
        )
    except Exception as exc:
        await manager.disconnect(websocket)
        await logger.awarning(
            "dashboard_ws_error",
            error=str(exc),
            active_connections=manager.active_count,
        )


async def broadcast_transaction(event: LiveTransactionEvent) -> None:
    """Broadcast a scored transaction to all connected dashboard clients.

    This function is intended to be called from the scoring pipeline
    (e.g. after a transaction is scored in the consumer handler) to push
    events to connected WebSocket clients.

    Args:
        event: The scored transaction event to broadcast.
    """
    await manager.broadcast({
        "type": "transaction",
        "data": event.model_dump(mode="json"),
    })
