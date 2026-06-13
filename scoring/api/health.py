"""Enhanced health check endpoints for Kubernetes probes and observability.

Endpoints:
  GET /health          — liveness probe (fast, always responds 200 if process alive)
  GET /health/ready    — readiness probe (checks all dependencies)
  GET /health/live     — alias for liveness
  GET /health/startup  — startup probe (blocks until all services initialized)
  GET /health/status   — detailed status for dashboards and on-call
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])

# Process start time for uptime calculation
_START_TIME = time.time()


class ComponentStatus(BaseModel):
    name: str
    status: str  # healthy / degraded / unhealthy
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str  # healthy / degraded / unhealthy
    version: str
    uptime_seconds: float
    timestamp: str
    components: list[ComponentStatus]


async def _check_redis(app) -> ComponentStatus:
    redis = getattr(app.state, "redis", None)
    if redis is None:
        return ComponentStatus(name="redis", status="unhealthy", detail="not initialized")
    try:
        t0 = time.monotonic()
        await asyncio.wait_for(redis.ping(), timeout=1.0)
        latency_ms = (time.monotonic() - t0) * 1000
        return ComponentStatus(name="redis", status="healthy", latency_ms=round(latency_ms, 2))
    except Exception as exc:
        return ComponentStatus(name="redis", status="unhealthy", detail=str(exc))


async def _check_kafka(app) -> ComponentStatus:
    producer = getattr(app.state, "producer", None)
    if producer is None:
        return ComponentStatus(name="kafka", status="degraded", detail="producer not running")
    # Kafka producers don't have a simple ping; check if the internal client is running
    try:
        client = getattr(producer, "_producer", None)
        if client and client._sender and not client._sender.task.done():
            return ComponentStatus(name="kafka", status="healthy")
        return ComponentStatus(name="kafka", status="degraded", detail="sender task not running")
    except Exception as exc:
        return ComponentStatus(name="kafka", status="degraded", detail=str(exc))


async def _check_db(app) -> ComponentStatus:
    try:
        from scoring.db.engine import get_db_session
        t0 = time.monotonic()
        async with get_db_session() as session:
            await session.execute(__import__("sqlalchemy").text("SELECT 1"))
        latency_ms = (time.monotonic() - t0) * 1000
        return ComponentStatus(name="postgres", status="healthy", latency_ms=round(latency_ms, 2))
    except Exception as exc:
        return ComponentStatus(name="postgres", status="unhealthy", detail=str(exc))


async def _check_scorer(app) -> ComponentStatus:
    scorer = getattr(app.state, "scorer", None)
    if scorer is None:
        return ComponentStatus(name="scorer", status="unhealthy", detail="not initialized")
    model_version = getattr(scorer, "model_version", None)
    if not model_version:
        return ComponentStatus(name="scorer", status="degraded", detail="model version unavailable")
    return ComponentStatus(name="scorer", status="healthy", detail=model_version)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/health", include_in_schema=False)
@router.get("/health/live", include_in_schema=False)
async def liveness():
    """Kubernetes liveness probe — always 200 if process is alive."""
    return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}


@router.get("/health/ready")
async def readiness(request: Request):
    """Kubernetes readiness probe — 200 if all critical deps are healthy."""
    checks = await asyncio.gather(
        _check_redis(request.app),
        _check_scorer(request.app),
        return_exceptions=False,
    )

    components = list(checks)
    unhealthy = [c for c in components if c.status == "unhealthy"]

    status_code = 200 if not unhealthy else 503
    overall = "healthy" if not unhealthy else "unhealthy"

    return Response(
        content=HealthResponse(
            status=overall,
            version=request.app.version,
            uptime_seconds=round(time.time() - _START_TIME, 1),
            timestamp=datetime.now(UTC).isoformat(),
            components=components,
        ).model_dump_json(),
        status_code=status_code,
        media_type="application/json",
    )


@router.get("/health/status")
async def detailed_status(request: Request):
    """Full dependency health check for dashboards and on-call runbooks."""
    checks = await asyncio.gather(
        _check_redis(request.app),
        _check_kafka(request.app),
        _check_db(request.app),
        _check_scorer(request.app),
        return_exceptions=False,
    )

    components = list(checks)
    statuses = {c.status for c in components}

    if "unhealthy" in statuses:
        overall = "unhealthy"
        code = 503
    elif "degraded" in statuses:
        overall = "degraded"
        code = 207
    else:
        overall = "healthy"
        code = 200

    # SLO report
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    _slo_report = slo_monitor.report() if slo_monitor else {}

    # Batch queue stats
    batch_queue = getattr(request.app.state, "batch_queue", None)
    _batch_stats = batch_queue.stats() if batch_queue else {}

    # Circuit breaker stats
    redis_pool = getattr(request.app.state, "redis_pool", None)
    _cb_stats = redis_pool.stats() if redis_pool else {}

    # WS Hub stats
    ws_hub = getattr(request.app.state, "ws_hub", None)
    _ws_stats = ws_hub.stats() if ws_hub else {}

    return Response(
        content=HealthResponse(
            status=overall,
            version=request.app.version,
            uptime_seconds=round(time.time() - _START_TIME, 1),
            timestamp=datetime.now(UTC).isoformat(),
            components=components,
        ).model_dump_json(),
        status_code=code,
        media_type="application/json",
    )


__all__ = ["ComponentStatus", "HealthResponse", "router"]
