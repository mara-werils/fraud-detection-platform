"""API usage metering and billing unit tracking.

Records every API call with latency, status, and billed units.
Provides aggregation for billing dashboards and quota enforcement.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import func, select

from scoring.db.engine import get_db_session
from scoring.db.models import Organization, UsageRecord

logger = structlog.get_logger(__name__)

# Billing unit weights per endpoint category
BILLING_WEIGHTS: dict[str, int] = {
    "/api/v1/score": 1,
    "/api/v1/batch/score": 5,
    "/api/v1/sanctions/check": 2,
    "/api/v1/transactions": 0,
    "/api/v1/cases": 0,
    "/api/v1/feedback": 0,
    "/api/v1/export": 3,
}


def _billing_units(endpoint: str, status_code: int) -> int:
    """Calculate billing units for a request.

    Failed requests (5xx) are not billed.
    """
    if status_code >= 500:
        return 0
    for prefix, weight in BILLING_WEIGHTS.items():
        if endpoint.startswith(prefix):
            return weight
    return 0


class UsageMeter:
    """Collects usage events and flushes them to PostgreSQL in batches.

    Events are buffered in-memory and flushed every flush_interval_seconds.
    This avoids a DB write on every request.
    """

    def __init__(self, flush_interval_seconds: float = 10.0, buffer_size: int = 1000) -> None:
        self._buffer: list[dict] = []
        self._flush_interval = flush_interval_seconds
        self._buffer_size = buffer_size
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._counters: dict[str, int] = defaultdict(int)  # in-memory fast counters

    def start(self) -> None:
        """Start the background flush task."""
        self._flush_task = asyncio.create_task(self._flush_loop(), name="usage-meter-flush")

    async def stop(self) -> None:
        """Flush remaining buffer and stop."""
        if self._flush_task:
            self._flush_task.cancel()
        await self._flush()

    async def record(
        self,
        org_id: str | None,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        request_size: int = 0,
        response_size: int = 0,
    ) -> None:
        """Record an API call event."""
        if not org_id:
            return

        units = _billing_units(endpoint, status_code)
        event = {
            "id": str(uuid.uuid4()),
            "org_id": org_id,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "request_size": request_size,
            "response_size": response_size,
            "billed_units": units,
            "timestamp": datetime.now(timezone.utc),
        }

        # Fast in-memory counter
        self._counters[f"{org_id}:total"] += 1
        self._counters[f"{org_id}:units"] += units

        async with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._buffer_size:
                await self._flush()

    async def get_org_usage(self, org_id: str, since: datetime | None = None) -> dict[str, Any]:
        """Return aggregated usage for an org from the database."""
        async with get_db_session() as session:
            q = select(
                func.count(UsageRecord.id).label("total_requests"),
                func.sum(UsageRecord.billed_units).label("total_units"),
                func.avg(UsageRecord.latency_ms).label("avg_latency_ms"),
                func.sum(
                    (UsageRecord.status_code >= 500).cast(int)
                ).label("error_count"),
            ).where(UsageRecord.org_id == uuid.UUID(org_id))

            if since:
                q = q.where(UsageRecord.timestamp >= since)

            result = await session.execute(q)
            row = result.one()

        # Get quota info
        async with get_db_session() as session:
            org = await session.get(Organization, uuid.UUID(org_id))

        monthly_quota = org.monthly_quota if org else 0
        total_units = int(row.total_units or 0)

        return {
            "org_id": org_id,
            "total_requests": int(row.total_requests or 0),
            "total_billed_units": total_units,
            "monthly_quota": monthly_quota,
            "quota_used_pct": round(total_units / monthly_quota * 100, 2) if monthly_quota else 0,
            "avg_latency_ms": round(float(row.avg_latency_ms or 0), 2),
            "error_count": int(row.error_count or 0),
        }

    async def get_realtime_counts(self, org_id: str) -> dict[str, int]:
        """Return in-memory fast counters (since process start)."""
        return {
            "requests_this_session": self._counters.get(f"{org_id}:total", 0),
            "units_this_session": self._counters.get(f"{org_id}:units", 0),
        }

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            try:
                await self._flush()
            except Exception as exc:
                await logger.awarning("usage_meter_flush_error", error=str(exc))

    async def _flush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()

        records = [
            UsageRecord(
                id=uuid.UUID(e["id"]),
                org_id=uuid.UUID(e["org_id"]),
                endpoint=e["endpoint"],
                method=e["method"],
                status_code=e["status_code"],
                latency_ms=e["latency_ms"],
                request_size=e["request_size"],
                response_size=e["response_size"],
                billed_units=e["billed_units"],
                timestamp=e["timestamp"],
            )
            for e in batch
        ]

        try:
            async with get_db_session() as session:
                session.add_all(records)
            await logger.ainfo("usage_meter_flushed", count=len(records))
        except Exception as exc:
            await logger.awarning("usage_meter_db_flush_failed", error=str(exc), count=len(records))
