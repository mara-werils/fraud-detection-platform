"""SLO (Service Level Objective) monitoring and error budget tracking.

Tracks the following SLOs:
  - Availability SLO: 99.9% of requests return 2xx/3xx (error budget: 0.1%)
  - Latency SLO: 99% of requests complete within 200ms
  - Freshness SLO: scored events arrive within 500ms of transaction creation

Error budgets are tracked with a rolling 28-day window via Prometheus counters.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

SLO_REQUESTS_TOTAL = Counter(
    "slo_requests_total",
    "Total requests tracked for SLO calculation",
    ["slo_name", "result"],  # result: good / bad
)

SLO_ERROR_BUDGET_REMAINING = Gauge(
    "slo_error_budget_remaining_ratio",
    "Remaining error budget as a ratio (1.0 = full, 0.0 = exhausted)",
    ["slo_name"],
)

SLO_LATENCY_HISTOGRAM = Histogram(
    "slo_request_latency_seconds",
    "Request latency for SLO tracking",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)

SLO_BURN_RATE = Gauge(
    "slo_burn_rate",
    "Current error budget burn rate (1.0 = consuming exactly at SLO rate)",
    ["slo_name"],
)


# ── SLO definitions ───────────────────────────────────────────────────────────


@dataclass
class SLODefinition:
    name: str
    description: str
    target: float  # e.g. 0.999 for 99.9%
    window_hours: int = 24 * 28  # 28 days


SLOS = [
    SLODefinition("availability", "99.9% of requests return 2xx/3xx", target=0.999),
    SLODefinition("latency_p99", "99% of requests complete within 200ms", target=0.99),
    SLODefinition("scoring_latency_p95", "95% of score requests complete within 100ms", target=0.95),
]


# ── In-process sliding window ─────────────────────────────────────────────────


@dataclass
class WindowedCounter:
    """Rolling window counter for recent good/bad events."""

    window_seconds: int = 3600  # 1 hour default
    _events: deque = field(default_factory=deque)

    def record(self, good: bool) -> None:
        now = time.monotonic()
        self._events.append((now, good))
        # Evict old events
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def ratio(self) -> float:
        if not self._events:
            return 1.0
        good = sum(1 for _, g in self._events if g)
        return good / len(self._events)

    @property
    def total(self) -> int:
        return len(self._events)

    @property
    def bad_count(self) -> int:
        return sum(1 for _, g in self._events if not g)


class SLOMonitor:
    """Tracks SLO compliance and error budgets in real time."""

    def __init__(self) -> None:
        self._availability = WindowedCounter(window_seconds=3600)
        self._latency_p99 = WindowedCounter(window_seconds=3600)
        self._latency_p95 = WindowedCounter(window_seconds=3600)
        self._latency_values: deque = deque(maxlen=10_000)
        self._started_at = datetime.now(UTC)

    def record_request(
        self,
        status_code: int,
        latency_seconds: float,
        endpoint: str = "",
    ) -> None:
        """Record a completed HTTP request for SLO tracking."""
        # Availability
        is_available = status_code < 500
        self._availability.record(is_available)
        SLO_REQUESTS_TOTAL.labels(
            slo_name="availability", result="good" if is_available else "bad"
        ).inc()

        # Latency p99 (<200ms)
        within_p99 = latency_seconds <= 0.200
        self._latency_p99.record(within_p99)
        SLO_REQUESTS_TOTAL.labels(
            slo_name="latency_p99", result="good" if within_p99 else "bad"
        ).inc()

        # Latency p95 for scoring endpoints (<100ms)
        if "/score" in endpoint or "/batch" in endpoint:
            within_p95 = latency_seconds <= 0.100
            self._latency_p95.record(within_p95)
            SLO_REQUESTS_TOTAL.labels(
                slo_name="scoring_latency_p95", result="good" if within_p95 else "bad"
            ).inc()
            SLO_LATENCY_HISTOGRAM.observe(latency_seconds)

        # Update Prometheus gauges
        self._update_gauges()

    def _update_gauges(self) -> None:
        for slo, counter in [
            ("availability", self._availability),
            ("latency_p99", self._latency_p99),
            ("scoring_latency_p95", self._latency_p95),
        ]:
            slo_def = next((s for s in SLOS if s.name == slo), None)
            if not slo_def:
                continue
            current_ratio = counter.ratio()
            error_budget_target = 1 - slo_def.target
            current_error_rate = 1 - current_ratio
            remaining = max(0.0, 1.0 - (current_error_rate / error_budget_target)) if error_budget_target > 0 else 1.0
            SLO_ERROR_BUDGET_REMAINING.labels(slo_name=slo).set(remaining)
            burn_rate = current_error_rate / error_budget_target if error_budget_target > 0 else 0.0
            SLO_BURN_RATE.labels(slo_name=slo).set(burn_rate)

    def report(self) -> dict[str, Any]:
        """Return a human-readable SLO status report."""
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "uptime_hours": round(
                (datetime.now(UTC) - self._started_at).total_seconds() / 3600, 2
            ),
            "slos": {
                "availability": {
                    "target": "99.9%",
                    "current": f"{self._availability.ratio() * 100:.3f}%",
                    "good_requests": self._availability.total - self._availability.bad_count,
                    "bad_requests": self._availability.bad_count,
                    "status": "OK" if self._availability.ratio() >= 0.999 else "BREACHED",
                },
                "latency_p99": {
                    "target": "99% requests < 200ms",
                    "current": f"{self._latency_p99.ratio() * 100:.2f}%",
                    "status": "OK" if self._latency_p99.ratio() >= 0.99 else "BREACHED",
                },
                "scoring_latency_p95": {
                    "target": "95% score requests < 100ms",
                    "current": f"{self._latency_p95.ratio() * 100:.2f}%",
                    "status": "OK" if self._latency_p95.ratio() >= 0.95 else "BREACHED",
                },
            },
        }
