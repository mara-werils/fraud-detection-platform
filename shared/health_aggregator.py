"""Health check aggregator for multi-service status monitoring.

Collects health status from registered service probes and computes an
overall system health.  Each probe is an async callable that returns a
``ComponentHealth``.  The aggregator merges individual results into a
summary with one of three statuses:

  - **healthy**: all components report healthy.
  - **degraded**: at least one component is unhealthy but the system can
    still serve requests (e.g. a cache is down but the database is up).
  - **unhealthy**: one or more *critical* components are unhealthy.

Designed for integration with Kubernetes liveness / readiness probes and
the ``/health`` endpoint in the scoring service.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Awaitable

import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "HealthProbe",
    "HealthAggregator",
]


class HealthStatus(StrEnum):
    """Overall health state for a component or the whole system."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health report for a single component."""

    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class SystemHealth:
    """Aggregated health report for the entire system."""

    status: HealthStatus
    components: list[ComponentHealth]
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "checked_at": self.checked_at,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": round(c.latency_ms, 2),
                    "details": c.details,
                    **({"error": c.error} if c.error else {}),
                }
                for c in self.components
            ],
        }


# Type alias for a health probe coroutine.
HealthProbe = Callable[[], Awaitable[ComponentHealth]]


class HealthAggregator:
    """Aggregates health probes into a single system health report.

    Parameters
    ----------
    timeout_seconds:
        Maximum time to wait for any single probe before marking the
        component as unhealthy.
    """

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self._probes: dict[str, tuple[HealthProbe, bool]] = {}
        self._timeout = timeout_seconds

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self, name: str, probe: HealthProbe, *, critical: bool = False
    ) -> None:
        """Register a health probe.

        Parameters
        ----------
        name:
            Human-readable component name (e.g. ``"postgres"``).
        probe:
            Async callable returning a ``ComponentHealth``.
        critical:
            If *True*, failure of this component drives the overall status
            to ``UNHEALTHY`` rather than ``DEGRADED``.
        """
        self._probes[name] = (probe, critical)
        logger.info("health_probe_registered", component=name, critical=critical)

    def unregister(self, name: str) -> None:
        """Remove a previously registered probe; a no-op if unknown."""
        self._probes.pop(name, None)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def check(self) -> SystemHealth:
        """Run all registered probes concurrently and aggregate results."""
        if not self._probes:
            return SystemHealth(status=HealthStatus.HEALTHY, components=[])

        tasks = {
            name: asyncio.create_task(self._run_probe(name, probe))
            for name, (probe, _) in self._probes.items()
        }

        results: list[ComponentHealth] = []
        for name, task in tasks.items():
            results.append(await task)

        status = self._compute_status(results)
        return SystemHealth(status=status, components=results)

    async def _run_probe(self, name: str, probe: HealthProbe) -> ComponentHealth:
        """Execute a single probe with timeout protection."""
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(probe(), timeout=self._timeout)
            result.latency_ms = (time.monotonic() - start) * 1000
            return result
        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - start) * 1000
            logger.warning("health_probe_timeout", component=name, elapsed_ms=elapsed)
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=elapsed,
                error=f"Probe timed out after {self._timeout}s",
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.monotonic() - start) * 1000
            logger.error("health_probe_error", component=name, error=str(exc))
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=elapsed,
                error=str(exc),
            )

    def _compute_status(self, results: list[ComponentHealth]) -> HealthStatus:
        """Derive overall status from individual component results."""
        any_unhealthy = False
        critical_unhealthy = False

        for result in results:
            if result.status == HealthStatus.UNHEALTHY:
                any_unhealthy = True
                _, is_critical = self._probes.get(result.name, (None, False))
                if is_critical:
                    critical_unhealthy = True

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY
        if any_unhealthy:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
