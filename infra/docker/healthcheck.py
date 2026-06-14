#!/usr/bin/env python3
"""Comprehensive health check system for the fraud-detection-platform.

Designed to be invoked by Docker HEALTHCHECK, Kubernetes probes, or standalone
monitoring.  Checks every critical dependency, caches results to avoid
hammering services, and detects degraded-mode operation.

Exit codes:
  0 — healthy (all dependencies reachable)
  1 — unhealthy (one or more critical dependencies unreachable)
  2 — degraded (non-critical dependency issues, service can still operate)

Environment variables consumed (all have sensible defaults):
  SCORING_HEALTH_URL        — scoring service health endpoint
  REDIS_URL                 — Redis connection string
  KAFKA_BOOTSTRAP_SERVERS   — comma-separated Kafka brokers
  DATABASE_URL              — PostgreSQL DSN (asyncpg format)
  FEATURE_STORE_URL         — feature store HTTP endpoint
  MODEL_PATH                — path to ML model artifact
  HEALTHCHECK_CACHE_TTL     — seconds to cache results (default 10)
  HEALTHCHECK_TIMEOUT       — per-check timeout in seconds (default 5)
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCORING_HEALTH_URL = os.getenv("SCORING_HEALTH_URL", "http://127.0.0.1:8000/health/live")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://fraud:fraud@localhost:5432/fraud")
FEATURE_STORE_URL = os.getenv("FEATURE_STORE_URL", "http://127.0.0.1:8001/health")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/current")
HEALTHCHECK_CACHE_TTL = int(os.getenv("HEALTHCHECK_CACHE_TTL", "10"))
HEALTHCHECK_TIMEOUT = int(os.getenv("HEALTHCHECK_TIMEOUT", "5"))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single dependency health check."""

    name: str
    status: str  # healthy | degraded | unhealthy
    latency_ms: float = 0.0
    detail: str = ""
    critical: bool = True  # if True, failure -> overall unhealthy


@dataclass
class HealthReport:
    """Aggregated health report across all dependencies."""

    status: str = "healthy"
    timestamp: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    degraded_mode: bool = False
    cached: bool = False


# ---------------------------------------------------------------------------
# Result cache — prevents hammering dependencies on rapid health checks
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}
_cache_ts: float = 0.0


def _get_cached_report() -> HealthReport | None:
    """Return the cached report if it is still fresh."""
    if _cache and (time.monotonic() - _cache_ts) < HEALTHCHECK_CACHE_TTL:
        report = HealthReport(**_cache)
        report.cached = True
        return report
    return None


def _store_cache(report: HealthReport) -> None:
    global _cache, _cache_ts
    _cache = {
        "status": report.status,
        "timestamp": report.timestamp,
        "checks": report.checks,
        "degraded_mode": report.degraded_mode,
    }
    _cache_ts = time.monotonic()


# ---------------------------------------------------------------------------
# Individual dependency checks
# ---------------------------------------------------------------------------


def _parse_redis_url(url: str) -> tuple[str, int]:
    """Extract host and port from a redis:// URL."""
    # Strip scheme
    after_scheme = url.split("://", 1)[-1]
    # Strip auth
    if "@" in after_scheme:
        after_scheme = after_scheme.split("@", 1)[-1]
    # Strip db number
    host_port = after_scheme.split("/", 1)[0]
    if ":" in host_port:
        host, port_str = host_port.rsplit(":", 1)
        return host, int(port_str)
    return host_port, 6379


def check_redis() -> CheckResult:
    """Check Redis connectivity and measure round-trip latency."""
    host, port = _parse_redis_url(REDIS_URL)
    t0 = time.monotonic()
    try:
        sock = socket.create_connection((host, port), timeout=HEALTHCHECK_TIMEOUT)
        # Send inline PING command
        sock.sendall(b"PING\r\n")
        data = sock.recv(64)
        sock.close()
        latency_ms = (time.monotonic() - t0) * 1000
        if b"PONG" in data:
            return CheckResult(
                name="redis",
                status="healthy",
                latency_ms=round(latency_ms, 2),
                detail=f"{host}:{port}",
            )
        return CheckResult(
            name="redis",
            status="unhealthy",
            latency_ms=round(latency_ms, 2),
            detail=f"unexpected reply: {data!r}",
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="redis",
            status="unhealthy",
            latency_ms=round(latency_ms, 2),
            detail=str(exc),
        )


def check_kafka() -> CheckResult:
    """Check Kafka broker connectivity by opening a TCP socket to each broker."""
    brokers = [b.strip() for b in KAFKA_BOOTSTRAP_SERVERS.split(",") if b.strip()]
    reachable = 0
    details: list[str] = []
    t0 = time.monotonic()

    for broker in brokers:
        try:
            host, port_str = broker.rsplit(":", 1)
            port = int(port_str)
            sock = socket.create_connection((host, port), timeout=HEALTHCHECK_TIMEOUT)
            sock.close()
            reachable += 1
        except Exception as exc:
            details.append(f"{broker}: {exc}")

    latency_ms = (time.monotonic() - t0) * 1000

    if reachable == len(brokers):
        return CheckResult(
            name="kafka",
            status="healthy",
            latency_ms=round(latency_ms, 2),
            detail=f"{reachable}/{len(brokers)} brokers reachable",
        )
    if reachable > 0:
        return CheckResult(
            name="kafka",
            status="degraded",
            latency_ms=round(latency_ms, 2),
            detail=f"{reachable}/{len(brokers)} brokers; {'; '.join(details)}",
            critical=False,
        )
    return CheckResult(
        name="kafka",
        status="unhealthy",
        latency_ms=round(latency_ms, 2),
        detail="; ".join(details),
    )


def _parse_pg_url(url: str) -> tuple[str, int]:
    """Extract host and port from a PostgreSQL DSN."""
    # Handle both postgresql+asyncpg:// and postgresql://
    after_scheme = url.split("://", 1)[-1]
    if "@" in after_scheme:
        after_scheme = after_scheme.split("@", 1)[-1]
    host_port = after_scheme.split("/", 1)[0]
    if ":" in host_port:
        host, port_str = host_port.rsplit(":", 1)
        return host, int(port_str)
    return host_port, 5432


def check_database() -> CheckResult:
    """Check PostgreSQL connectivity via TCP socket probe."""
    host, port = _parse_pg_url(DATABASE_URL)
    t0 = time.monotonic()
    try:
        sock = socket.create_connection((host, port), timeout=HEALTHCHECK_TIMEOUT)
        sock.close()
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="postgres",
            status="healthy",
            latency_ms=round(latency_ms, 2),
            detail=f"{host}:{port}",
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="postgres",
            status="unhealthy",
            latency_ms=round(latency_ms, 2),
            detail=str(exc),
        )


def check_model_loading() -> CheckResult:
    """Check whether the ML model artifacts are present and loadable."""
    t0 = time.monotonic()
    model_dir = Path(MODEL_PATH)

    if not model_dir.exists():
        # If there is no model directory at all, treat as degraded — the
        # service can still operate using the rule engine fallback.
        return CheckResult(
            name="ml_model",
            status="degraded",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            detail=f"model path {MODEL_PATH} does not exist (rule-engine fallback active)",
            critical=False,
        )

    # Look for common model artifacts
    artifacts = list(model_dir.glob("*.json")) + list(model_dir.glob("*.bin")) + list(
        model_dir.glob("*.pkl")
    )
    latency_ms = (time.monotonic() - t0) * 1000

    if artifacts:
        return CheckResult(
            name="ml_model",
            status="healthy",
            latency_ms=round(latency_ms, 2),
            detail=f"{len(artifacts)} artifacts in {MODEL_PATH}",
        )
    return CheckResult(
        name="ml_model",
        status="degraded",
        latency_ms=round(latency_ms, 2),
        detail=f"no model artifacts found in {MODEL_PATH} (rule-engine fallback active)",
        critical=False,
    )


def check_feature_store() -> CheckResult:
    """Check feature store availability via HTTP health endpoint."""
    t0 = time.monotonic()
    try:
        req = Request(FEATURE_STORE_URL, method="GET")
        with urlopen(req, timeout=HEALTHCHECK_TIMEOUT) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            if resp.status < 400:
                return CheckResult(
                    name="feature_store",
                    status="healthy",
                    latency_ms=round(latency_ms, 2),
                    detail=f"HTTP {resp.status}",
                    critical=False,
                )
            return CheckResult(
                name="feature_store",
                status="degraded",
                latency_ms=round(latency_ms, 2),
                detail=f"HTTP {resp.status}",
                critical=False,
            )
    except URLError as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="feature_store",
            status="degraded",
            latency_ms=round(latency_ms, 2),
            detail=str(exc.reason) if hasattr(exc, "reason") else str(exc),
            critical=False,
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="feature_store",
            status="degraded",
            latency_ms=round(latency_ms, 2),
            detail=str(exc),
            critical=False,
        )


def check_scoring_service() -> CheckResult:
    """Check the scoring service liveness endpoint via HTTP."""
    t0 = time.monotonic()
    try:
        req = Request(SCORING_HEALTH_URL, method="GET")
        with urlopen(req, timeout=HEALTHCHECK_TIMEOUT) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            if resp.status < 400:
                return CheckResult(
                    name="scoring_service",
                    status="healthy",
                    latency_ms=round(latency_ms, 2),
                    detail=f"HTTP {resp.status}",
                )
            return CheckResult(
                name="scoring_service",
                status="unhealthy",
                latency_ms=round(latency_ms, 2),
                detail=f"HTTP {resp.status}",
            )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="scoring_service",
            status="unhealthy",
            latency_ms=round(latency_ms, 2),
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Aggregation and degraded-mode detection
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_redis,
    check_kafka,
    check_database,
    check_model_loading,
    check_feature_store,
    check_scoring_service,
]


def aggregate_health(results: list[CheckResult]) -> HealthReport:
    """Aggregate individual check results into an overall health report.

    Logic:
    - If any *critical* check is unhealthy -> overall unhealthy.
    - If any check is degraded (or a non-critical check is unhealthy)
      -> overall degraded.
    - Otherwise -> healthy.
    """
    report = HealthReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        checks=[asdict(r) for r in results],
    )

    critical_unhealthy = any(r.status == "unhealthy" and r.critical for r in results)
    any_degraded = any(r.status == "degraded" for r in results)
    non_critical_unhealthy = any(r.status == "unhealthy" and not r.critical for r in results)

    if critical_unhealthy:
        report.status = "unhealthy"
    elif any_degraded or non_critical_unhealthy:
        report.status = "degraded"
        report.degraded_mode = True
    else:
        report.status = "healthy"

    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_healthcheck() -> HealthReport:
    """Execute all health checks with caching support."""
    cached = _get_cached_report()
    if cached is not None:
        return cached

    results = [check() for check in ALL_CHECKS]
    report = aggregate_health(results)
    _store_cache(report)
    return report


def main() -> int:
    """CLI entry point.  Returns process exit code."""
    report = run_healthcheck()
    print(json.dumps(asdict(report), indent=2))

    if report.status == "unhealthy":
        return 1
    if report.status == "degraded":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
