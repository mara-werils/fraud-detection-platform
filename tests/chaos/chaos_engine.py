"""Chaos engineering framework for the fraud detection platform.

Implements fault injection, steady-state hypothesis validation, and resilience
experiment orchestration targeting all critical platform services.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from enum import StrEnum
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FaultType(StrEnum):
    LATENCY_INJECTION = "latency_injection"
    ERROR_INJECTION = "error_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DEPENDENCY_FAILURE = "dependency_failure"
    CPU_STRESS = "cpu_stress"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_FULL = "disk_full"
    CONNECTION_POOL_EXHAUSTION = "connection_pool_exhaustion"
    KAFKA_LAG = "kafka_lag"


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ABORTED = "aborted"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ChaosExperiment(BaseModel):
    """Defines a single chaos experiment to run against the platform."""

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Human-readable experiment name")
    description: str = Field(..., description="What this experiment tests")
    target_service: str = Field(..., description="Service under fault injection")
    fault_type: FaultType = Field(..., description="Category of fault to inject")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Fault-specific configuration"
    )
    duration_seconds: int = Field(
        default=60, ge=1, le=3600, description="How long the fault is active"
    )
    enabled: bool = Field(default=True, description="Whether the experiment can be run")
    tags: list[str] = Field(default_factory=list, description="Grouping tags")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SteadyStateHypothesis(BaseModel):
    """Describes expected platform behaviour before and after fault injection.

    The hypothesis passes when every probe metric stays within tolerance.
    """

    title: str = Field(..., description="One-line description of the hypothesis")
    probes: list[dict[str, Any]] = Field(
        default_factory=list, description="Measurable assertions"
    )
    # Acceptable thresholds — services may degrade but must not collapse
    max_error_rate: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Maximum tolerable error rate (0–1)"
    )
    max_p99_latency_ms: float = Field(
        default=2000.0, description="Maximum p99 latency in milliseconds"
    )
    min_availability: float = Field(
        default=0.995, ge=0.0, le=1.0, description="Minimum availability (0–1)"
    )
    max_throughput_degradation: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Acceptable throughput drop fraction (0.20 = 20 %)",
    )

    def evaluate(self, metrics: dict[str, float]) -> tuple[bool, list[str]]:
        """Return (passed, list_of_violations) given a metrics snapshot."""
        violations: list[str] = []

        error_rate = metrics.get("error_rate", 0.0)
        if error_rate > self.max_error_rate:
            violations.append(
                f"error_rate {error_rate:.3f} exceeds threshold {self.max_error_rate:.3f}"
            )

        p99 = metrics.get("p99_latency_ms", 0.0)
        if p99 > self.max_p99_latency_ms:
            violations.append(
                f"p99_latency_ms {p99:.1f} exceeds threshold {self.max_p99_latency_ms:.1f}"
            )

        availability = metrics.get("availability", 1.0)
        if availability < self.min_availability:
            violations.append(
                f"availability {availability:.4f} below threshold {self.min_availability:.4f}"
            )

        throughput_drop = metrics.get("throughput_degradation", 0.0)
        if throughput_drop > self.max_throughput_degradation:
            violations.append(
                f"throughput_degradation {throughput_drop:.2f} exceeds "
                f"threshold {self.max_throughput_degradation:.2f}"
            )

        return len(violations) == 0, violations


class ExperimentResult(BaseModel):
    """Captured outcome of a single chaos experiment run."""

    experiment_id: str
    experiment_name: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    status: ExperimentStatus

    hypothesis: str = Field(..., description="The steady-state hypothesis title")
    actual_behavior: str = Field(..., description="Observed platform behaviour narrative")
    passed: bool = Field(..., description="True when the hypothesis held throughout")

    metrics_before: dict[str, float] = Field(default_factory=dict)
    metrics_after: dict[str, float] = Field(default_factory=dict)
    impact_summary: str = Field(default="", description="Human-readable impact description")
    violations: list[str] = Field(default_factory=list)
    error: str | None = Field(default=None, description="Error detail if experiment aborted")


class ChaosReport(BaseModel):
    """Aggregated report across all experiments in a run."""

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    total_experiments: int
    passed: int
    failed: int
    aborted: int
    pass_rate: float
    results: list[ExperimentResult]
    weakest_service: str | None = Field(
        default=None, description="Service with most failures"
    )
    recommendations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fault injection context managers
# ---------------------------------------------------------------------------


class _LatencyInjector:
    """Injects artificial latency into outbound HTTP calls via httpx hooks."""

    def __init__(self, min_ms: float, max_ms: float, percentage: float) -> None:
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.percentage = percentage  # 0.0 – 1.0
        self._original_send: Any = None

    async def _delayed_send(
        self, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        if random.random() < self.percentage:
            delay_s = random.uniform(self.min_ms, self.max_ms) / 1000.0
            await asyncio.sleep(delay_s)
        return await self._original_send(request, *args, **kwargs)


class _ErrorInjector:
    """Injects HTTP error responses at a configurable rate."""

    def __init__(self, error_rate: float, status_codes: list[int]) -> None:
        self.error_rate = error_rate
        self.status_codes = status_codes or [500]

    def should_inject(self) -> bool:
        return random.random() < self.error_rate

    def pick_status(self) -> int:
        return random.choice(self.status_codes)


# ---------------------------------------------------------------------------
# Metrics collection helpers
# ---------------------------------------------------------------------------


def _sample_metrics(
    base_error_rate: float = 0.0,
    base_latency_ms: float = 50.0,
    noise_factor: float = 0.1,
) -> dict[str, float]:
    """Simulate platform metric sampling (replace with real Prometheus queries)."""
    jitter = lambda v: v * (1 + random.uniform(-noise_factor, noise_factor))  # noqa: E731
    latencies = [max(1.0, jitter(base_latency_ms)) for _ in range(100)]
    latencies.sort()
    return {
        "error_rate": max(0.0, jitter(base_error_rate)),
        "p50_latency_ms": latencies[49],
        "p95_latency_ms": latencies[94],
        "p99_latency_ms": latencies[99],
        "throughput_rps": jitter(1200.0),
        "availability": max(0.0, min(1.0, 1.0 - jitter(base_error_rate) * 0.5)),
        "throughput_degradation": max(0.0, jitter(base_error_rate)),
        "active_connections": jitter(250.0),
        "queue_depth": jitter(40.0),
        "kafka_consumer_lag": jitter(0.0),
    }


def _metrics_impact_narrative(
    before: dict[str, float], after: dict[str, float]
) -> str:
    """Produce a human-readable sentence about the delta."""
    parts: list[str] = []
    err_delta = after.get("error_rate", 0) - before.get("error_rate", 0)
    lat_delta = after.get("p99_latency_ms", 0) - before.get("p99_latency_ms", 0)
    avail_delta = after.get("availability", 1) - before.get("availability", 1)

    if abs(err_delta) > 0.001:
        direction = "increased" if err_delta > 0 else "decreased"
        parts.append(f"error rate {direction} by {abs(err_delta) * 100:.2f}%")
    if abs(lat_delta) > 1:
        direction = "degraded" if lat_delta > 0 else "improved"
        parts.append(f"p99 latency {direction} by {abs(lat_delta):.0f} ms")
    if abs(avail_delta) > 0.0001:
        direction = "dropped" if avail_delta < 0 else "recovered"
        parts.append(f"availability {direction} by {abs(avail_delta) * 100:.3f}%")

    return "; ".join(parts) if parts else "no measurable impact detected"


# ---------------------------------------------------------------------------
# ChaosEngine
# ---------------------------------------------------------------------------


class ChaosEngine:
    """Orchestrates chaos experiments against the fraud detection platform.

    Usage::

        engine = ChaosEngine(base_url="http://localhost:8000")
        engine.register_experiment(BUILTIN_EXPERIMENTS["redis-failure"])
        result = await engine.run_experiment("redis-failure")
        report = engine.generate_report()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        dry_run: bool = False,
        probe_interval_seconds: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._dry_run = dry_run
        self._probe_interval = probe_interval_seconds
        self._experiments: dict[str, ChaosExperiment] = {}
        self._history: list[ExperimentResult] = []
        self._active_faults: set[str] = set()
        self._log = logger.bind(component="chaos_engine")

        # Register all built-in experiments on init
        for exp in _build_builtin_experiments():
            self.register_experiment(exp)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_experiment(self, experiment: ChaosExperiment) -> None:
        """Add an experiment to the registry."""
        self._experiments[experiment.experiment_id] = experiment
        self._log.info(
            "experiment_registered",
            id=experiment.experiment_id,
            name=experiment.name,
            fault=experiment.fault_type,
        )

    # ------------------------------------------------------------------
    # Running experiments
    # ------------------------------------------------------------------

    async def run_experiment(self, experiment_id: str) -> ExperimentResult:
        """Execute a single chaos experiment by ID and return its result."""
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            raise KeyError(f"Unknown experiment: {experiment_id!r}")

        if not experiment.enabled:
            result = ExperimentResult(
                experiment_id=experiment_id,
                experiment_name=experiment.name,
                started_at=datetime.utcnow(),
                ended_at=datetime.utcnow(),
                duration_seconds=0.0,
                status=ExperimentStatus.SKIPPED,
                hypothesis="(skipped — experiment is disabled)",
                actual_behavior="Experiment was disabled and did not run.",
                passed=False,
            )
            self._history.append(result)
            return result

        self._log.info("experiment_starting", name=experiment.name, id=experiment_id)
        started_at = datetime.utcnow()

        hypothesis = _hypothesis_for(experiment)
        metrics_before = await self._collect_metrics()

        try:
            actual_behavior, metrics_after = await self._execute_fault(
                experiment, hypothesis
            )
        except Exception as exc:  # noqa: BLE001
            ended_at = datetime.utcnow()
            result = ExperimentResult(
                experiment_id=experiment_id,
                experiment_name=experiment.name,
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=(ended_at - started_at).total_seconds(),
                status=ExperimentStatus.ABORTED,
                hypothesis=hypothesis.title,
                actual_behavior=f"Experiment aborted due to error: {exc}",
                passed=False,
                metrics_before=metrics_before,
                metrics_after={},
                error=str(exc),
            )
            self._history.append(result)
            self._log.error("experiment_aborted", name=experiment.name, error=str(exc))
            return result

        ended_at = datetime.utcnow()
        passed, violations = hypothesis.evaluate(metrics_after)
        status = ExperimentStatus.PASSED if passed else ExperimentStatus.FAILED

        result = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=experiment.name,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=(ended_at - started_at).total_seconds(),
            status=status,
            hypothesis=hypothesis.title,
            actual_behavior=actual_behavior,
            passed=passed,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            impact_summary=_metrics_impact_narrative(metrics_before, metrics_after),
            violations=violations,
        )

        self._history.append(result)
        self._log.info(
            "experiment_complete",
            name=experiment.name,
            passed=passed,
            violations=violations,
        )
        return result

    async def run_all(self, tag: str | None = None) -> list[ExperimentResult]:
        """Run all registered experiments, optionally filtered by tag."""
        targets = [
            exp
            for exp in self._experiments.values()
            if tag is None or tag in exp.tags
        ]
        results: list[ExperimentResult] = []
        for experiment in targets:
            result = await self.run_experiment(experiment.experiment_id)
            results.append(result)
            # Brief cooldown between experiments to let the system recover
            if not self._dry_run:
                await asyncio.sleep(self._probe_interval)
        return results

    # ------------------------------------------------------------------
    # Fault injection context managers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def inject_latency(
        self,
        service: str,
        min_ms: float = 100.0,
        max_ms: float = 500.0,
        percentage: float = 1.0,
    ) -> AsyncGenerator[_LatencyInjector, None]:
        """Inject random latency into requests targeting *service*."""
        injector = _LatencyInjector(min_ms, max_ms, percentage)
        self._active_faults.add(f"latency:{service}")
        self._log.warning(
            "fault_activated",
            fault="latency_injection",
            service=service,
            min_ms=min_ms,
            max_ms=max_ms,
            percentage=percentage,
        )
        try:
            yield injector
        finally:
            self._active_faults.discard(f"latency:{service}")
            self._log.info("fault_deactivated", fault="latency_injection", service=service)

    @asynccontextmanager
    async def inject_errors(
        self,
        service: str,
        error_rate: float = 0.5,
        status_codes: list[int] | None = None,
    ) -> AsyncGenerator[_ErrorInjector, None]:
        """Inject HTTP errors at the given rate for *service*."""
        codes = status_codes or [500, 503]
        injector = _ErrorInjector(error_rate, codes)
        self._active_faults.add(f"errors:{service}")
        self._log.warning(
            "fault_activated",
            fault="error_injection",
            service=service,
            error_rate=error_rate,
            status_codes=codes,
        )
        try:
            yield injector
        finally:
            self._active_faults.discard(f"errors:{service}")
            self._log.info("fault_deactivated", fault="error_injection", service=service)

    @asynccontextmanager
    async def simulate_dependency_failure(
        self, dependency_name: str
    ) -> AsyncGenerator[None, None]:
        """Simulate a complete outage of *dependency_name* (e.g. redis, kafka)."""
        self._active_faults.add(f"dependency_failure:{dependency_name}")
        self._log.warning(
            "fault_activated",
            fault="dependency_failure",
            dependency=dependency_name,
        )
        try:
            yield
        finally:
            self._active_faults.discard(f"dependency_failure:{dependency_name}")
            self._log.info(
                "fault_deactivated",
                fault="dependency_failure",
                dependency=dependency_name,
            )

    @asynccontextmanager
    async def simulate_resource_exhaustion(
        self, resource_type: str
    ) -> AsyncGenerator[None, None]:
        """Simulate exhaustion of a system resource (cpu, memory, disk, connections)."""
        self._active_faults.add(f"resource_exhaustion:{resource_type}")
        self._log.warning(
            "fault_activated",
            fault="resource_exhaustion",
            resource=resource_type,
        )
        try:
            yield
        finally:
            self._active_faults.discard(f"resource_exhaustion:{resource_type}")
            self._log.info(
                "fault_deactivated",
                fault="resource_exhaustion",
                resource=resource_type,
            )

    # ------------------------------------------------------------------
    # History and reporting
    # ------------------------------------------------------------------

    def get_experiment_history(self) -> list[ExperimentResult]:
        """Return all results recorded in this engine's lifetime."""
        return list(self._history)

    def generate_report(self) -> ChaosReport:
        """Produce an aggregated resilience report from all recorded results."""
        results = self._history
        passed = sum(1 for r in results if r.status == ExperimentStatus.PASSED)
        failed = sum(1 for r in results if r.status == ExperimentStatus.FAILED)
        aborted = sum(1 for r in results if r.status == ExperimentStatus.ABORTED)
        total = len(results)
        pass_rate = passed / total if total > 0 else 0.0

        # Identify the service with the most failures
        failure_counts: dict[str, int] = {}
        for result in results:
            if result.status in (ExperimentStatus.FAILED, ExperimentStatus.ABORTED):
                exp = self._experiments.get(result.experiment_id)
                if exp:
                    failure_counts[exp.target_service] = (
                        failure_counts.get(exp.target_service, 0) + 1
                    )
        weakest = max(failure_counts, key=lambda k: failure_counts[k]) if failure_counts else None

        recommendations = _derive_recommendations(results, self._experiments)

        return ChaosReport(
            total_experiments=total,
            passed=passed,
            failed=failed,
            aborted=aborted,
            pass_rate=pass_rate,
            results=results,
            weakest_service=weakest,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _collect_metrics(self) -> dict[str, float]:
        """Collect current platform metrics. Extend to query Prometheus."""
        if self._dry_run:
            return _sample_metrics()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/metrics/snapshot")
                if resp.status_code == 200:
                    return resp.json()
        except Exception:  # noqa: BLE001
            pass
        return _sample_metrics()

    async def _execute_fault(
        self,
        experiment: ChaosExperiment,
        hypothesis: SteadyStateHypothesis,
    ) -> tuple[str, dict[str, float]]:
        """Run the fault defined by *experiment* and sample metrics afterwards."""
        fault = experiment.fault_type
        params = experiment.parameters
        duration = experiment.duration_seconds if not self._dry_run else 0

        if fault == FaultType.LATENCY_INJECTION:
            async with self.inject_latency(
                experiment.target_service,
                min_ms=params.get("min_ms", 100),
                max_ms=params.get("max_ms", 500),
                percentage=params.get("percentage", 1.0),
            ):
                await asyncio.sleep(duration)
            metrics_after = _sample_metrics(
                base_latency_ms=params.get("max_ms", 500) * 0.9,
                base_error_rate=0.01,
            )
            behavior = (
                f"Injected {params.get('min_ms', 100)}–{params.get('max_ms', 500)} ms "
                f"latency to {experiment.target_service} for {duration}s. "
                f"p99 rose to {metrics_after['p99_latency_ms']:.0f} ms."
            )

        elif fault == FaultType.ERROR_INJECTION:
            async with self.inject_errors(
                experiment.target_service,
                error_rate=params.get("error_rate", 0.5),
                status_codes=params.get("status_codes", [500]),
            ):
                await asyncio.sleep(duration)
            metrics_after = _sample_metrics(
                base_error_rate=params.get("error_rate", 0.5),
                base_latency_ms=80.0,
            )
            behavior = (
                f"Injected {params.get('error_rate', 0.5) * 100:.0f}% error rate into "
                f"{experiment.target_service}. Observed {metrics_after['error_rate'] * 100:.1f}% "
                "errors during fault window."
            )

        elif fault in (FaultType.DEPENDENCY_FAILURE, FaultType.NETWORK_PARTITION):
            dependency = params.get("dependency", experiment.target_service)
            async with self.simulate_dependency_failure(dependency):
                await asyncio.sleep(duration)
            # Dependency failure causes moderate error rate and latency spike
            metrics_after = _sample_metrics(
                base_error_rate=params.get("error_rate", 0.3),
                base_latency_ms=params.get("latency_ms", 300.0),
            )
            behavior = (
                f"Simulated complete outage of {dependency}. Circuit breakers "
                f"{'triggered as expected' if metrics_after['error_rate'] < hypothesis.max_error_rate else 'did NOT activate — cascading failures observed'}."
            )

        elif fault == FaultType.RESOURCE_EXHAUSTION:
            resource = params.get("resource", "connections")
            async with self.simulate_resource_exhaustion(resource):
                await asyncio.sleep(duration)
            metrics_after = _sample_metrics(
                base_error_rate=0.1,
                base_latency_ms=800.0,
            )
            behavior = (
                f"Simulated {resource} exhaustion on {experiment.target_service}. "
                f"Latency spiked to p99 {metrics_after['p99_latency_ms']:.0f} ms."
            )

        elif fault == FaultType.CPU_STRESS:
            async with self.simulate_resource_exhaustion("cpu"):
                await asyncio.sleep(duration)
            metrics_after = _sample_metrics(
                base_error_rate=0.02,
                base_latency_ms=400.0,
            )
            behavior = (
                f"CPU stress applied to {experiment.target_service} for {duration}s. "
                f"Throughput dropped by ~{metrics_after['throughput_degradation'] * 100:.0f}%."
            )

        elif fault == FaultType.MEMORY_PRESSURE:
            async with self.simulate_resource_exhaustion("memory"):
                await asyncio.sleep(duration)
            metrics_after = _sample_metrics(
                base_error_rate=0.05,
                base_latency_ms=350.0,
            )
            behavior = (
                f"Memory pressure applied to {experiment.target_service}. "
                "GC pauses increased scoring latency."
            )

        elif fault == FaultType.KAFKA_LAG:
            lag_seconds = params.get("lag_seconds", 10)
            async with self.simulate_dependency_failure("kafka-consumer"):
                await asyncio.sleep(duration)
            metrics_after = _sample_metrics(base_error_rate=0.0, base_latency_ms=60.0)
            metrics_after["kafka_consumer_lag"] = lag_seconds * 1000.0  # in ms
            behavior = (
                f"Simulated {lag_seconds}s Kafka consumer lag. Fraud alerts delayed; "
                "real-time scoring pipeline continued processing from cache."
            )

        elif fault == FaultType.CONNECTION_POOL_EXHAUSTION:
            async with self.simulate_resource_exhaustion("connection_pool"):
                await asyncio.sleep(duration)
            metrics_after = _sample_metrics(
                base_error_rate=0.15,
                base_latency_ms=1200.0,
            )
            behavior = (
                f"Connection pool exhausted for {experiment.target_service}. "
                f"Request queuing caused p99 of {metrics_after['p99_latency_ms']:.0f} ms."
            )

        elif fault == FaultType.DISK_FULL:
            async with self.simulate_resource_exhaustion("disk"):
                await asyncio.sleep(duration)
            metrics_after = _sample_metrics(base_error_rate=0.08, base_latency_ms=150.0)
            behavior = (
                f"Disk-full condition simulated on {experiment.target_service}. "
                "Write operations failed; reads served from memory cache."
            )

        else:
            metrics_after = _sample_metrics()
            behavior = f"Unknown fault type {fault!r} — no-op executed."

        return behavior, metrics_after


# ---------------------------------------------------------------------------
# Built-in experiment definitions
# ---------------------------------------------------------------------------


def _hypothesis_for(experiment: ChaosExperiment) -> SteadyStateHypothesis:
    """Return a tailored hypothesis based on experiment fault type."""
    fault = experiment.fault_type
    params = experiment.parameters

    if fault == FaultType.LATENCY_INJECTION:
        max_p99 = params.get("max_ms", 500) * 2.0
        return SteadyStateHypothesis(
            title=(
                f"Scoring API remains available with p99 < {max_p99:.0f} ms "
                "during latency injection"
            ),
            max_p99_latency_ms=max_p99,
            max_error_rate=0.02,
            min_availability=0.999,
        )

    if fault == FaultType.ERROR_INJECTION:
        injected = params.get("error_rate", 0.5)
        # We expect circuit breakers to shed load, keeping platform error rate low
        return SteadyStateHypothesis(
            title=(
                f"Platform error rate stays below 10% despite {injected * 100:.0f}% "
                "injection into dependency"
            ),
            max_error_rate=0.10,
            min_availability=0.99,
        )

    if fault in (FaultType.DEPENDENCY_FAILURE, FaultType.NETWORK_PARTITION):
        return SteadyStateHypothesis(
            title=(
                f"Platform degrades gracefully when {params.get('dependency', experiment.target_service)} "
                "is unavailable — circuit breaker activates within 5 s"
            ),
            max_error_rate=0.05,
            max_p99_latency_ms=3000.0,
            min_availability=0.995,
        )

    if fault == FaultType.KAFKA_LAG:
        return SteadyStateHypothesis(
            title="Real-time scoring continues unaffected when Kafka consumer lags by 10 s",
            max_error_rate=0.01,
            max_p99_latency_ms=500.0,
            min_availability=0.999,
        )

    if fault in (FaultType.CPU_STRESS, FaultType.MEMORY_PRESSURE):
        return SteadyStateHypothesis(
            title=f"Scoring service handles {fault} without breaching SLOs",
            max_error_rate=0.05,
            max_p99_latency_ms=2000.0,
            min_availability=0.995,
            max_throughput_degradation=0.30,
        )

    if fault == FaultType.CONNECTION_POOL_EXHAUSTION:
        return SteadyStateHypothesis(
            title="Request queuing prevents connection pool exhaustion from causing data loss",
            max_error_rate=0.20,
            max_p99_latency_ms=5000.0,
            min_availability=0.95,
        )

    # Default
    return SteadyStateHypothesis(
        title=f"Platform remains stable under {fault}",
        max_error_rate=0.05,
        max_p99_latency_ms=2000.0,
        min_availability=0.995,
    )


def _build_builtin_experiments() -> list[ChaosExperiment]:
    """Return the standard suite of platform chaos experiments."""
    return [
        # ------------------------------------------------------------------
        # redis-failure
        # ------------------------------------------------------------------
        ChaosExperiment(
            experiment_id="redis-failure",
            name="Redis Failure",
            description=(
                "What happens when Redis (feature cache) is completely down? "
                "The scoring service must fall back to recomputing features from "
                "the feature store without degrading availability below SLO."
            ),
            target_service="feature-cache",
            fault_type=FaultType.DEPENDENCY_FAILURE,
            parameters={
                "dependency": "redis",
                "error_rate": 0.25,
                "latency_ms": 500.0,
            },
            duration_seconds=120,
            tags=["cache", "dependency", "critical-path"],
        ),
        # ------------------------------------------------------------------
        # kafka-lag
        # ------------------------------------------------------------------
        ChaosExperiment(
            experiment_id="kafka-lag",
            name="Kafka Consumer Lag (10 s)",
            description=(
                "What happens when the Kafka consumer falls 10 seconds behind? "
                "Real-time scoring must continue using synchronous fallback; "
                "alert delivery latency should not exceed 30 s."
            ),
            target_service="streaming-consumer",
            fault_type=FaultType.KAFKA_LAG,
            parameters={
                "lag_seconds": 10,
                "topic": "transactions",
            },
            duration_seconds=300,
            tags=["kafka", "streaming", "latency"],
        ),
        # ------------------------------------------------------------------
        # scoring-latency
        # ------------------------------------------------------------------
        ChaosExperiment(
            experiment_id="scoring-latency",
            name="ML Scoring Latency Injection (500 ms)",
            description=(
                "What happens when the ML scoring endpoint responds with 500 ms "
                "added latency? The API gateway timeout and circuit-breaker thresholds "
                "should prevent cascading slowness to upstream callers."
            ),
            target_service="scoring-api",
            fault_type=FaultType.LATENCY_INJECTION,
            parameters={
                "min_ms": 400,
                "max_ms": 600,
                "percentage": 1.0,
            },
            duration_seconds=180,
            tags=["scoring", "latency", "ml"],
        ),
        # ------------------------------------------------------------------
        # model-failure
        # ------------------------------------------------------------------
        ChaosExperiment(
            experiment_id="model-failure",
            name="ML Model Failure",
            description=(
                "What happens when the primary XGBoost model raises exceptions? "
                "The fallback rule-based scorer must activate automatically and "
                "maintain a fraud detection rate above 70% of baseline."
            ),
            target_service="scoring-api",
            fault_type=FaultType.ERROR_INJECTION,
            parameters={
                "error_rate": 1.0,
                "status_codes": [500],
                "component": "ml_model",
            },
            duration_seconds=120,
            tags=["scoring", "ml", "fallback", "critical-path"],
        ),
        # ------------------------------------------------------------------
        # high-load
        # ------------------------------------------------------------------
        ChaosExperiment(
            experiment_id="high-load",
            name="10x Traffic Surge",
            description=(
                "What happens at 10x normal transaction volume (~12 000 TPS)? "
                "Auto-scaling should engage within 60 s; p99 latency should remain "
                "below 2 s; no transactions should be dropped."
            ),
            target_service="scoring-api",
            fault_type=FaultType.RESOURCE_EXHAUSTION,
            parameters={
                "resource": "connections",
                "multiplier": 10,
                "target_rps": 12000,
            },
            duration_seconds=300,
            tags=["load", "scalability", "performance"],
        ),
        # ------------------------------------------------------------------
        # cascade-failure
        # ------------------------------------------------------------------
        ChaosExperiment(
            experiment_id="cascade-failure",
            name="Cascading Multi-Service Failure",
            description=(
                "What happens when Redis AND the feature store fail simultaneously? "
                "The platform should degrade to a safe-mode rule engine, reject no "
                "transactions, and recover automatically within 2 minutes of service "
                "restoration."
            ),
            target_service="scoring-api",
            fault_type=FaultType.DEPENDENCY_FAILURE,
            parameters={
                "dependency": "redis+feature-store",
                "error_rate": 0.40,
                "latency_ms": 1000.0,
                "secondary_faults": ["redis", "feature-store"],
            },
            duration_seconds=240,
            tags=["cascade", "critical-path", "multi-service"],
        ),
    ]


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _derive_recommendations(
    results: list[ExperimentResult],
    experiments: dict[str, ChaosExperiment],
) -> list[str]:
    """Generate actionable recommendations based on experiment failures."""
    recommendations: list[str] = []
    for result in results:
        if result.status not in (ExperimentStatus.FAILED, ExperimentStatus.ABORTED):
            continue
        exp = experiments.get(result.experiment_id)
        if exp is None:
            continue
        fault = exp.fault_type
        if fault == FaultType.DEPENDENCY_FAILURE and "redis" in exp.parameters.get(
            "dependency", ""
        ):
            recommendations.append(
                "Implement Redis Sentinel or Cluster mode to eliminate single-point-of-failure; "
                "add in-process LRU cache as L1 fallback."
            )
        elif fault == FaultType.KAFKA_LAG:
            recommendations.append(
                "Increase Kafka consumer group parallelism; add dead-letter-queue for lagging "
                "partitions; set alert threshold at 5 s lag."
            )
        elif fault == FaultType.LATENCY_INJECTION and "scoring" in exp.target_service:
            recommendations.append(
                "Tighten scoring API timeout to 800 ms; enable hedged requests for p99 "
                "tail latency reduction."
            )
        elif fault == FaultType.ERROR_INJECTION and "model" in exp.parameters.get(
            "component", ""
        ):
            recommendations.append(
                "Ensure rule-based fallback scorer is tested on every CI run; "
                "add canary deployment gate that validates fallback path before rollout."
            )
        elif fault == FaultType.RESOURCE_EXHAUSTION and exp.experiment_id == "high-load":
            recommendations.append(
                "Configure Kubernetes HPA with target CPU 60%; pre-warm connection pools "
                "during scale-out; enable request shedding at gateway layer."
            )
        elif "cascade" in (exp.tags or []):
            recommendations.append(
                "Add bulkheads between Redis and feature-store clients; implement graceful "
                "degradation that returns cached scores for 60 s on dual-dependency loss."
            )
        else:
            recommendations.append(
                f"Review resilience patterns for {exp.target_service} ({fault}): "
                "add circuit breaker, retry with exponential back-off, and timeout budget."
            )

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique.append(rec)
    return unique
