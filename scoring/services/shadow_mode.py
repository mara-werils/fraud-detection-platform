"""Shadow mode and canary deployment service for the fraud detection platform.

Shadow mode runs challenger models in parallel with the primary scorer without
affecting the live decision.  Canary deployment gradually shifts real traffic
to a new model and auto-promotes or rolls back based on configurable thresholds.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Lock
from typing import Any

import structlog
from prometheus_client import Gauge, Histogram
from pydantic import BaseModel, Field, field_validator

from scoring.models.ensemble import BaseScorer
from shared.schemas import FeatureVector, ScoredTransaction, Transaction

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

shadow_model_scores = Histogram(
    "shadow_model_scores",
    "Fraud scores produced by shadow models",
    labelnames=["model_name"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

shadow_model_divergence = Histogram(
    "shadow_model_divergence",
    "Absolute score divergence between shadow model and primary scorer",
    buckets=(0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
)

shadow_model_latency = Histogram(
    "shadow_model_latency",
    "Latency of shadow model scoring in milliseconds",
    labelnames=["model_name"],
    buckets=(1, 5, 10, 25, 50, 100, 200, 500, 1000),
)

canary_traffic_percentage = Gauge(
    "canary_traffic_percentage",
    "Percentage of traffic routed to the canary model",
    labelnames=["model_name"],
)

canary_error_rate = Gauge(
    "canary_error_rate",
    "Current error rate of the canary model",
    labelnames=["model_name"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ShadowModelConfig(BaseModel):
    """Configuration for a shadow model."""

    model_name: str = Field(..., description="Unique name identifying this shadow model")
    model_version: str = Field(..., description="Version string of the shadow model")
    traffic_percentage: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of transactions that will be shadow-scored (0.0–1.0)",
    )
    enabled: bool = Field(default=True, description="Whether this shadow model is active")
    compare_with_primary: bool = Field(
        default=True,
        description="Whether to compute divergence metrics against the primary score",
    )


class ShadowResult(BaseModel):
    """Result produced by a single shadow model for one transaction."""

    model_name: str
    model_version: str
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = Field(..., ge=0.0)
    error: str | None = Field(default=None, description="Error message if scoring failed")
    features_used: list[str] = Field(
        default_factory=list,
        description="Names of features consumed by this model",
    )


class ComparisonReport(BaseModel):
    """Comparison of primary vs shadow model scores for one transaction."""

    primary_score: float = Field(..., ge=0.0, le=1.0)
    shadow_scores: dict[str, ShadowResult] = Field(
        default_factory=dict,
        description="Mapping of model_name -> ShadowResult",
    )
    score_divergence: dict[str, float] = Field(
        default_factory=dict,
        description="Absolute divergence per shadow model vs primary",
    )
    agreement_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of shadows whose decision agrees with primary (BLOCK/REVIEW/ALLOW)",
    )
    latency_overhead_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Extra wall-clock time added by shadow scoring (max shadow latency)",
    )


class CanaryConfig(BaseModel):
    """Configuration for a canary deployment."""

    model_name: str = Field(..., description="Unique name for this canary model")
    traffic_percentage: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Initial fraction of live traffic routed to the canary",
    )
    auto_promote_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Agreement rate above which the canary is promoted to primary",
    )
    auto_rollback_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Agreement rate below which the canary is rolled back",
    )
    min_samples: int = Field(
        default=500,
        ge=1,
        description="Minimum number of scored samples before auto-decisions fire",
    )
    max_error_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Error rate above which the canary is immediately rolled back",
    )
    promotion_window_minutes: int = Field(
        default=60,
        ge=1,
        description="Rolling window length (minutes) used to compute agreement and error rates",
    )

    @field_validator("auto_promote_threshold")
    @classmethod
    def promote_above_rollback(cls, v: float, info: Any) -> float:
        rollback = info.data.get("auto_rollback_threshold", 0.0)
        if v <= rollback:
            raise ValueError(
                "auto_promote_threshold must be strictly greater than auto_rollback_threshold"
            )
        return v


# ---------------------------------------------------------------------------
# Internal dataclasses (not exposed via API, used for in-process bookkeeping)
# ---------------------------------------------------------------------------


class CanaryAction(StrEnum):
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    CONTINUE = "continue"


@dataclass
class _ScoredSample:
    """A single data point recorded for canary evaluation."""

    timestamp: datetime
    agreed: bool  # shadow decision == primary decision
    error: bool
    shadow_score: float
    primary_score: float


@dataclass
class CanaryEvaluation:
    """Result of evaluating a canary deployment at a point in time."""

    model_name: str
    total_samples: int
    window_samples: int
    agreement_rate: float
    error_rate: float
    avg_divergence: float
    recommended_action: CanaryAction
    traffic_percentage: float
    started_at: datetime
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "window_samples": self.window_samples,
            "agreement_rate": round(self.agreement_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "avg_divergence": round(self.avg_divergence, 4),
            "recommended_action": self.recommended_action,
            "traffic_percentage": round(self.traffic_percentage, 4),
            "started_at": self.started_at.isoformat(),
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Decision-boundary helper (mirrors EnsembleScorer.decide thresholds)
# ---------------------------------------------------------------------------


def _decision_bucket(score: float, threshold_block: float = 0.8, threshold_review: float = 0.5) -> str:
    """Map a fraud score to a decision label used for agreement comparison."""
    if score >= threshold_block:
        return "block"
    if score >= threshold_review:
        return "review"
    return "allow"


# ---------------------------------------------------------------------------
# ShadowModeService
# ---------------------------------------------------------------------------


class ShadowModeService:
    """Runs registered shadow models in parallel with the primary scorer.

    Shadow models never affect live decisions; they observe the same
    transaction and features, produce independent scores, and publish
    comparison metrics to Prometheus.

    Args:
        concurrency_limit: Maximum number of concurrent shadow scoring tasks.
    """

    def __init__(self, concurrency_limit: int = 10) -> None:
        self._lock = Lock()
        self._models: dict[str, tuple[ShadowModelConfig, BaseScorer]] = {}
        self._semaphore = asyncio.Semaphore(concurrency_limit)

        # Aggregate comparison stats (thread-safe, guarded by _lock)
        self._total_comparisons: int = 0
        self._total_agreements: int = 0
        self._divergence_sum: float = 0.0
        self._latency_overhead_sum: float = 0.0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_shadow(self, config: ShadowModelConfig, scorer: BaseScorer) -> None:
        """Register a shadow model.

        Args:
            config: Shadow model configuration.
            scorer: Instantiated scorer implementing BaseScorer.
        """
        with self._lock:
            self._models[config.model_name] = (config, scorer)

        logger.info(
            "shadow_model_registered",
            model_name=config.model_name,
            model_version=config.model_version,
            traffic_percentage=config.traffic_percentage,
        )

    def unregister_shadow(self, model_name: str) -> None:
        """Remove a previously registered shadow model.

        Args:
            model_name: Name used when the model was registered.
        """
        with self._lock:
            removed = self._models.pop(model_name, None)

        if removed:
            logger.info("shadow_model_unregistered", model_name=model_name)
        else:
            logger.warning("shadow_model_not_found", model_name=model_name)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def run_shadow_scoring(
        self,
        transaction: Transaction,
        features: FeatureVector,
    ) -> dict[str, ShadowResult]:
        """Score a transaction through all eligible shadow models concurrently.

        Eligibility is determined by each model's ``traffic_percentage`` and
        ``enabled`` flag.  Errors in individual shadow models are captured
        rather than propagated so the primary scorer is never disrupted.

        Args:
            transaction: The transaction being evaluated.
            features: Pre-computed feature vector for the transaction.

        Returns:
            Mapping of model_name -> ShadowResult for every model that ran.
        """
        with self._lock:
            candidates = [
                (cfg, scorer)
                for cfg, scorer in self._models.values()
                if cfg.enabled and self.should_sample(cfg.traffic_percentage)
            ]

        if not candidates:
            return {}

        tasks = [
            self._score_one(cfg, scorer, transaction, features)
            for cfg, scorer in candidates
        ]
        results_list: list[ShadowResult] = await asyncio.gather(*tasks)

        return {r.model_name: r for r in results_list}

    async def _score_one(
        self,
        config: ShadowModelConfig,
        scorer: BaseScorer,
        transaction: Transaction,
        features: FeatureVector,
    ) -> ShadowResult:
        """Run a single shadow model with timing, error capture, and metrics."""
        start = time.perf_counter()
        error: str | None = None
        fraud_score: float = 0.0
        features_used: list[str] = []

        async with self._semaphore:
            try:
                scored: ScoredTransaction = await scorer.score(transaction, features)
                fraud_score = scored.fraud_score
                if scored.features is not None:
                    features_used = list(scored.features.model_fields.keys())
            except Exception as exc:
                error = str(exc)
                logger.warning(
                    "shadow_scoring_error",
                    model_name=config.model_name,
                    transaction_id=str(transaction.transaction_id),
                    error=error,
                )

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Prometheus
        if error is None:
            shadow_model_scores.labels(model_name=config.model_name).observe(fraud_score)
        shadow_model_latency.labels(model_name=config.model_name).observe(latency_ms)

        return ShadowResult(
            model_name=config.model_name,
            model_version=config.model_version,
            fraud_score=fraud_score,
            latency_ms=latency_ms,
            error=error,
            features_used=features_used,
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_results(
        self,
        primary: ScoredTransaction,
        shadows: dict[str, ShadowResult],
    ) -> ComparisonReport:
        """Compare primary scorer results against shadow model results.

        Computes per-model score divergence, overall agreement rate, and
        latency overhead, then publishes divergence to Prometheus.

        Args:
            primary: The ScoredTransaction produced by the primary scorer.
            shadows: Shadow results returned by ``run_shadow_scoring``.

        Returns:
            ComparisonReport summarising all comparisons.
        """
        primary_decision = _decision_bucket(primary.fraud_score)
        divergences: dict[str, float] = {}
        agreements: list[bool] = []
        max_shadow_latency: float = 0.0

        for model_name, shadow in shadows.items():
            if shadow.error is not None:
                continue

            div = abs(primary.fraud_score - shadow.fraud_score)
            divergences[model_name] = div

            shadow_decision = _decision_bucket(shadow.fraud_score)
            agreements.append(shadow_decision == primary_decision)

            max_shadow_latency = max(max_shadow_latency, shadow.latency_ms)

            shadow_model_divergence.observe(div)

        agreement_rate = sum(agreements) / len(agreements) if agreements else 1.0

        # Update aggregate stats
        with self._lock:
            self._total_comparisons += 1
            if agreements:
                self._total_agreements += int(all(agreements))
            self._divergence_sum += sum(divergences.values()) / max(len(divergences), 1)
            self._latency_overhead_sum += max_shadow_latency

        logger.debug(
            "shadow_comparison",
            transaction_id=str(primary.transaction_id),
            primary_score=round(primary.fraud_score, 4),
            divergences={k: round(v, 4) for k, v in divergences.items()},
            agreement_rate=round(agreement_rate, 4),
        )

        return ComparisonReport(
            primary_score=primary.fraud_score,
            shadow_scores=shadows,
            score_divergence=divergences,
            agreement_rate=agreement_rate,
            latency_overhead_ms=max_shadow_latency,
        )

    # ------------------------------------------------------------------
    # Stats & utilities
    # ------------------------------------------------------------------

    def get_comparison_stats(self) -> dict[str, Any]:
        """Return aggregate comparison statistics since service start.

        Returns:
            Dict with total comparisons, overall agreement rate, mean
            divergence, mean latency overhead, and registered model names.
        """
        with self._lock:
            total = self._total_comparisons
            agreements = self._total_agreements
            div_sum = self._divergence_sum
            lat_sum = self._latency_overhead_sum
            registered = list(self._models.keys())

        return {
            "total_comparisons": total,
            "overall_agreement_rate": round(agreements / total, 4) if total else 0.0,
            "mean_divergence": round(div_sum / total, 4) if total else 0.0,
            "mean_latency_overhead_ms": round(lat_sum / total, 4) if total else 0.0,
            "registered_models": registered,
        }

    @staticmethod
    def should_sample(traffic_percentage: float) -> bool:
        """Determine probabilistically whether this request should be shadow-scored.

        Args:
            traffic_percentage: Fraction in [0.0, 1.0].

        Returns:
            True if the request should be forwarded to the shadow model.
        """
        if traffic_percentage <= 0.0:
            return False
        if traffic_percentage >= 1.0:
            return True
        return random.random() < traffic_percentage


# ---------------------------------------------------------------------------
# CanaryDeploymentService
# ---------------------------------------------------------------------------


class CanaryDeploymentService:
    """Manages canary deployments that route a portion of live traffic to a
    new model and automatically promote or roll back based on real-time metrics.

    Unlike shadow mode, the canary model's decision *may* replace the primary
    for the sampled requests (depends on the caller's routing logic).  This
    service focuses on measurement and decision-making.

    Args:
        shadow_service: The ShadowModeService used to actually run canary scoring.
    """

    def __init__(self, shadow_service: ShadowModeService | None = None) -> None:
        self._shadow = shadow_service or ShadowModeService()
        self._lock = Lock()

        # model_name -> (CanaryConfig, started_at, samples_deque)
        self._canaries: dict[str, tuple[CanaryConfig, datetime, deque[_ScoredSample]]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_canary(self, config: CanaryConfig, scorer: BaseScorer) -> None:
        """Register and start a canary deployment.

        The canary is wired into the underlying ShadowModeService so that
        ``run_shadow_scoring`` automatically sends the configured traffic
        fraction to this model.

        Args:
            config: Canary configuration.
            scorer: Instantiated scorer for the canary model.
        """
        shadow_cfg = ShadowModelConfig(
            model_name=config.model_name,
            model_version=getattr(scorer, "model_version", "canary"),
            traffic_percentage=config.traffic_percentage,
            enabled=True,
            compare_with_primary=True,
        )
        self._shadow.register_shadow(shadow_cfg, scorer)

        window_size = config.min_samples * 10  # generous ring buffer
        with self._lock:
            self._canaries[config.model_name] = (
                config,
                datetime.now(UTC),
                deque(maxlen=window_size),
            )

        canary_traffic_percentage.labels(model_name=config.model_name).set(
            config.traffic_percentage
        )
        canary_error_rate.labels(model_name=config.model_name).set(0.0)

        logger.info(
            "canary_started",
            model_name=config.model_name,
            traffic_percentage=config.traffic_percentage,
            auto_promote_threshold=config.auto_promote_threshold,
            auto_rollback_threshold=config.auto_rollback_threshold,
        )

    def stop_canary(self, model_name: str) -> None:
        """Stop and remove a canary deployment.

        Args:
            model_name: Name of the canary to stop.
        """
        self._shadow.unregister_shadow(model_name)

        with self._lock:
            removed = self._canaries.pop(model_name, None)

        if removed:
            canary_traffic_percentage.labels(model_name=model_name).set(0.0)
            canary_error_rate.labels(model_name=model_name).set(0.0)
            logger.info("canary_stopped", model_name=model_name)
        else:
            logger.warning("canary_not_found_on_stop", model_name=model_name)

    # ------------------------------------------------------------------
    # Recording results
    # ------------------------------------------------------------------

    def record_result(
        self,
        model_name: str,
        primary: ScoredTransaction,
        shadow: ShadowResult,
    ) -> None:
        """Append one scored sample to the canary's rolling window.

        Should be called after ``ShadowModeService.run_shadow_scoring`` for
        every canary result so that ``evaluate_canary`` has data to work with.

        Args:
            model_name: Canary model name.
            primary: Primary scorer's ScoredTransaction.
            shadow: Shadow/canary ShadowResult for the same transaction.
        """
        with self._lock:
            entry = self._canaries.get(model_name)
            if entry is None:
                return
            config, started_at, samples = entry

        primary_decision = _decision_bucket(primary.fraud_score)
        shadow_decision = _decision_bucket(shadow.fraud_score) if shadow.error is None else None
        agreed = shadow_decision == primary_decision if shadow_decision is not None else False
        is_error = shadow.error is not None

        sample = _ScoredSample(
            timestamp=datetime.now(UTC),
            agreed=agreed,
            error=is_error,
            shadow_score=shadow.fraud_score,
            primary_score=primary.fraud_score,
        )

        with self._lock:
            samples.append(sample)

        # Update Prometheus error rate gauge
        window_samples = self._window_samples(samples, config.promotion_window_minutes)
        if window_samples:
            err_rate = sum(1 for s in window_samples if s.error) / len(window_samples)
            canary_error_rate.labels(model_name=model_name).set(err_rate)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_canary(self, model_name: str) -> CanaryEvaluation:
        """Evaluate a running canary and recommend an action.

        Computes agreement rate and error rate over the configured
        ``promotion_window_minutes`` and applies the configured thresholds to
        recommend PROMOTE, ROLLBACK, or CONTINUE.

        Args:
            model_name: Canary model name.

        Returns:
            CanaryEvaluation with metrics and a recommended action.

        Raises:
            KeyError: If no canary with that name is registered.
        """
        with self._lock:
            entry = self._canaries.get(model_name)

        if entry is None:
            raise KeyError(f"No canary registered with name '{model_name}'")

        config, started_at, samples = entry
        window_samples = self._window_samples(samples, config.promotion_window_minutes)
        total_samples = len(samples)
        window_n = len(window_samples)

        if window_n == 0:
            return CanaryEvaluation(
                model_name=model_name,
                total_samples=total_samples,
                window_samples=0,
                agreement_rate=0.0,
                error_rate=0.0,
                avg_divergence=0.0,
                recommended_action=CanaryAction.CONTINUE,
                traffic_percentage=config.traffic_percentage,
                started_at=started_at,
            )

        error_count = sum(1 for s in window_samples if s.error)
        error_rate = error_count / window_n

        valid = [s for s in window_samples if not s.error]
        agreement_rate = sum(1 for s in valid if s.agreed) / len(valid) if valid else 0.0
        avg_divergence = (
            sum(abs(s.shadow_score - s.primary_score) for s in valid) / len(valid)
            if valid else 0.0
        )

        # Determine recommended action
        if window_n < config.min_samples:
            action = CanaryAction.CONTINUE
        elif error_rate > config.max_error_rate:
            action = CanaryAction.ROLLBACK
            logger.warning(
                "canary_error_rate_exceeded",
                model_name=model_name,
                error_rate=round(error_rate, 4),
                max_error_rate=config.max_error_rate,
            )
        elif agreement_rate >= config.auto_promote_threshold:
            action = CanaryAction.PROMOTE
        elif agreement_rate <= config.auto_rollback_threshold:
            action = CanaryAction.ROLLBACK
        else:
            action = CanaryAction.CONTINUE

        logger.info(
            "canary_evaluated",
            model_name=model_name,
            window_n=window_n,
            agreement_rate=round(agreement_rate, 4),
            error_rate=round(error_rate, 4),
            avg_divergence=round(avg_divergence, 4),
            action=action,
        )

        return CanaryEvaluation(
            model_name=model_name,
            total_samples=total_samples,
            window_samples=window_n,
            agreement_rate=agreement_rate,
            error_rate=error_rate,
            avg_divergence=avg_divergence,
            recommended_action=action,
            traffic_percentage=config.traffic_percentage,
            started_at=started_at,
        )

    def auto_evaluate(self) -> list[tuple[str, CanaryAction]]:
        """Evaluate all running canaries and act on PROMOTE / ROLLBACK decisions.

        For PROMOTE: logs and stops the canary (the caller is responsible for
        swapping the primary scorer).  For ROLLBACK: stops the canary
        immediately.  CONTINUE entries are included in the return value for
        observability.

        Returns:
            List of (model_name, CanaryAction) tuples for all canaries.
        """
        with self._lock:
            names = list(self._canaries.keys())

        actions: list[tuple[str, CanaryAction]] = []

        for name in names:
            try:
                evaluation = self.evaluate_canary(name)
            except KeyError:
                continue  # canary was stopped concurrently

            actions.append((name, evaluation.recommended_action))

            if evaluation.recommended_action == CanaryAction.PROMOTE:
                logger.info(
                    "canary_auto_promote",
                    model_name=name,
                    agreement_rate=round(evaluation.agreement_rate, 4),
                    window_samples=evaluation.window_samples,
                )
                self.stop_canary(name)

            elif evaluation.recommended_action == CanaryAction.ROLLBACK:
                logger.warning(
                    "canary_auto_rollback",
                    model_name=name,
                    agreement_rate=round(evaluation.agreement_rate, 4),
                    error_rate=round(evaluation.error_rate, 4),
                    window_samples=evaluation.window_samples,
                )
                self.stop_canary(name)

        return actions

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_canary_status(self) -> dict[str, Any]:
        """Return a snapshot of all running canaries.

        Returns:
            Dict mapping model_name to its current evaluation dict.
        """
        with self._lock:
            names = list(self._canaries.keys())

        status: dict[str, Any] = {}
        for name in names:
            try:
                ev = self.evaluate_canary(name)
                status[name] = ev.to_dict()
            except KeyError:
                pass  # stopped between list and evaluate

        return status

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _window_samples(
        samples: deque[_ScoredSample],
        window_minutes: int,
    ) -> list[_ScoredSample]:
        """Return samples within the rolling window.

        Args:
            samples: Full sample deque for a canary.
            window_minutes: Rolling window duration in minutes.

        Returns:
            Subset of samples whose timestamp falls within the window.
        """
        cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
        return [s for s in samples if s.timestamp >= cutoff]
