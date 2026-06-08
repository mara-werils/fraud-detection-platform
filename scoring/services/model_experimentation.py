"""Model A/B testing and experimentation framework.

Provides traffic splitting with configurable percentages, multi-armed bandit
support (Thompson sampling, UCB1), statistical significance testing
(chi-squared, z-test), experiment lifecycle management, per-variant metric
tracking, and auto-conclusion when significance is reached.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    CONCLUDED = "concluded"


class AllocationStrategy(StrEnum):
    FIXED = "fixed"
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB1 = "ucb1"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VariantMetrics:
    """Accumulated metrics for a single experiment variant."""

    variant_id: str
    total_requests: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    total_latency_ms: float = 0.0
    total_fraud_score: float = 0.0

    # Thompson sampling priors (Beta distribution parameters)
    ts_alpha: float = 1.0
    ts_beta: float = 1.0

    # UCB1 bookkeeping
    ucb_total_reward: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def avg_fraud_score(self) -> float:
        return self.total_fraud_score / self.total_requests if self.total_requests > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "total_requests": self.total_requests,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "false_positive_rate": round(self.false_positive_rate, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "avg_fraud_score": round(self.avg_fraud_score, 6),
        }


@dataclass
class SignificanceResult:
    """Result of a statistical significance test between two variants."""

    test_name: str
    metric_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    control_value: float
    treatment_value: float
    winner: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "metric_name": self.metric_name,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "control_value": round(self.control_value, 6),
            "treatment_value": round(self.treatment_value, 6),
            "winner": self.winner,
        }


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    experiment_id: str
    name: str
    variants: dict[str, float]  # variant_id -> traffic percentage (must sum to 1.0)
    control_variant: str
    allocation_strategy: AllocationStrategy = AllocationStrategy.FIXED
    confidence_level: float = 0.95
    min_samples_per_variant: int = 100
    max_duration_hours: float | None = None
    auto_conclude: bool = True
    primary_metric: str = "precision"

    def validate(self) -> None:
        """Validate configuration constraints.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.control_variant not in self.variants:
            raise ValueError(
                f"Control variant '{self.control_variant}' not found in variants"
            )
        if len(self.variants) < 2:
            raise ValueError("At least two variants are required")
        total = sum(self.variants.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Traffic percentages must sum to 1.0, got {total:.6f}"
            )
        for vid, pct in self.variants.items():
            if pct < 0.0 or pct > 1.0:
                raise ValueError(
                    f"Traffic percentage for '{vid}' must be in [0, 1], got {pct}"
                )
        if self.confidence_level <= 0.0 or self.confidence_level >= 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.min_samples_per_variant < 1:
            raise ValueError("min_samples_per_variant must be >= 1")


@dataclass
class Experiment:
    """Runtime state for an experiment."""

    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    concluded_at: datetime | None = None
    conclusion_reason: str | None = None
    winner: str | None = None
    variant_metrics: dict[str, VariantMetrics] = field(default_factory=dict)
    significance_results: list[SignificanceResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.variant_metrics:
            for variant_id in self.config.variants:
                self.variant_metrics[variant_id] = VariantMetrics(variant_id=variant_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.config.experiment_id,
            "name": self.config.name,
            "status": self.status,
            "allocation_strategy": self.config.allocation_strategy,
            "control_variant": self.config.control_variant,
            "variants": self.config.variants,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "concluded_at": self.concluded_at.isoformat() if self.concluded_at else None,
            "conclusion_reason": self.conclusion_reason,
            "winner": self.winner,
            "variant_metrics": {
                vid: m.to_dict() for vid, m in self.variant_metrics.items()
            },
            "significance_results": [r.to_dict() for r in self.significance_results],
        }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _z_to_p(z: float) -> float:
    """Approximate two-tailed p-value from z-score (Abramowitz & Stegun)."""
    z = abs(z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989422804014327  # 1 / sqrt(2 * pi)
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    p_one_tail = d * math.exp(-0.5 * z * z) * poly
    return 2.0 * p_one_tail


def proportion_z_test(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    confidence_level: float = 0.95,
) -> tuple[float, float, bool]:
    """Two-proportion z-test.

    Returns:
        Tuple of (z_statistic, p_value, is_significant).
    """
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0, False

    p_a = successes_a / n_a
    p_b = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)

    if p_pool <= 0.0 or p_pool >= 1.0:
        return 0.0, 1.0, False

    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0.0:
        return 0.0, 1.0, False

    z = (p_a - p_b) / se
    p_value = _z_to_p(z)
    return z, p_value, p_value < (1.0 - confidence_level)


def chi_squared_test(
    observed: list[list[int]],
    confidence_level: float = 0.95,
) -> tuple[float, float, bool]:
    """Chi-squared test for a 2xK contingency table.

    Args:
        observed: List of rows, each row is a list of counts.
                  Typically [[successes_a, failures_a], [successes_b, failures_b]].
        confidence_level: Significance threshold.

    Returns:
        Tuple of (chi2_statistic, p_value, is_significant).
    """
    n_rows = len(observed)
    if n_rows < 2:
        return 0.0, 1.0, False

    n_cols = len(observed[0])
    if n_cols < 2:
        return 0.0, 1.0, False

    # Row totals, column totals, grand total
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[r][c] for r in range(n_rows)) for c in range(n_cols)]
    grand_total = sum(row_totals)

    if grand_total == 0:
        return 0.0, 1.0, False

    # Compute chi-squared statistic
    chi2 = 0.0
    for r in range(n_rows):
        for c in range(n_cols):
            expected = row_totals[r] * col_totals[c] / grand_total
            if expected > 0:
                chi2 += (observed[r][c] - expected) ** 2 / expected

    # Degrees of freedom
    df = (n_rows - 1) * (n_cols - 1)

    # Approximate p-value using Wilson-Hilferty normal approximation
    p_value = _chi2_survival(chi2, df)
    return chi2, p_value, p_value < (1.0 - confidence_level)


def _chi2_survival(x: float, df: int) -> float:
    """Approximate survival function (1 - CDF) of chi-squared distribution.

    Uses the Wilson-Hilferty transformation to approximate chi-squared
    as a standard normal variable.
    """
    if df <= 0 or x <= 0:
        return 1.0

    # Wilson-Hilferty approximation: transform chi2 to approx normal
    k = df
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))

    # One-tailed p from the normal approximation
    return _z_to_p(z) / 2.0 if z > 0 else 1.0


# ---------------------------------------------------------------------------
# Traffic allocator
# ---------------------------------------------------------------------------


class TrafficAllocator:
    """Allocates incoming requests to experiment variants.

    Supports fixed-weight splitting, Thompson sampling, and UCB1.
    """

    def __init__(self, experiment: Experiment) -> None:
        self._experiment = experiment

    def allocate(self) -> str:
        """Select a variant for the next request.

        Returns:
            The variant_id that should handle this request.
        """
        strategy = self._experiment.config.allocation_strategy

        if strategy == AllocationStrategy.FIXED:
            return self._fixed_allocate()
        elif strategy == AllocationStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling()
        elif strategy == AllocationStrategy.UCB1:
            return self._ucb1()
        else:
            return self._fixed_allocate()

    def _fixed_allocate(self) -> str:
        """Allocate based on configured traffic percentages."""
        r = random.random()
        cumulative = 0.0
        variants = list(self._experiment.config.variants.items())
        for variant_id, pct in variants:
            cumulative += pct
            if r < cumulative:
                return variant_id
        # Fallback to last variant (handles floating point edge cases)
        return variants[-1][0]

    def _thompson_sampling(self) -> str:
        """Allocate using Thompson sampling (Beta-Bernoulli model).

        Samples from each variant's Beta(alpha, beta) posterior and picks
        the variant with the highest sampled value.
        """
        best_sample = -1.0
        best_variant = ""
        for variant_id, metrics in self._experiment.variant_metrics.items():
            sample = random.betavariate(metrics.ts_alpha, metrics.ts_beta)
            if sample > best_sample:
                best_sample = sample
                best_variant = variant_id
        return best_variant

    def _ucb1(self) -> str:
        """Allocate using the UCB1 algorithm.

        For each variant, computes: mean_reward + sqrt(2 * ln(total) / n_i).
        Variants with zero observations get infinite UCB to ensure exploration.
        """
        total_requests = sum(
            m.total_requests for m in self._experiment.variant_metrics.values()
        )

        if total_requests == 0:
            # No data yet - pick uniformly at random
            return random.choice(list(self._experiment.variant_metrics.keys()))

        best_ucb = -1.0
        best_variant = ""

        for variant_id, metrics in self._experiment.variant_metrics.items():
            if metrics.total_requests == 0:
                return variant_id  # Ensure every arm gets at least one pull

            mean_reward = metrics.ucb_total_reward / metrics.total_requests
            exploration = math.sqrt(
                2.0 * math.log(total_requests) / metrics.total_requests
            )
            ucb_value = mean_reward + exploration

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_variant = variant_id

        return best_variant


# ---------------------------------------------------------------------------
# ExperimentationService
# ---------------------------------------------------------------------------


class ExperimentationService:
    """Manages the full lifecycle of model experiments.

    Handles experiment creation, starting, traffic allocation, metric
    recording, significance testing, and auto-conclusion.
    """

    # Z-values for common confidence levels
    Z_VALUES: dict[float, float] = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }

    def __init__(self) -> None:
        self._lock = Lock()
        self._experiments: dict[str, Experiment] = {}
        self._allocators: dict[str, TrafficAllocator] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """Create a new experiment in DRAFT status.

        Args:
            config: Experiment configuration.

        Returns:
            The created Experiment.

        Raises:
            ValueError: If configuration is invalid or experiment already exists.
        """
        config.validate()

        with self._lock:
            if config.experiment_id in self._experiments:
                raise ValueError(
                    f"Experiment '{config.experiment_id}' already exists"
                )

            experiment = Experiment(config=config)
            self._experiments[config.experiment_id] = experiment
            self._allocators[config.experiment_id] = TrafficAllocator(experiment)

        logger.info(
            "experiment_created",
            experiment_id=config.experiment_id,
            name=config.name,
            variants=list(config.variants.keys()),
            strategy=config.allocation_strategy,
        )

        return experiment

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Transition an experiment from DRAFT to RUNNING.

        Args:
            experiment_id: ID of the experiment to start.

        Returns:
            The updated Experiment.

        Raises:
            KeyError: If experiment not found.
            ValueError: If experiment is not in DRAFT status.
        """
        with self._lock:
            experiment = self._get_experiment(experiment_id)
            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError(
                    f"Cannot start experiment in '{experiment.status}' status"
                )

            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.now(UTC)

        logger.info("experiment_started", experiment_id=experiment_id)
        return experiment

    def conclude_experiment(
        self,
        experiment_id: str,
        winner: str | None = None,
        reason: str = "manual",
    ) -> Experiment:
        """Conclude a running experiment.

        Args:
            experiment_id: ID of the experiment to conclude.
            winner: Winning variant ID, or None if inconclusive.
            reason: Reason for conclusion.

        Returns:
            The concluded Experiment.

        Raises:
            KeyError: If experiment not found.
            ValueError: If experiment is not in RUNNING status.
        """
        with self._lock:
            experiment = self._get_experiment(experiment_id)
            if experiment.status != ExperimentStatus.RUNNING:
                raise ValueError(
                    f"Cannot conclude experiment in '{experiment.status}' status"
                )

            experiment.status = ExperimentStatus.CONCLUDED
            experiment.concluded_at = datetime.now(UTC)
            experiment.winner = winner
            experiment.conclusion_reason = reason

        logger.info(
            "experiment_concluded",
            experiment_id=experiment_id,
            winner=winner,
            reason=reason,
        )
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Retrieve an experiment by ID.

        Args:
            experiment_id: ID of the experiment.

        Returns:
            The Experiment.

        Raises:
            KeyError: If experiment not found.
        """
        with self._lock:
            return self._get_experiment(experiment_id)

    def list_experiments(
        self, status: ExperimentStatus | None = None
    ) -> list[Experiment]:
        """List experiments, optionally filtered by status.

        Args:
            status: Filter by this status, or None for all.

        Returns:
            List of matching experiments.
        """
        with self._lock:
            experiments = list(self._experiments.values())

        if status is not None:
            experiments = [e for e in experiments if e.status == status]
        return experiments

    # ------------------------------------------------------------------
    # Traffic allocation
    # ------------------------------------------------------------------

    def allocate_traffic(self, experiment_id: str) -> str:
        """Allocate traffic to a variant for the given experiment.

        Args:
            experiment_id: ID of the running experiment.

        Returns:
            The selected variant_id.

        Raises:
            KeyError: If experiment not found.
            ValueError: If experiment is not RUNNING.
        """
        with self._lock:
            experiment = self._get_experiment(experiment_id)
            if experiment.status != ExperimentStatus.RUNNING:
                raise ValueError(
                    f"Cannot allocate traffic for experiment in '{experiment.status}' status"
                )
            allocator = self._allocators[experiment_id]

        return allocator.allocate()

    # ------------------------------------------------------------------
    # Metric recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        experiment_id: str,
        variant_id: str,
        predicted_fraud: bool,
        actual_fraud: bool,
        latency_ms: float,
        fraud_score: float = 0.0,
    ) -> None:
        """Record the outcome of a single scored transaction for a variant.

        Args:
            experiment_id: ID of the experiment.
            variant_id: ID of the variant that scored the transaction.
            predicted_fraud: Whether the model predicted fraud.
            actual_fraud: Ground truth label.
            latency_ms: Time taken to score in milliseconds.
            fraud_score: Raw fraud score from the model.

        Raises:
            KeyError: If experiment or variant not found.
        """
        with self._lock:
            experiment = self._get_experiment(experiment_id)
            metrics = experiment.variant_metrics.get(variant_id)
            if metrics is None:
                raise KeyError(
                    f"Variant '{variant_id}' not found in experiment '{experiment_id}'"
                )

            metrics.total_requests += 1
            metrics.total_latency_ms += latency_ms
            metrics.total_fraud_score += fraud_score

            if predicted_fraud and actual_fraud:
                metrics.true_positives += 1
            elif predicted_fraud and not actual_fraud:
                metrics.false_positives += 1
            elif not predicted_fraud and actual_fraud:
                metrics.false_negatives += 1
            else:
                metrics.true_negatives += 1

            # Update Thompson sampling posterior
            if actual_fraud and predicted_fraud:
                metrics.ts_alpha += 1.0
            elif actual_fraud and not predicted_fraud:
                metrics.ts_beta += 1.0

            # Update UCB reward (reward = correct prediction)
            if predicted_fraud == actual_fraud:
                metrics.ucb_total_reward += 1.0

        # Check auto-conclusion outside the lock
        if experiment.config.auto_conclude and experiment.status == ExperimentStatus.RUNNING:
            self._check_auto_conclude(experiment_id)

    # ------------------------------------------------------------------
    # Significance testing
    # ------------------------------------------------------------------

    def run_significance_tests(self, experiment_id: str) -> list[SignificanceResult]:
        """Run statistical significance tests for an experiment.

        Compares each treatment variant against the control using both
        z-test (proportions) and chi-squared (contingency table) for
        precision, recall, and false positive rate.

        Args:
            experiment_id: ID of the experiment.

        Returns:
            List of SignificanceResult objects.

        Raises:
            KeyError: If experiment not found.
        """
        with self._lock:
            experiment = self._get_experiment(experiment_id)
            control_id = experiment.config.control_variant
            control_metrics = experiment.variant_metrics[control_id]
            treatment_variants = {
                vid: m
                for vid, m in experiment.variant_metrics.items()
                if vid != control_id
            }
            confidence = experiment.config.confidence_level

        results: list[SignificanceResult] = []

        for treatment_id, treatment_metrics in treatment_variants.items():
            # Skip if insufficient data
            if (
                control_metrics.total_requests < experiment.config.min_samples_per_variant
                or treatment_metrics.total_requests < experiment.config.min_samples_per_variant
            ):
                continue

            # --- Precision z-test ---
            ctrl_precision_successes = control_metrics.true_positives
            ctrl_precision_n = control_metrics.true_positives + control_metrics.false_positives
            treat_precision_successes = treatment_metrics.true_positives
            treat_precision_n = treatment_metrics.true_positives + treatment_metrics.false_positives

            if ctrl_precision_n > 0 and treat_precision_n > 0:
                z, p, sig = proportion_z_test(
                    ctrl_precision_successes, ctrl_precision_n,
                    treat_precision_successes, treat_precision_n,
                    confidence,
                )
                ctrl_val = ctrl_precision_successes / ctrl_precision_n
                treat_val = treat_precision_successes / treat_precision_n
                winner = None
                if sig:
                    winner = treatment_id if treat_val > ctrl_val else control_id
                results.append(SignificanceResult(
                    test_name="z_test",
                    metric_name="precision",
                    statistic=z,
                    p_value=p,
                    is_significant=sig,
                    confidence_level=confidence,
                    control_value=ctrl_val,
                    treatment_value=treat_val,
                    winner=winner,
                ))

            # --- Recall z-test ---
            ctrl_recall_successes = control_metrics.true_positives
            ctrl_recall_n = control_metrics.true_positives + control_metrics.false_negatives
            treat_recall_successes = treatment_metrics.true_positives
            treat_recall_n = treatment_metrics.true_positives + treatment_metrics.false_negatives

            if ctrl_recall_n > 0 and treat_recall_n > 0:
                z, p, sig = proportion_z_test(
                    ctrl_recall_successes, ctrl_recall_n,
                    treat_recall_successes, treat_recall_n,
                    confidence,
                )
                ctrl_val = ctrl_recall_successes / ctrl_recall_n
                treat_val = treat_recall_successes / treat_recall_n
                winner = None
                if sig:
                    winner = treatment_id if treat_val > ctrl_val else control_id
                results.append(SignificanceResult(
                    test_name="z_test",
                    metric_name="recall",
                    statistic=z,
                    p_value=p,
                    is_significant=sig,
                    confidence_level=confidence,
                    control_value=ctrl_val,
                    treatment_value=treat_val,
                    winner=winner,
                ))

            # --- False positive rate chi-squared ---
            ctrl_fp = control_metrics.false_positives
            ctrl_tn = control_metrics.true_negatives
            treat_fp = treatment_metrics.false_positives
            treat_tn = treatment_metrics.true_negatives

            if (ctrl_fp + ctrl_tn) > 0 and (treat_fp + treat_tn) > 0:
                observed = [
                    [ctrl_fp, ctrl_tn],
                    [treat_fp, treat_tn],
                ]
                chi2, p, sig = chi_squared_test(observed, confidence)
                ctrl_val = control_metrics.false_positive_rate
                treat_val = treatment_metrics.false_positive_rate
                winner = None
                if sig:
                    # Lower FPR is better
                    winner = treatment_id if treat_val < ctrl_val else control_id
                results.append(SignificanceResult(
                    test_name="chi_squared",
                    metric_name="false_positive_rate",
                    statistic=chi2,
                    p_value=p,
                    is_significant=sig,
                    confidence_level=confidence,
                    control_value=ctrl_val,
                    treatment_value=treat_val,
                    winner=winner,
                ))

        with self._lock:
            experiment = self._get_experiment(experiment_id)
            experiment.significance_results = results

        return results

    # ------------------------------------------------------------------
    # Auto-conclusion
    # ------------------------------------------------------------------

    def _check_auto_conclude(self, experiment_id: str) -> None:
        """Check whether an experiment should be auto-concluded.

        Auto-concludes when all treatment vs control comparisons show
        statistical significance for the primary metric.
        """
        with self._lock:
            experiment = self._experiments.get(experiment_id)
            if experiment is None or experiment.status != ExperimentStatus.RUNNING:
                return
            control_id = experiment.config.control_variant
            min_samples = experiment.config.min_samples_per_variant
            # Check minimum samples
            for metrics in experiment.variant_metrics.values():
                if metrics.total_requests < min_samples:
                    return

        # Run significance tests (acquires lock internally)
        results = self.run_significance_tests(experiment_id)
        if not results:
            return

        # Check if the primary metric shows significance for all treatments
        primary_metric = experiment.config.primary_metric
        primary_results = [r for r in results if r.metric_name == primary_metric]

        if not primary_results:
            return

        all_significant = all(r.is_significant for r in primary_results)
        if not all_significant:
            return

        # Determine overall winner from primary metric results
        winner_counts: dict[str, int] = defaultdict(int)
        for r in primary_results:
            if r.winner:
                winner_counts[r.winner] += 1

        overall_winner = None
        if winner_counts:
            overall_winner = max(winner_counts, key=winner_counts.get)  # type: ignore[arg-type]

        self.conclude_experiment(
            experiment_id,
            winner=overall_winner,
            reason="auto_significance_reached",
        )

        logger.info(
            "experiment_auto_concluded",
            experiment_id=experiment_id,
            winner=overall_winner,
            significant_results=len(primary_results),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any]:
        """Get a full summary of an experiment including metrics and tests.

        Args:
            experiment_id: ID of the experiment.

        Returns:
            Dict with full experiment state.

        Raises:
            KeyError: If experiment not found.
        """
        with self._lock:
            experiment = self._get_experiment(experiment_id)
            return experiment.to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_experiment(self, experiment_id: str) -> Experiment:
        """Retrieve experiment or raise KeyError. Must be called under lock."""
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment '{experiment_id}' not found")
        return experiment
