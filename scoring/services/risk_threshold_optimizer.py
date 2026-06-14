"""Risk threshold optimizer for minimizing fraud losses.

Analyses historical fraud scoring decisions to recommend optimal score
thresholds that balance false positives (legitimate transactions blocked)
against false negatives (fraudulent transactions approved).

The optimizer evaluates candidate thresholds across a configurable range
and selects the one that minimizes the expected cost based on a cost
matrix:

  - ``cost_fp``: cost per false positive (e.g. customer friction, lost revenue).
  - ``cost_fn``: cost per false negative (e.g. chargeback + investigation).
  - ``cost_tp``: reward/saving per true positive (negative cost).
  - ``cost_tn``: baseline cost for correct approvals (usually 0).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostMatrix:
    """Per-outcome costs used to evaluate a threshold."""

    cost_fp: float = 25.0    # cost per false positive
    cost_fn: float = 500.0   # cost per false negative
    cost_tp: float = -50.0   # saving per caught fraud (negative = benefit)
    cost_tn: float = 0.0     # cost per correct approval


@dataclass(frozen=True)
class ScoredTransaction:
    """A historical transaction with its score and ground-truth label."""

    score: float
    is_fraud: bool


@dataclass
class ThresholdMetrics:
    """Performance metrics at a specific threshold."""

    threshold: float
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fpr(self) -> float:
        """False positive rate."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    def total_cost(self, cost: CostMatrix) -> float:
        return (
            self.tp * cost.cost_tp
            + self.fp * cost.cost_fp
            + self.fn * cost.cost_fn
            + self.tn * cost.cost_tn
        )


@dataclass
class OptimizationResult:
    """Output of the threshold optimization process."""

    best_threshold: float
    best_cost: float
    best_metrics: ThresholdMetrics
    evaluated: list[ThresholdMetrics] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class RiskThresholdOptimizer:
    """Suggests optimal fraud score thresholds by grid search over costs.

    Parameters
    ----------
    cost_matrix:
        Defines the economic cost of each confusion-matrix outcome.
    threshold_min / threshold_max / step:
        Define the grid of candidate thresholds to evaluate.
    """

    def __init__(
        self,
        cost_matrix: CostMatrix | None = None,
        threshold_min: float = 0.0,
        threshold_max: float = 100.0,
        step: float = 1.0,
    ) -> None:
        self._cost = cost_matrix or CostMatrix()
        self._min = threshold_min
        self._max = threshold_max
        self._step = step

    def optimize(
        self, history: Sequence[ScoredTransaction]
    ) -> OptimizationResult:
        """Find the threshold that minimizes total expected cost."""
        if not history:
            raise ValueError("Cannot optimize on empty history.")

        candidates = self._generate_thresholds()
        evaluated: list[ThresholdMetrics] = []

        best_cost = float("inf")
        best_metrics: ThresholdMetrics | None = None

        for threshold in candidates:
            metrics = self._evaluate(threshold, history)
            cost = metrics.total_cost(self._cost)
            evaluated.append(metrics)

            if cost < best_cost:
                best_cost = cost
                best_metrics = metrics

        assert best_metrics is not None
        logger.info(
            "threshold_optimized",
            threshold=best_metrics.threshold,
            cost=best_cost,
            precision=round(best_metrics.precision, 4),
            recall=round(best_metrics.recall, 4),
        )

        return OptimizationResult(
            best_threshold=best_metrics.threshold,
            best_cost=best_cost,
            best_metrics=best_metrics,
            evaluated=evaluated,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_thresholds(self) -> list[float]:
        thresholds: list[float] = []
        t = self._min
        while t <= self._max:
            thresholds.append(round(t, 4))
            t += self._step
        return thresholds

    @staticmethod
    def _evaluate(
        threshold: float, history: Sequence[ScoredTransaction]
    ) -> ThresholdMetrics:
        m = ThresholdMetrics(threshold=threshold)
        for txn in history:
            flagged = txn.score >= threshold
            if flagged and txn.is_fraud:
                m.tp += 1
            elif flagged and not txn.is_fraud:
                m.fp += 1
            elif not flagged and txn.is_fraud:
                m.fn += 1
            else:
                m.tn += 1
        return m
