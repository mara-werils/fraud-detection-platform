"""Statistical significance testing for A/B experiments.

Provides z-test for proportions (flag rates) and t-test for
continuous metrics (scores, latency) with confidence intervals
and automatic winner determination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ProportionTestResult:
    """Result of a two-proportion z-test."""

    metric_name: str
    control_rate: float
    treatment_rate: float
    absolute_diff: float
    relative_diff: float
    z_score: float
    p_value: float
    is_significant: bool
    confidence_level: float
    winner: str | None  # "control", "treatment", or None


@dataclass
class ContinuousTestResult:
    """Result of a two-sample t-test approximation."""

    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_diff: float
    relative_diff: float
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    confidence_interval: tuple[float, float]
    winner: str | None


@dataclass
class ABTestSignificance:
    """Full statistical analysis of an A/B test."""

    experiment_name: str
    control_name: str
    treatment_name: str
    control_size: int
    treatment_size: int
    proportion_tests: list[ProportionTestResult]
    continuous_tests: list[ContinuousTestResult]
    overall_winner: str | None
    recommendation: str
    minimum_sample_reached: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "control": self.control_name,
            "treatment": self.treatment_name,
            "control_size": self.control_size,
            "treatment_size": self.treatment_size,
            "minimum_sample_reached": self.minimum_sample_reached,
            "overall_winner": self.overall_winner,
            "recommendation": self.recommendation,
            "proportion_tests": [
                {
                    "metric": t.metric_name,
                    "control_rate": round(t.control_rate, 6),
                    "treatment_rate": round(t.treatment_rate, 6),
                    "absolute_diff": round(t.absolute_diff, 6),
                    "relative_diff": round(t.relative_diff, 4),
                    "z_score": round(t.z_score, 4),
                    "p_value": round(t.p_value, 6),
                    "is_significant": t.is_significant,
                    "winner": t.winner,
                }
                for t in self.proportion_tests
            ],
            "continuous_tests": [
                {
                    "metric": t.metric_name,
                    "control_mean": round(t.control_mean, 4),
                    "treatment_mean": round(t.treatment_mean, 4),
                    "absolute_diff": round(t.absolute_diff, 4),
                    "relative_diff": round(t.relative_diff, 4),
                    "t_statistic": round(t.t_statistic, 4),
                    "p_value": round(t.p_value, 6),
                    "is_significant": t.is_significant,
                    "confidence_interval": (
                        round(t.confidence_interval[0], 4),
                        round(t.confidence_interval[1], 4),
                    ),
                    "winner": t.winner,
                }
                for t in self.continuous_tests
            ],
        }


class ABStatisticsCalculator:
    """Calculator for A/B test statistical significance.

    Args:
        confidence_level: Significance level (default 0.95 = 95%).
        min_sample_size: Minimum samples per group before testing.
    """

    # Z-values for common confidence levels
    Z_VALUES = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }

    def __init__(
        self,
        confidence_level: float = 0.95,
        min_sample_size: int = 100,
    ) -> None:
        self._confidence_level = confidence_level
        self._min_sample_size = min_sample_size
        self._z_critical = self.Z_VALUES.get(confidence_level, 1.960)

    def analyze(
        self,
        experiment_name: str,
        control_name: str,
        treatment_name: str,
        control_metrics: dict[str, Any],
        treatment_metrics: dict[str, Any],
    ) -> ABTestSignificance:
        """Run full statistical analysis on A/B test results.

        Args:
            experiment_name: Name of the experiment.
            control_name: Name of the control group.
            treatment_name: Name of the treatment group.
            control_metrics: Metrics dict from GroupMetrics.to_dict().
            treatment_metrics: Metrics dict from GroupMetrics.to_dict().

        Returns:
            ABTestSignificance with all test results and recommendation.
        """
        n_control = control_metrics.get("total_scored", 0)
        n_treatment = treatment_metrics.get("total_scored", 0)

        min_reached = n_control >= self._min_sample_size and n_treatment >= self._min_sample_size

        proportion_tests: list[ProportionTestResult] = []
        continuous_tests: list[ContinuousTestResult] = []

        if min_reached:
            # Test flag rate (proportion test)
            flag_test = self._proportion_test(
                metric_name="flag_rate",
                control_successes=control_metrics.get("total_flagged", 0),
                control_n=n_control,
                treatment_successes=treatment_metrics.get("total_flagged", 0),
                treatment_n=n_treatment,
                lower_is_better=True,
            )
            proportion_tests.append(flag_test)

            # Test block rate
            block_test = self._proportion_test(
                metric_name="block_rate",
                control_successes=control_metrics.get("total_blocked", 0),
                control_n=n_control,
                treatment_successes=treatment_metrics.get("total_blocked", 0),
                treatment_n=n_treatment,
                lower_is_better=True,
            )
            proportion_tests.append(block_test)

            # Test average score (continuous)
            score_test = self._continuous_test(
                metric_name="avg_score",
                control_mean=control_metrics.get("avg_score", 0),
                treatment_mean=treatment_metrics.get("avg_score", 0),
                n_control=n_control,
                n_treatment=n_treatment,
            )
            continuous_tests.append(score_test)

            # Test latency
            latency_test = self._continuous_test(
                metric_name="avg_latency_ms",
                control_mean=control_metrics.get("avg_latency_ms", 0),
                treatment_mean=treatment_metrics.get("avg_latency_ms", 0),
                n_control=n_control,
                n_treatment=n_treatment,
                lower_is_better=True,
            )
            continuous_tests.append(latency_test)

        # Determine overall winner
        overall_winner = self._determine_winner(proportion_tests, continuous_tests)
        recommendation = self._make_recommendation(
            min_reached, overall_winner, n_control, n_treatment
        )

        return ABTestSignificance(
            experiment_name=experiment_name,
            control_name=control_name,
            treatment_name=treatment_name,
            control_size=n_control,
            treatment_size=n_treatment,
            proportion_tests=proportion_tests,
            continuous_tests=continuous_tests,
            overall_winner=overall_winner,
            recommendation=recommendation,
            minimum_sample_reached=min_reached,
        )

    def _proportion_test(
        self,
        metric_name: str,
        control_successes: int,
        control_n: int,
        treatment_successes: int,
        treatment_n: int,
        lower_is_better: bool = False,
    ) -> ProportionTestResult:
        """Two-proportion z-test."""
        p1 = control_successes / control_n if control_n > 0 else 0
        p2 = treatment_successes / treatment_n if treatment_n > 0 else 0

        abs_diff = p2 - p1
        rel_diff = abs_diff / p1 if p1 > 0 else 0

        # Pooled proportion
        p_pool = (control_successes + treatment_successes) / (control_n + treatment_n)

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n)) if p_pool > 0 and p_pool < 1 else 1e-10

        z = abs_diff / se if se > 0 else 0
        p_value = self._z_to_p(abs(z))
        is_significant = p_value < (1 - self._confidence_level)

        winner = None
        if is_significant:
            if lower_is_better:
                winner = "treatment" if p2 < p1 else "control"
            else:
                winner = "treatment" if p2 > p1 else "control"

        return ProportionTestResult(
            metric_name=metric_name,
            control_rate=p1,
            treatment_rate=p2,
            absolute_diff=abs_diff,
            relative_diff=rel_diff,
            z_score=z,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self._confidence_level,
            winner=winner,
        )

    def _continuous_test(
        self,
        metric_name: str,
        control_mean: float,
        treatment_mean: float,
        n_control: int,
        n_treatment: int,
        lower_is_better: bool = False,
    ) -> ContinuousTestResult:
        """Approximate two-sample t-test using z-approximation for large N."""
        abs_diff = treatment_mean - control_mean
        rel_diff = abs_diff / control_mean if control_mean > 0 else 0

        # Estimate SE using the means as rough variance proxies
        # In production, track variance in GroupMetrics for exact calculation
        se_control = abs(control_mean) * 0.5 / math.sqrt(n_control) if n_control > 0 else 1
        se_treatment = abs(treatment_mean) * 0.5 / math.sqrt(n_treatment) if n_treatment > 0 else 1
        se = math.sqrt(se_control**2 + se_treatment**2) if (se_control + se_treatment) > 0 else 1e-10

        t_stat = abs_diff / se if se > 0 else 0
        p_value = self._z_to_p(abs(t_stat))
        is_significant = p_value < (1 - self._confidence_level)

        ci_lower = abs_diff - self._z_critical * se
        ci_upper = abs_diff + self._z_critical * se

        winner = None
        if is_significant:
            if lower_is_better:
                winner = "treatment" if treatment_mean < control_mean else "control"
            else:
                winner = "treatment" if treatment_mean > control_mean else "control"

        return ContinuousTestResult(
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_diff=abs_diff,
            relative_diff=rel_diff,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self._confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            winner=winner,
        )

    @staticmethod
    def _determine_winner(
        prop_tests: list[ProportionTestResult],
        cont_tests: list[ContinuousTestResult],
    ) -> str | None:
        """Determine overall winner based on significant results."""
        winners: dict[str, int] = {"control": 0, "treatment": 0}

        for test in prop_tests:
            if test.winner:
                winners[test.winner] += 1
        for test in cont_tests:
            if test.winner:
                winners[test.winner] += 1

        if winners["treatment"] > winners["control"]:
            return "treatment"
        elif winners["control"] > winners["treatment"]:
            return "control"
        return None

    @staticmethod
    def _make_recommendation(
        min_reached: bool,
        winner: str | None,
        n_control: int,
        n_treatment: int,
    ) -> str:
        """Generate a human-readable recommendation."""
        if not min_reached:
            needed = max(100 - n_control, 100 - n_treatment, 0)
            return f"Need {needed} more samples before statistical analysis is possible."

        if winner == "treatment":
            return "Treatment (challenger) is performing better. Consider promoting to production."
        elif winner == "control":
            return "Control is performing better. The challenger model should be revised."
        else:
            return "No statistically significant difference detected. Continue collecting data."

    @staticmethod
    def _z_to_p(z: float) -> float:
        """Approximate two-tailed p-value from z-score using rational approximation."""
        if z < 0:
            z = -z

        # Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + 0.2316419 * z)
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        poly = t * (
            0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
        p_one_tail = d * math.exp(-0.5 * z * z) * poly
        return 2 * p_one_tail
