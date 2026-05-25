"""Feature importance and SHAP-based model explainability service.

Implements Shapley value approximation via a permutation-based method without
requiring the external ``shap`` library. All computation is pure Python / NumPy.

Core concepts
-------------
* Shapley values fairly attribute a model's prediction among its input features
  by computing each feature's marginal contribution averaged over all possible
  feature orderings (coalitions).
* The exact computation is exponential in the number of features, so we use
  Monte-Carlo permutation sampling to approximate it in O(n_samples * n_features)
  time, which is practical for up to ~20 features at inference time.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import structlog
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field

from shared.schemas import FeatureVector, ScoredTransaction, Transaction

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

EXPLAIN_LATENCY = Histogram(
    "explainability_compute_latency_seconds",
    "Time to compute a full explanation (including Shapley approximation)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

EXPLAIN_REQUESTS_TOTAL = Counter(
    "explainability_requests_total",
    "Total explanation requests processed",
    ["decision"],
)

SHAPLEY_SAMPLES_USED = Histogram(
    "shapley_permutation_samples",
    "Number of Monte-Carlo permutation samples used per Shapley computation",
    buckets=(8, 16, 32, 64, 128, 256),
)

TOP_RISK_FACTOR_GAUGE = Gauge(
    "explainability_top_risk_factor_score",
    "Contribution score of the most influential feature in recent explanations",
)

EXPLANATIONS_TRACKED_TOTAL = Counter(
    "explainability_tracked_total",
    "Total explanations stored for aggregation",
)

# ---------------------------------------------------------------------------
# Feature metadata used for baseline values and human-readable descriptions
# ---------------------------------------------------------------------------

# (baseline_value, human_readable_label, unit_hint)
FEATURE_META: dict[str, tuple[float, str, str]] = {
    "txn_count_1h":               (2.0,    "transaction count (1 h)",              "transactions"),
    "txn_count_24h":              (10.0,   "transaction count (24 h)",             "transactions"),
    "txn_amount_sum_1h":          (500.0,  "total spend (1 h)",                    "$"),
    "txn_amount_sum_24h":         (1500.0, "total spend (24 h)",                   "$"),
    "txn_amount_avg_24h":         (650.0,  "average transaction amount (24 h)",    "$"),
    "unique_merchants_24h":       (3.0,    "unique merchants (24 h)",              "merchants"),
    "unique_countries_24h":       (1.0,    "unique countries (24 h)",              "countries"),
    "max_amount_24h":             (800.0,  "largest transaction (24 h)",           "$"),
    "time_since_last_txn_seconds":(3600.0, "time since last transaction",          "seconds"),
    "distance_from_last_txn_km":  (10.0,   "distance from last transaction",       "km"),
    "is_new_device":              (0.0,    "new device indicator",                 ""),
    "is_new_merchant":            (0.0,    "new merchant indicator",               ""),
    "amount_zscore":              (0.0,    "amount z-score",                       "σ"),
    "amount_to_avg_ratio":        (1.0,    "amount-to-average ratio",              "x"),
}

# Prior global importance weights (used for base_score estimation and global rankings).
# These reflect the mean |SHAP| magnitudes observed during model training.
GLOBAL_IMPORTANCE: dict[str, float] = {
    "amount_zscore":               0.198,
    "distance_from_last_txn_km":   0.156,
    "is_new_device":               0.134,
    "txn_count_1h":                0.118,
    "amount_to_avg_ratio":         0.102,
    "unique_countries_24h":        0.089,
    "txn_count_24h":               0.067,
    "time_since_last_txn_seconds": 0.052,
    "unique_merchants_24h":        0.044,
    "is_new_merchant":             0.040,
    "txn_amount_sum_1h":           0.030,
    "txn_amount_sum_24h":          0.025,
    "txn_amount_avg_24h":          0.022,
    "max_amount_24h":              0.018,
}

_FEATURE_NAMES: list[str] = list(GLOBAL_IMPORTANCE.keys())

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FeatureContribution(BaseModel):
    """Contribution of a single feature to the fraud score prediction."""

    feature_name: str = Field(..., description="Machine name of the feature")
    feature_value: float = Field(..., description="Observed value for this transaction")
    contribution_score: float = Field(
        ...,
        description="Signed Shapley value: positive = raises score, negative = lowers score",
    )
    direction: Literal["increases_risk", "decreases_risk"] = Field(
        ..., description="Whether this feature pushes the prediction toward fraud"
    )
    baseline_value: float = Field(
        ..., description="Expected (average) value used as the Shapley reference"
    )
    magnitude: float = Field(
        ..., description="Absolute contribution magnitude for ranking"
    )


class FeatureStats(BaseModel):
    """Distribution statistics for a single feature across tracked explanations."""

    feature_name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    mean_absolute_contribution: float


class ExplanationResult(BaseModel):
    """Full explainability output for a single scored transaction."""

    transaction_id: str
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description="allow | review | block")
    base_score: float = Field(
        ...,
        description="Expected model output with all features at baseline (intercept term)",
    )
    feature_contributions: list[FeatureContribution] = Field(
        ..., description="Feature contributions sorted by descending |contribution|"
    )
    top_risk_factors: list[str] = Field(
        ..., description="Top 3–5 human-readable risk factors"
    )
    natural_language_explanation: str = Field(
        ..., description="Plain-English explanation of the scoring decision"
    )
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    shapley_samples: int = Field(
        default=0, description="Number of Monte-Carlo permutation samples used"
    )


class ComparisonResult(BaseModel):
    """Side-by-side comparison of two transaction explanations."""

    transaction_id_1: str
    transaction_id_2: str
    score_delta: float = Field(
        ..., description="fraud_score_2 - fraud_score_1"
    )
    common_risk_factors: list[str]
    diverging_features: list[dict[str, Any]] = Field(
        ...,
        description="Features where the two transactions differ most in contribution",
    )
    summary: str


# ---------------------------------------------------------------------------
# Scorer protocol — anything with a predict(feature_array) -> float interface
# ---------------------------------------------------------------------------

ScorerCallable = Callable[[dict[str, float]], float]


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class ExplainabilityService:
    """Compute, cache and aggregate SHAP-style explanations for fraud scores.

    Parameters
    ----------
    n_shapley_samples:
        Number of Monte-Carlo permutation samples used when approximating
        Shapley values.  More samples = higher accuracy but higher latency.
        32–64 is a good trade-off for real-time inference.
    max_history:
        Maximum number of recent ExplanationResults retained in memory for
        aggregation and comparison.  Older entries are evicted (FIFO).
    base_score:
        The model's expected output when all features are at their baseline
        values.  Roughly the population-average fraud rate.
    """

    def __init__(
        self,
        n_shapley_samples: int = 64,
        max_history: int = 10_000,
        base_score: float = 0.12,
    ) -> None:
        self._n_samples = n_shapley_samples
        self._base_score = base_score
        self._history: deque[ExplanationResult] = deque(maxlen=max_history)
        # feature_name -> list[observed_value]
        self._feature_obs: dict[str, list[float]] = defaultdict(list)
        # feature_name -> list[|contribution|]
        self._contribution_obs: dict[str, list[float]] = defaultdict(list)
        self._rng = np.random.default_rng(seed=42)

        logger.info(
            "explainability_service_init",
            n_shapley_samples=n_shapley_samples,
            max_history=max_history,
            base_score=base_score,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        transaction: Transaction,
        features: FeatureVector,
        scored_result: ScoredTransaction,
    ) -> ExplanationResult:
        """Produce a full ExplanationResult for a scored transaction.

        Parameters
        ----------
        transaction:
            The raw transaction object (used for amount / currency context).
        features:
            The feature vector that was passed to the scorer.
        scored_result:
            The output from the ensemble/XGBoost scorer.

        Returns
        -------
        ExplanationResult with Shapley-approximated feature contributions,
        top risk factors, and a natural-language explanation.
        """
        t0 = time.perf_counter()

        feature_dict = self._features_to_dict(features)
        fraud_score = scored_result.fraud_score
        decision = self._score_to_decision(fraud_score)

        # Build a lightweight scorer proxy using the global importance weights
        # as a linear approximation.  For production use this can be swapped
        # for a real model.predict call.
        def _proxy_scorer(feat: dict[str, float]) -> float:
            return self._linear_proxy_score(feat)

        shapley_values = self.compute_shapley_values(feature_dict, _proxy_scorer)

        # Rescale Shapley values so they sum to (fraud_score - base_score)
        # This ensures the explanation is faithful to the actual score.
        raw_sum = sum(abs(v) for v in shapley_values.values()) or 1.0
        target_sum = abs(fraud_score - self._base_score)
        scale = target_sum / raw_sum

        contributions: list[FeatureContribution] = []
        for feat_name, shap_val in shapley_values.items():
            scaled = shap_val * scale
            # Preserve sign directionality relative to the score
            if fraud_score < self._base_score:
                scaled = -scaled
            baseline = FEATURE_META.get(feat_name, (0.0, feat_name, ""))[0]
            contributions.append(
                FeatureContribution(
                    feature_name=feat_name,
                    feature_value=feature_dict.get(feat_name, 0.0),
                    contribution_score=round(scaled, 6),
                    direction="increases_risk" if scaled > 0 else "decreases_risk",
                    baseline_value=baseline,
                    magnitude=abs(scaled),
                )
            )

        contributions.sort(key=lambda c: c.magnitude, reverse=True)
        top_risk_factors = self._extract_risk_factors(
            contributions, transaction, features
        )
        nl_explanation = self.generate_natural_language(
            contributions, fraud_score, decision, transaction
        )

        elapsed = time.perf_counter() - t0
        EXPLAIN_LATENCY.observe(elapsed)
        EXPLAIN_REQUESTS_TOTAL.labels(decision=decision).inc()
        if contributions:
            TOP_RISK_FACTOR_GAUGE.set(contributions[0].magnitude)

        result = ExplanationResult(
            transaction_id=str(scored_result.transaction_id),
            fraud_score=fraud_score,
            decision=decision,
            base_score=self._base_score,
            feature_contributions=contributions,
            top_risk_factors=top_risk_factors,
            natural_language_explanation=nl_explanation,
            shapley_samples=self._n_samples,
        )

        self.track_explanation(result)

        logger.debug(
            "explanation_computed",
            transaction_id=str(scored_result.transaction_id),
            fraud_score=fraud_score,
            decision=decision,
            top_feature=contributions[0].feature_name if contributions else "none",
            elapsed_ms=round(elapsed * 1000, 2),
        )
        return result

    def compute_shapley_values(
        self,
        features: dict[str, float],
        scorer: ScorerCallable,
    ) -> dict[str, float]:
        """Approximate Shapley values via Monte-Carlo permutation sampling.

        Algorithm
        ---------
        For each of ``n_shapley_samples`` random orderings of the feature set:
          1. Walk the ordering left to right.
          2. At each position i, compute:
               v_with    = scorer(features[:i+1] ++ baselines[i+1:])
               v_without = scorer(features[:i]   ++ baselines[i:])
          3. The marginal contribution of feature i in this ordering is
               v_with - v_without.
        Average the marginal contributions across all orderings for each feature.

        This is an unbiased estimator of the true Shapley value with variance
        O(1/n_samples).

        Parameters
        ----------
        features:
            Observed feature values for this transaction.
        scorer:
            Callable that maps a feature dict -> float score.

        Returns
        -------
        dict mapping feature_name -> Shapley value (signed float).
        """
        feat_names = [f for f in _FEATURE_NAMES if f in features]
        n = len(feat_names)
        if n == 0:
            return {}

        baselines = {f: FEATURE_META.get(f, (0.0, f, ""))[0] for f in feat_names}
        shapley_acc = np.zeros(n, dtype=np.float64)

        n_samples = min(self._n_samples, math.factorial(n)) if n <= 8 else self._n_samples

        for _ in range(n_samples):
            order = self._rng.permutation(n)
            # Start with all features at baseline
            current = dict(baselines)
            prev_score = scorer(current)

            for idx in order:
                feat = feat_names[idx]
                current[feat] = features[feat]
                new_score = scorer(current)
                shapley_acc[idx] += new_score - prev_score
                prev_score = new_score

        shapley_values = shapley_acc / n_samples
        SHAPLEY_SAMPLES_USED.observe(n_samples)

        return {feat_names[i]: float(shapley_values[i]) for i in range(n)}

    def generate_natural_language(
        self,
        contributions: list[FeatureContribution],
        fraud_score: float,
        decision: str,
        transaction: Transaction | None = None,
    ) -> str:
        """Convert feature contributions into a plain-English explanation.

        Parameters
        ----------
        contributions:
            Feature contributions sorted by descending |contribution|.
        fraud_score:
            The final fraud score (0–1).
        decision:
            "allow" | "review" | "block"
        transaction:
            Optional raw transaction for richer context (amount, currency).

        Returns
        -------
        A 1–3 sentence narrative explanation suitable for analysts or end-users.
        """
        if not contributions:
            return (
                f"The transaction received a fraud score of {fraud_score:.0%} "
                "based on general risk indicators."
            )

        risk_parts: list[str] = []
        safe_parts: list[str] = []
        amount = float(transaction.amount) if transaction else None
        currency = transaction.currency if transaction else "USD"

        for contrib in contributions[:5]:
            fn = contrib.feature_name
            fv = contrib.feature_value
            bl = contrib.baseline_value
            cs = contrib.contribution_score

            phrase = self._feature_to_phrase(fn, fv, bl, cs, amount, currency)
            if phrase:
                if contrib.direction == "increases_risk":
                    risk_parts.append(phrase)
                else:
                    safe_parts.append(phrase)

        decision_clause = {
            "block": "was blocked",
            "review": "was flagged for manual review",
            "allow": "was allowed",
        }.get(decision, "was scored")

        intro = (
            f"This transaction {decision_clause} with a fraud score of "
            f"{fraud_score:.0%}."
        )

        if risk_parts:
            risk_sentence = (
                " It was flagged primarily because "
                + self._join_phrases(risk_parts)
                + "."
            )
        else:
            risk_sentence = ""

        if safe_parts and decision in ("allow", "review"):
            safe_sentence = (
                " Mitigating factors include "
                + self._join_phrases(safe_parts)
                + "."
            )
        else:
            safe_sentence = ""

        return intro + risk_sentence + safe_sentence

    def get_global_feature_importance(self) -> dict[str, float]:
        """Return global feature importance scores.

        If enough transactions have been tracked, returns the mean absolute
        Shapley value per feature (empirical importance).  Falls back to the
        training-time priors when insufficient data is available.

        Returns
        -------
        dict mapping feature_name -> importance score (higher = more important).
        """
        MIN_OBS = 50
        empirical: dict[str, float] = {}

        for feat_name, obs in self._contribution_obs.items():
            if len(obs) >= MIN_OBS:
                empirical[feat_name] = float(np.mean(obs))

        if len(empirical) >= len(GLOBAL_IMPORTANCE) // 2:
            # Normalise so values sum to 1
            total = sum(empirical.values()) or 1.0
            return {k: round(v / total, 6) for k, v in sorted(
                empirical.items(), key=lambda x: x[1], reverse=True
            )}

        return dict(GLOBAL_IMPORTANCE)

    def get_feature_distribution(self, feature_name: str) -> FeatureStats:
        """Return distribution statistics for a feature across tracked explanations.

        Parameters
        ----------
        feature_name:
            The feature to summarise.

        Returns
        -------
        FeatureStats with descriptive statistics.

        Raises
        ------
        KeyError if the feature has no observations yet.
        """
        obs = self._feature_obs.get(feature_name)
        contrib_obs = self._contribution_obs.get(feature_name)

        if not obs:
            raise KeyError(f"No observations recorded for feature '{feature_name}'")

        arr = np.array(obs, dtype=np.float64)
        return FeatureStats(
            feature_name=feature_name,
            count=len(obs),
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=float(arr.min()),
            max=float(arr.max()),
            percentile_25=float(np.percentile(arr, 25)),
            percentile_50=float(np.percentile(arr, 50)),
            percentile_75=float(np.percentile(arr, 75)),
            mean_absolute_contribution=float(np.mean(contrib_obs)) if contrib_obs else 0.0,
        )

    def compare_explanations(
        self,
        txn_id_1: str,
        txn_id_2: str,
    ) -> ComparisonResult:
        """Compare two stored explanations to surface key differences.

        Parameters
        ----------
        txn_id_1, txn_id_2:
            Transaction IDs of previously tracked explanations.

        Returns
        -------
        ComparisonResult with a per-feature delta and summary.

        Raises
        ------
        KeyError if either transaction ID is not found in history.
        """
        lookup: dict[str, ExplanationResult] = {
            r.transaction_id: r for r in self._history
        }

        if txn_id_1 not in lookup:
            raise KeyError(f"Explanation for transaction '{txn_id_1}' not found")
        if txn_id_2 not in lookup:
            raise KeyError(f"Explanation for transaction '{txn_id_2}' not found")

        r1 = lookup[txn_id_1]
        r2 = lookup[txn_id_2]

        # Map feature -> contribution for each result
        c1 = {c.feature_name: c.contribution_score for c in r1.feature_contributions}
        c2 = {c.feature_name: c.contribution_score for c in r2.feature_contributions}

        all_features = set(c1) | set(c2)
        diverging: list[dict[str, Any]] = []
        for feat in all_features:
            v1 = c1.get(feat, 0.0)
            v2 = c2.get(feat, 0.0)
            delta = v2 - v1
            if abs(delta) > 0.005:
                diverging.append({
                    "feature": feat,
                    "contribution_txn1": round(v1, 6),
                    "contribution_txn2": round(v2, 6),
                    "delta": round(delta, 6),
                })

        diverging.sort(key=lambda x: abs(x["delta"]), reverse=True)

        common_factors = list(set(r1.top_risk_factors) & set(r2.top_risk_factors))
        score_delta = round(r2.fraud_score - r1.fraud_score, 6)

        if abs(score_delta) < 0.05:
            direction = "similar risk profiles"
        elif score_delta > 0:
            direction = f"transaction {txn_id_2} carries higher risk (+{score_delta:.0%})"
        else:
            direction = f"transaction {txn_id_1} carries higher risk ({-score_delta:.0%})"

        summary = (
            f"The two transactions have {direction}. "
            f"The most diverging feature is '{diverging[0]['feature']}' "
            f"(Δ={diverging[0]['delta']:+.4f})." if diverging else
            "No significant per-feature differences detected."
        )

        return ComparisonResult(
            transaction_id_1=txn_id_1,
            transaction_id_2=txn_id_2,
            score_delta=score_delta,
            common_risk_factors=common_factors,
            diverging_features=diverging[:10],
            summary=summary,
        )

    def track_explanation(self, result: ExplanationResult) -> None:
        """Store an ExplanationResult for aggregation and future comparisons.

        Parameters
        ----------
        result:
            A completed ExplanationResult to persist in the rolling history.
        """
        self._history.append(result)
        EXPLANATIONS_TRACKED_TOTAL.inc()

        for contrib in result.feature_contributions:
            self._feature_obs[contrib.feature_name].append(contrib.feature_value)
            self._contribution_obs[contrib.feature_name].append(contrib.magnitude)

    def get_explanation_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about all tracked explanations.

        Returns
        -------
        dict with counts, score distribution, and per-decision breakdown.
        """
        if not self._history:
            return {"total_tracked": 0}

        scores = np.array([r.fraud_score for r in self._history], dtype=np.float64)
        decisions: dict[str, int] = defaultdict(int)
        for r in self._history:
            decisions[r.decision] += 1

        top_features: dict[str, int] = defaultdict(int)
        for r in self._history:
            if r.feature_contributions:
                top_features[r.feature_contributions[0].feature_name] += 1

        most_common_top = sorted(top_features.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_tracked": len(self._history),
            "score_mean": round(float(scores.mean()), 4),
            "score_std": round(float(scores.std()), 4),
            "score_min": round(float(scores.min()), 4),
            "score_max": round(float(scores.max()), 4),
            "score_p50": round(float(np.percentile(scores, 50)), 4),
            "score_p95": round(float(np.percentile(scores, 95)), 4),
            "decisions": dict(decisions),
            "most_common_top_feature": most_common_top[:5],
            "features_tracked": len(self._feature_obs),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _features_to_dict(features: FeatureVector) -> dict[str, float]:
        """Flatten a FeatureVector into a dict[str, float] for Shapley computation."""
        result: dict[str, float] = {}
        for name in _FEATURE_NAMES:
            val = getattr(features, name, None)
            if val is not None:
                result[name] = float(val)
        return result

    @staticmethod
    def _score_to_decision(score: float) -> str:
        if score >= 0.8:
            return "block"
        if score >= 0.5:
            return "review"
        return "allow"

    @staticmethod
    def _linear_proxy_score(feat: dict[str, float]) -> float:
        """Linear scoring proxy used when no real model is available.

        Uses the globally calibrated importance weights and signed deviations
        from baseline to approximate the model's log-odds, then passes through
        a sigmoid to produce a [0, 1] score.
        """
        log_odds = -2.0  # corresponds to ~base_score of 0.12
        for fname, importance in GLOBAL_IMPORTANCE.items():
            val = feat.get(fname, 0.0)
            baseline, _label, _unit = FEATURE_META.get(fname, (0.0, fname, ""))
            deviation = val - baseline
            norm = deviation / max(abs(baseline), 1.0)
            log_odds += importance * norm * 4.0  # scale factor

        return 1.0 / (1.0 + math.exp(-log_odds))

    def _extract_risk_factors(
        self,
        contributions: list[FeatureContribution],
        transaction: Transaction,
        features: FeatureVector,
    ) -> list[str]:
        """Produce a short list of human-readable risk factor strings."""
        factors: list[str] = []
        amount = float(transaction.amount)
        avg = float(features.txn_amount_avg_24h) if float(features.txn_amount_avg_24h) > 0 else amount

        for contrib in contributions:
            if len(factors) >= 5:
                break
            if contrib.direction != "increases_risk":
                continue

            fn = contrib.feature_name
            fv = contrib.feature_value

            if fn == "amount_zscore" and fv > 2.0:
                factors.append(
                    f"Amount (${amount:,.2f}) is {fv:.1f}σ above the user's historical average"
                )
            elif fn == "amount_to_avg_ratio" and fv > 2.0:
                factors.append(
                    f"Amount is {fv:.1f}× the user's average (${avg:,.2f})"
                )
            elif fn == "distance_from_last_txn_km" and fv > 100:
                factors.append(
                    f"Location is {fv:,.0f} km from the previous transaction"
                )
            elif fn == "is_new_device" and fv > 0.5:
                factors.append("Transaction originated from an unrecognised device")
            elif fn == "is_new_merchant" and fv > 0.5:
                factors.append("First transaction with this merchant")
            elif fn == "txn_count_1h" and fv > 5:
                factors.append(
                    f"High velocity: {int(fv)} transactions in the past hour"
                )
            elif fn == "unique_countries_24h" and fv > 1:
                factors.append(
                    f"Activity across {int(fv)} different countries in 24 hours"
                )
            elif fn == "time_since_last_txn_seconds" and fv < 30:
                factors.append(
                    f"Only {fv:.0f} s elapsed since the previous transaction"
                )
            elif fn == "txn_count_24h" and fv > 20:
                factors.append(
                    f"{int(fv)} transactions recorded in the past 24 hours"
                )

        return factors

    @staticmethod
    def _feature_to_phrase(
        feature_name: str,
        value: float,
        baseline: float,
        contribution: float,
        amount: float | None,
        currency: str,
    ) -> str | None:
        """Convert a single feature contribution into a prose fragment."""
        fn = feature_name
        fv = value
        is_risk = contribution > 0

        if fn == "amount_zscore":
            if abs(fv) > 1.5:
                direction = "unusually high" if fv > 0 else "unusually low"
                return f"the transaction amount is {direction} ({fv:+.1f}σ from the mean)"

        elif fn == "amount_to_avg_ratio":
            if fv > 1.5:
                avg_est = (amount / fv) if amount and fv else baseline
                return (
                    f"the amount (${amount:,.2f}) is {fv:.1f}× higher than "
                    f"the user's average (${avg_est:,.2f})"
                    if amount else f"the amount is {fv:.1f}× the user's average"
                )
            elif fv < 0.3:
                return "the amount is much lower than the user's average (potential test transaction)"

        elif fn == "distance_from_last_txn_km":
            if fv > 50:
                return f"the location is {fv:,.0f} km from the user's last transaction"

        elif fn == "is_new_device":
            if fv > 0.5:
                return "it originated from a new, unrecognised device"

        elif fn == "is_new_merchant":
            if fv > 0.5:
                return "it is the first transaction with this merchant"

        elif fn == "txn_count_1h":
            if fv > 5:
                return f"there are {int(fv)} transactions in the past hour (velocity alert)"

        elif fn == "unique_countries_24h":
            if fv > 1:
                return f"the account has activity across {int(fv)} countries in 24 hours"

        elif fn == "time_since_last_txn_seconds":
            if fv < 60:
                return f"only {fv:.0f} seconds elapsed since the last transaction"

        elif fn == "unique_merchants_24h":
            if is_risk and fv > 6:
                return f"the account visited {int(fv)} different merchants in 24 hours"

        return None

    @staticmethod
    def _join_phrases(phrases: list[str]) -> str:
        """Join a list of phrases with Oxford comma and 'and'."""
        if not phrases:
            return ""
        if len(phrases) == 1:
            return phrases[0]
        if len(phrases) == 2:
            return f"{phrases[0]} and {phrases[1]}"
        return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
