"""Ensemble scorer combining XGBoost, GNN, and rule-based models."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import structlog

from scoring.models.rule_engine import RuleEngine, ScoringResult
from shared.schemas import FeatureVector, ScoredTransaction, Transaction

logger = structlog.get_logger(__name__)


class Decision(StrEnum):
    """Scoring decision based on fraud score thresholds."""

    BLOCK = "block"
    REVIEW = "review"
    ALLOW = "allow"


class BaseScorer(ABC):
    """Abstract base class for fraud scorers."""

    @abstractmethod
    async def score(self, transaction: Transaction, features: FeatureVector) -> ScoredTransaction:
        """Score a transaction and return a ScoredTransaction.

        Args:
            transaction: The raw transaction to evaluate.
            features: Pre-computed feature vector.

        Returns:
            A ScoredTransaction with fraud score and metadata.
        """
        ...

    @staticmethod
    def decide(
        fraud_score: float,
        threshold_block: float = 0.8,
        threshold_review: float = 0.5,
    ) -> Decision:
        """Make a decision based on fraud score thresholds.

        Args:
            fraud_score: The computed fraud score in [0, 1].
            threshold_block: Score above which to block. Default 0.8.
            threshold_review: Score above which to flag for review. Default 0.5.

        Returns:
            Decision enum value: BLOCK, REVIEW, or ALLOW.
        """
        if fraud_score >= threshold_block:
            return Decision.BLOCK
        if fraud_score >= threshold_review:
            return Decision.REVIEW
        return Decision.ALLOW


# Default ensemble weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "xgboost": 0.6,
    "gnn": 0.3,
    "rules": 0.1,
}


class EnsembleScorer(BaseScorer):
    """Ensemble scorer that combines XGBoost, GNN, and rule-based models.

    Supports graceful degradation: if a model fails, its weight is
    redistributed proportionally among the surviving models.

    Args:
        model_version: Human-readable version string.
        threshold_block: Score at or above which to block.
        threshold_review: Score at or above which to flag for review.
        rule_engine: Rule-based scorer instance.
        xgboost_scorer: XGBoost scorer instance (optional).
        gnn_scorer: GNN scorer instance (optional).
        weights: Dict mapping model name to weight. Defaults to
            xgboost=0.6, gnn=0.3, rules=0.1.
        llm_explainer: Optional LLM explainer for async explanations.
    """

    def __init__(
        self,
        model_version: str = "ensemble-v1",
        threshold_block: float = 0.8,
        threshold_review: float = 0.5,
        rule_engine: RuleEngine | None = None,
        xgboost_scorer: Any | None = None,
        gnn_scorer: Any | None = None,
        weights: dict[str, float] | None = None,
        llm_explainer: Any | None = None,
    ) -> None:
        self._model_version = model_version
        self._threshold_block = threshold_block
        self._threshold_review = threshold_review
        self._rule_engine = rule_engine or RuleEngine()
        self._xgboost_scorer = xgboost_scorer
        self._gnn_scorer = gnn_scorer
        self._weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self._llm_explainer = llm_explainer
        self._model_type = "ensemble"
        self._total_scored: int = 0
        self._total_flagged: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def total_scored(self) -> int:
        return self._total_scored

    @property
    def total_flagged(self) -> int:
        return self._total_flagged

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def score(self, transaction: Transaction, features: FeatureVector) -> ScoredTransaction:
        """Score a transaction using the full model ensemble.

        Collects scores from all available models, applies weighted
        combination with graceful degradation, and optionally triggers
        async LLM explanation.

        Args:
            transaction: The raw transaction to evaluate.
            features: Pre-computed feature vector.

        Returns:
            A fully populated ScoredTransaction.
        """
        start = time.perf_counter()

        # Collect individual model scores
        model_scores: dict[str, float] = {}
        failed_models: list[str] = []

        # --- Rule engine (sync, always available) ---
        try:
            rule_result: ScoringResult = self._rule_engine.score(transaction, features)
            model_scores["rules"] = rule_result.fraud_score
        except Exception as exc:
            logger.error("ensemble_rule_engine_error", error=str(exc))
            failed_models.append("rules")

        # --- XGBoost (sync) ---
        if self._xgboost_scorer is not None:
            try:
                xgb_score = self._xgboost_scorer.score(features)
                model_scores["xgboost"] = xgb_score
            except Exception as exc:
                logger.error("ensemble_xgboost_error", error=str(exc))
                failed_models.append("xgboost")
        else:
            failed_models.append("xgboost")

        # --- GNN (async) ---
        if self._gnn_scorer is not None:
            try:
                gnn_score = await self._gnn_scorer.score(transaction.user_id, features)
                model_scores["gnn"] = gnn_score
            except Exception as exc:
                logger.error("ensemble_gnn_error", error=str(exc))
                failed_models.append("gnn")
        else:
            failed_models.append("gnn")

        # Compute weighted ensemble score with graceful degradation
        ensemble_score = self._compute_weighted_score(model_scores, failed_models)
        confidence = self._compute_confidence(model_scores)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        decision = self.decide(ensemble_score, self._threshold_block, self._threshold_review)

        is_flagged = decision in (Decision.BLOCK, Decision.REVIEW)
        flag_reason: str | None = None
        if is_flagged:
            triggered = (
                rule_result.triggered_rules
                if "rules" in model_scores and rule_result.triggered_rules
                else []
            )
            reason_parts = [f"{decision.value}"]
            if triggered:
                reason_parts.append(f"rules: {', '.join(triggered)}")
            for name, s in model_scores.items():
                if name != "rules":
                    reason_parts.append(f"{name}: {s:.3f}")
            flag_reason = " | ".join(reason_parts)

        self._total_scored += 1
        if is_flagged:
            self._total_flagged += 1

        # Log individual + ensemble scores
        logger.info(
            "ensemble_scored",
            transaction_id=str(transaction.transaction_id),
            model_scores=model_scores,
            ensemble_score=round(ensemble_score, 4),
            confidence=round(confidence, 4),
            failed_models=failed_models,
            decision=decision.value,
            latency_ms=round(elapsed_ms, 2),
        )

        scored = ScoredTransaction(
            transaction_id=transaction.transaction_id,
            user_id=transaction.user_id,
            amount=transaction.amount,
            currency=transaction.currency,
            transaction_type=transaction.transaction_type,
            timestamp=transaction.timestamp,
            fraud_score=ensemble_score,
            model_version=self._model_version,
            scoring_latency_ms=elapsed_ms,
            features=features,
            is_flagged=is_flagged,
            flag_reason=flag_reason,
        )

        # Fire-and-forget LLM explanation (does not block response)
        if self._llm_explainer is not None and is_flagged:
            all_scores = {**model_scores, "ensemble": ensemble_score}
            asyncio.create_task(
                self._generate_explanation(transaction, features, all_scores, decision.value)
            )

        return scored

    # ------------------------------------------------------------------
    # Weighted scoring with graceful degradation
    # ------------------------------------------------------------------

    def _compute_weighted_score(
        self,
        model_scores: dict[str, float],
        failed_models: list[str],
    ) -> float:
        """Compute weighted ensemble score, redistributing failed model weights.

        Args:
            model_scores: Mapping of model name to score for successful models.
            failed_models: List of model names that failed.

        Returns:
            Weighted fraud score clamped to [0, 1].
        """
        if not model_scores:
            logger.error("ensemble_all_models_failed")
            return 0.5  # neutral fallback

        # Calculate total weight of successful models for normalization
        active_weight_sum = sum(self._weights.get(name, 0.0) for name in model_scores)

        if active_weight_sum <= 0:
            # Equal weighting if configured weights are zero
            return sum(model_scores.values()) / len(model_scores)

        # Normalize weights so they sum to 1.0 across active models
        weighted_sum = sum(
            (self._weights.get(name, 0.0) / active_weight_sum) * score
            for name, score in model_scores.items()
        )

        return max(0.0, min(1.0, weighted_sum))

    @staticmethod
    def _compute_confidence(model_scores: dict[str, float]) -> float:
        """Compute a confidence score based on model agreement.

        Higher confidence when models agree; lower when they disagree.
        Returns 1.0 if only one model is available.

        Args:
            model_scores: Mapping of model name to score.

        Returns:
            Confidence score in [0, 1].
        """
        if len(model_scores) <= 1:
            return 1.0

        scores = list(model_scores.values())
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        # Max possible variance for [0,1] scores is 0.25.
        # Confidence = 1 - normalized_variance
        return max(0.0, 1.0 - (variance / 0.25))

    # ------------------------------------------------------------------
    # LLM explanation (fire-and-forget)
    # ------------------------------------------------------------------

    async def _generate_explanation(
        self,
        transaction: Transaction,
        features: FeatureVector,
        scores: dict[str, float],
        decision: str,
    ) -> None:
        """Generate an LLM explanation, suppressing all errors."""
        try:
            await self._llm_explainer.explain(  # type: ignore[union-attr]
                transaction, features, scores, decision
            )
        except Exception as exc:
            logger.warning(
                "ensemble_explanation_error",
                txn_id=str(transaction.transaction_id),
                error=str(exc),
            )
