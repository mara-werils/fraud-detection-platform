"""Ensemble scorer interface with MVP rule-based implementation."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import StrEnum

import structlog

from shared.schemas import FeatureVector, ScoredTransaction, Transaction

from scoring.models.rule_engine import RuleEngine, ScoringResult

logger = structlog.get_logger(__name__)


class Decision(StrEnum):
    """Scoring decision based on fraud score thresholds."""

    BLOCK = "block"
    REVIEW = "review"
    ALLOW = "allow"


class BaseScorer(ABC):
    """Abstract base class for fraud scorers."""

    @abstractmethod
    def score(
        self, transaction: Transaction, features: FeatureVector
    ) -> ScoredTransaction:
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


class EnsembleScorer(BaseScorer):
    """Ensemble scorer that combines multiple scoring models.

    MVP implementation uses only the rule engine. Prepared for adding
    XGBoost and GNN models in future iterations.
    """

    def __init__(
        self,
        model_version: str = "rule-engine-v1",
        threshold_block: float = 0.8,
        threshold_review: float = 0.5,
        rule_engine: RuleEngine | None = None,
    ) -> None:
        self._model_version = model_version
        self._threshold_block = threshold_block
        self._threshold_review = threshold_review
        self._rule_engine = rule_engine or RuleEngine()
        self._model_type = "rule-based"
        self._total_scored: int = 0
        self._total_flagged: int = 0

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

    def score(
        self, transaction: Transaction, features: FeatureVector
    ) -> ScoredTransaction:
        """Score a transaction using the rule engine ensemble.

        Args:
            transaction: The raw transaction to evaluate.
            features: Pre-computed feature vector.

        Returns:
            A fully populated ScoredTransaction.
        """
        start = time.perf_counter()

        # MVP: only rule engine; future: combine with XGBoost, GNN
        rule_result: ScoringResult = self._rule_engine.score(transaction, features)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        decision = self.decide(
            rule_result.fraud_score,
            self._threshold_block,
            self._threshold_review,
        )

        is_flagged = decision in (Decision.BLOCK, Decision.REVIEW)
        flag_reason: str | None = None
        if is_flagged and rule_result.triggered_rules:
            flag_reason = (
                f"{decision.value}: {', '.join(rule_result.triggered_rules)}"
            )

        self._total_scored += 1
        if is_flagged:
            self._total_flagged += 1

        return ScoredTransaction(
            transaction_id=transaction.transaction_id,
            user_id=transaction.user_id,
            amount=transaction.amount,
            currency=transaction.currency,
            transaction_type=transaction.transaction_type,
            timestamp=transaction.timestamp,
            fraud_score=rule_result.fraud_score,
            model_version=self._model_version,
            scoring_latency_ms=elapsed_ms,
            features=features,
            is_flagged=is_flagged,
            flag_reason=flag_reason,
        )
