"""Advanced tests for the ensemble scorer.

Covers graceful degradation, weight redistribution, confidence computation,
mock XGBoost / GNN scorers, and concurrent scoring.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from scoring.models.ensemble import (
    DEFAULT_WEIGHTS,
    Decision,
    EnsembleScorer,
)
from scoring.models.rule_engine import RuleEngine
from shared.schemas import FeatureVector, ScoredTransaction, Transaction, TransactionType

# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

_USER_ID = UUID("cccccccc-0000-0000-0000-000000000001")
_TXN_ID = UUID("dddddddd-0000-0000-0000-000000000002")


def _make_transaction(
    amount: Decimal = Decimal("100.00"),
    txn_type: TransactionType = TransactionType.PURCHASE,
    hour: int = 12,
) -> Transaction:
    return Transaction(
        transaction_id=_TXN_ID,
        user_id=_USER_ID,
        amount=amount,
        transaction_type=txn_type,
        timestamp=datetime(2025, 6, 15, hour, 0, 0),
    )


def _make_features(
    txn_count_1h: int = 1,
    txn_amount_avg_24h: Decimal = Decimal("100.00"),
) -> FeatureVector:
    return FeatureVector(
        transaction_id=_TXN_ID,
        user_id=_USER_ID,
        txn_count_1h=txn_count_1h,
        txn_amount_avg_24h=txn_amount_avg_24h,
    )


def _mock_xgboost(return_value: float = 0.4) -> MagicMock:
    """Sync mock that mimics the XGBoost scorer interface."""
    m = MagicMock()
    m.score.return_value = return_value
    return m


def _mock_gnn(return_value: float = 0.3) -> AsyncMock:
    """Async mock that mimics the GNN scorer interface."""
    m = AsyncMock()
    m.score.return_value = return_value
    return m


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """The ensemble must continue scoring when one or more sub-models fail."""

    async def test_one_model_fails_xgboost(self) -> None:
        """When XGBoost raises, only rules and GNN contribute."""
        bad_xgb = MagicMock()
        bad_xgb.score.side_effect = RuntimeError("XGBoost exploded")

        scorer = EnsembleScorer(
            xgboost_scorer=bad_xgb,
            gnn_scorer=_mock_gnn(0.5),
        )
        txn = _make_transaction()
        feat = _make_features()
        result = await scorer.score(txn, feat)

        assert isinstance(result, ScoredTransaction)
        assert 0.0 <= result.fraud_score <= 1.0

    async def test_one_model_fails_gnn(self) -> None:
        """When GNN raises, only rules and XGBoost contribute."""
        bad_gnn = AsyncMock()
        bad_gnn.score.side_effect = RuntimeError("GNN timed out")

        scorer = EnsembleScorer(
            xgboost_scorer=_mock_xgboost(0.6),
            gnn_scorer=bad_gnn,
        )
        txn = _make_transaction()
        feat = _make_features()
        result = await scorer.score(txn, feat)

        assert isinstance(result, ScoredTransaction)
        assert 0.0 <= result.fraud_score <= 1.0

    async def test_two_models_fail(self) -> None:
        """When XGBoost and GNN both fail, only the rule engine score is used."""
        bad_xgb = MagicMock()
        bad_xgb.score.side_effect = ValueError("bad")
        bad_gnn = AsyncMock()
        bad_gnn.score.side_effect = ValueError("bad")

        scorer = EnsembleScorer(xgboost_scorer=bad_xgb, gnn_scorer=bad_gnn)
        txn = _make_transaction()
        feat = _make_features()
        result = await scorer.score(txn, feat)

        assert isinstance(result, ScoredTransaction)
        assert 0.0 <= result.fraud_score <= 1.0

    async def test_all_models_fail_returns_neutral_score(self) -> None:
        """When even the rule engine raises, the fallback score is 0.5."""
        bad_rules = MagicMock(spec=RuleEngine)
        bad_rules.score.side_effect = Exception("rules crashed")
        bad_xgb = MagicMock()
        bad_xgb.score.side_effect = Exception("xgb crashed")
        bad_gnn = AsyncMock()
        bad_gnn.score.side_effect = Exception("gnn crashed")

        scorer = EnsembleScorer(
            rule_engine=bad_rules,
            xgboost_scorer=bad_xgb,
            gnn_scorer=bad_gnn,
        )
        txn = _make_transaction()
        feat = _make_features()
        result = await scorer.score(txn, feat)

        # Fallback must be exactly 0.5
        assert result.fraud_score == pytest.approx(0.5)

    async def test_no_optional_models_uses_rules_only(self) -> None:
        """With no XGBoost or GNN provided, scoring should still succeed."""
        scorer = EnsembleScorer()  # default – no xgb or gnn
        txn = _make_transaction()
        feat = _make_features()
        result = await scorer.score(txn, feat)
        assert isinstance(result, ScoredTransaction)


# ---------------------------------------------------------------------------
# Weight redistribution
# ---------------------------------------------------------------------------


class TestWeightRedistribution:
    """Failed model weights are redistributed proportionally to survivors."""

    def test_compute_weighted_score_all_active(self) -> None:
        """With all models active, weights sum to 1 and score is weighted mean."""
        scorer = EnsembleScorer(weights={"rules": 0.1, "xgboost": 0.6, "gnn": 0.3})
        # rules=0, xgboost=1, gnn=0  → expected = 0.6
        score = scorer._compute_weighted_score(
            model_scores={"rules": 0.0, "xgboost": 1.0, "gnn": 0.0},
            failed_models=[],
        )
        assert score == pytest.approx(0.6, abs=1e-6)

    def test_weight_redistribution_one_missing(self) -> None:
        """When XGBoost is absent, its weight is redistributed to rules + gnn."""
        scorer = EnsembleScorer(weights={"rules": 0.1, "xgboost": 0.6, "gnn": 0.3})
        # Only rules (0.1) and gnn (0.3) active, total active weight = 0.4
        # rules_norm = 0.1/0.4 = 0.25, gnn_norm = 0.3/0.4 = 0.75
        # score = 0.25*0 + 0.75*1.0 = 0.75
        score = scorer._compute_weighted_score(
            model_scores={"rules": 0.0, "gnn": 1.0},
            failed_models=["xgboost"],
        )
        assert score == pytest.approx(0.75, abs=1e-6)

    def test_weight_redistribution_two_missing(self) -> None:
        """When only rules survive, the raw rule score is returned."""
        scorer = EnsembleScorer(weights={"rules": 0.1, "xgboost": 0.6, "gnn": 0.3})
        score = scorer._compute_weighted_score(
            model_scores={"rules": 0.8},
            failed_models=["xgboost", "gnn"],
        )
        assert score == pytest.approx(0.8, abs=1e-6)

    def test_fallback_on_empty_scores(self) -> None:
        """Empty model_scores dict produces neutral fallback 0.5."""
        scorer = EnsembleScorer()
        score = scorer._compute_weighted_score(model_scores={}, failed_models=["rules"])
        assert score == pytest.approx(0.5)

    def test_equal_weighting_when_configured_weights_sum_to_zero(self) -> None:
        """If all active model weights are 0, fall back to simple average."""
        scorer = EnsembleScorer(weights={"rules": 0.0, "xgboost": 0.0, "gnn": 0.0})
        score = scorer._compute_weighted_score(
            model_scores={"rules": 0.2, "xgboost": 0.8},
            failed_models=[],
        )
        assert score == pytest.approx(0.5, abs=1e-6)

    def test_score_clamped_after_redistribution(self) -> None:
        """Redistributed score must never exceed 1.0."""
        scorer = EnsembleScorer(weights={"rules": 0.5, "xgboost": 0.5})
        score = scorer._compute_weighted_score(
            model_scores={"rules": 1.0, "xgboost": 1.0},
            failed_models=[],
        )
        assert score <= 1.0


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------


class TestConfidenceComputation:
    """_compute_confidence reflects model agreement."""

    def test_perfect_agreement_is_high(self) -> None:
        """All models returning the same score should yield confidence 1.0."""
        confidence = EnsembleScorer._compute_confidence(
            {"rules": 0.7, "xgboost": 0.7, "gnn": 0.7}
        )
        assert confidence == pytest.approx(1.0)

    def test_maximum_disagreement_is_low(self) -> None:
        """Maximum spread between 0 and 1 should yield confidence close to 0."""
        confidence = EnsembleScorer._compute_confidence(
            {"rules": 0.0, "xgboost": 1.0}
        )
        assert confidence < 0.5

    def test_single_model_confidence_is_one(self) -> None:
        """When only one model is present confidence must be 1.0 (no comparison)."""
        confidence = EnsembleScorer._compute_confidence({"rules": 0.3})
        assert confidence == pytest.approx(1.0)

    def test_empty_scores_confidence(self) -> None:
        """Empty dict returns 1.0 (edge case, no models to disagree)."""
        confidence = EnsembleScorer._compute_confidence({})
        assert confidence == pytest.approx(1.0)

    def test_moderate_disagreement(self) -> None:
        """Moderate disagreement yields a confidence between 0 and 1."""
        confidence = EnsembleScorer._compute_confidence(
            {"rules": 0.4, "xgboost": 0.6}
        )
        assert 0.0 < confidence < 1.0

    def test_confidence_always_non_negative(self) -> None:
        """Confidence must never be negative regardless of inputs."""
        confidence = EnsembleScorer._compute_confidence(
            {"rules": 0.0, "xgboost": 1.0, "gnn": 0.5}
        )
        assert confidence >= 0.0


# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------


class TestDecisionThresholds:
    """Custom thresholds and boundary behaviour."""

    def test_default_thresholds(self) -> None:
        """Block >= 0.8, Review [0.5, 0.8), Allow < 0.5."""
        assert EnsembleScorer.decide(0.8) == Decision.BLOCK
        assert EnsembleScorer.decide(0.5) == Decision.REVIEW
        assert EnsembleScorer.decide(0.49) == Decision.ALLOW

    def test_custom_thresholds(self) -> None:
        """Custom block / review thresholds are respected."""
        assert EnsembleScorer.decide(0.7, threshold_block=0.7) == Decision.BLOCK
        assert EnsembleScorer.decide(0.69, threshold_block=0.7, threshold_review=0.4) == Decision.REVIEW
        assert EnsembleScorer.decide(0.39, threshold_block=0.7, threshold_review=0.4) == Decision.ALLOW

    def test_boundary_exactly_at_block(self) -> None:
        """Score exactly at block threshold should be BLOCK."""
        assert EnsembleScorer.decide(1.0) == Decision.BLOCK

    def test_boundary_exactly_at_review(self) -> None:
        """Score exactly at review threshold should be REVIEW."""
        assert EnsembleScorer.decide(0.5) == Decision.REVIEW


# ---------------------------------------------------------------------------
# Mock XGBoost and GNN scorers
# ---------------------------------------------------------------------------


class TestWithMockSubModels:
    """Validate end-to-end scoring with controlled sub-model outputs."""

    async def test_xgboost_score_influences_result(self) -> None:
        """High XGBoost score (weight=0.6) should dominate the ensemble."""
        scorer = EnsembleScorer(
            xgboost_scorer=_mock_xgboost(0.95),
            gnn_scorer=_mock_gnn(0.0),
            weights={"rules": 0.0, "xgboost": 1.0, "gnn": 0.0},
        )
        txn = _make_transaction(amount=Decimal("50.00"), hour=12)
        feat = _make_features(txn_count_1h=1, txn_amount_avg_24h=Decimal("100.00"))
        result = await scorer.score(txn, feat)
        assert result.fraud_score == pytest.approx(0.95)

    async def test_gnn_score_contributes(self) -> None:
        """GNN score should participate in the weighted sum."""
        scorer = EnsembleScorer(
            xgboost_scorer=_mock_xgboost(0.0),
            gnn_scorer=_mock_gnn(1.0),
            weights={"rules": 0.0, "xgboost": 0.0, "gnn": 1.0},
        )
        txn = _make_transaction(amount=Decimal("50.00"), hour=12)
        feat = _make_features()
        result = await scorer.score(txn, feat)
        assert result.fraud_score == pytest.approx(1.0)

    async def test_gnn_interface_called_with_user_id(self) -> None:
        """GNN scorer must be called with (user_id, features)."""
        mock_gnn = _mock_gnn(0.3)
        scorer = EnsembleScorer(gnn_scorer=mock_gnn)
        txn = _make_transaction()
        feat = _make_features()
        await scorer.score(txn, feat)
        mock_gnn.score.assert_called_once_with(txn.user_id, feat)

    async def test_xgboost_interface_called_with_features(self) -> None:
        """XGBoost scorer must be called with (features,) only."""
        mock_xgb = _mock_xgboost(0.4)
        scorer = EnsembleScorer(xgboost_scorer=mock_xgb)
        txn = _make_transaction()
        feat = _make_features()
        await scorer.score(txn, feat)
        mock_xgb.score.assert_called_once_with(feat)

    async def test_flagged_transaction_increments_counter(self) -> None:
        """Flagged transactions (BLOCK or REVIEW) must increment total_flagged."""
        scorer = EnsembleScorer(
            xgboost_scorer=_mock_xgboost(0.95),
            weights={"rules": 0.0, "xgboost": 1.0, "gnn": 0.0},
        )
        txn = _make_transaction()
        feat = _make_features()
        before = scorer.total_flagged
        await scorer.score(txn, feat)
        assert scorer.total_flagged == before + 1

    async def test_non_flagged_does_not_increment_flagged_counter(self) -> None:
        """Safe transactions (ALLOW) must NOT increment total_flagged."""
        scorer = EnsembleScorer(
            xgboost_scorer=_mock_xgboost(0.1),
            weights={"rules": 0.0, "xgboost": 1.0, "gnn": 0.0},
        )
        txn = _make_transaction()
        feat = _make_features()
        before = scorer.total_flagged
        await scorer.score(txn, feat)
        assert scorer.total_flagged == before


# ---------------------------------------------------------------------------
# Concurrent scoring
# ---------------------------------------------------------------------------


class TestConcurrentScoring:
    """Ensure scorer is safe under concurrent load."""

    async def test_concurrent_scoring_all_succeed(self) -> None:
        """N concurrent score calls should all return valid results."""
        scorer = EnsembleScorer(
            xgboost_scorer=_mock_xgboost(0.3),
            gnn_scorer=_mock_gnn(0.2),
        )
        txns = [_make_transaction() for _ in range(20)]
        feats = [_make_features() for _ in range(20)]
        results = await asyncio.gather(*[scorer.score(t, f) for t, f in zip(txns, feats)])
        assert len(results) == 20
        for r in results:
            assert isinstance(r, ScoredTransaction)
            assert 0.0 <= r.fraud_score <= 1.0

    async def test_total_scored_accurate_after_concurrent_calls(self) -> None:
        """total_scored must match the number of completed score calls."""
        scorer = EnsembleScorer()
        initial = scorer.total_scored
        n = 10
        txn = _make_transaction()
        feat = _make_features()
        await asyncio.gather(*[scorer.score(txn, feat) for _ in range(n)])
        assert scorer.total_scored == initial + n

    async def test_concurrent_mixed_failure_tolerance(self) -> None:
        """When some sub-model calls fail intermittently, no exception leaks out."""
        call_count = 0

        mock_gnn = AsyncMock()

        async def flaky_gnn(user_id: object, features: object) -> float:
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise RuntimeError("intermittent failure")
            return 0.4

        mock_gnn.score.side_effect = flaky_gnn

        scorer = EnsembleScorer(gnn_scorer=mock_gnn)
        txn = _make_transaction()
        feat = _make_features()
        results = await asyncio.gather(*[scorer.score(txn, feat) for _ in range(9)])
        assert all(isinstance(r, ScoredTransaction) for r in results)
