"""Integration tests for the fraud detection platform.

Exercises end-to-end flows across multiple layers:
  - Full scoring pipeline (transaction → features → score → response)
  - Webhook delivery flow
  - Case lifecycle (create → assign → investigate → resolve)
  - Feedback → model retraining data export flow

All tests are marked with @pytest.mark.integration.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from scoring.models.ensemble import EnsembleScorer
from scoring.models.rule_engine import RuleEngine
from scoring.services.case_manager import CaseManager, CaseStatus
from scoring.services.feedback_store import FeedbackStore
from scoring.services.transaction_store import TransactionStore
from scoring.services.webhook_manager import WebhookEvent, WebhookManager, WebhookStatus
from shared.schemas import (
    FeatureVector,
    ScoredTransaction,
    Transaction,
    TransactionType,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_ID = UUID("feedface-0000-0000-0000-000000000001")
_TXN_ID = UUID("deadbeef-0000-0000-0000-000000000002")


def _build_transaction(
    amount: Decimal = Decimal("100.00"),
    hour: int = 14,
    txn_type: TransactionType = TransactionType.PURCHASE,
) -> Transaction:
    return Transaction(
        transaction_id=_TXN_ID,
        user_id=_USER_ID,
        amount=amount,
        currency="USD",
        transaction_type=txn_type,
        timestamp=datetime(2025, 6, 15, hour, 0, 0),
    )


def _build_features(
    txn_count_1h: int = 2,
    txn_amount_avg_24h: Decimal = Decimal("100.00"),
    distance_km: float = 5.0,
    is_new_device: bool = False,
    unique_countries: int = 1,
) -> FeatureVector:
    return FeatureVector(
        transaction_id=_TXN_ID,
        user_id=_USER_ID,
        txn_count_1h=txn_count_1h,
        txn_amount_avg_24h=txn_amount_avg_24h,
        distance_from_last_txn_km=distance_km,
        is_new_device=is_new_device,
        unique_countries_24h=unique_countries,
    )


# ---------------------------------------------------------------------------
# Full scoring pipeline
# ---------------------------------------------------------------------------


class TestFullScoringPipeline:
    """Transaction → features → ensemble → ScoredTransaction."""

    async def test_normal_transaction_pipeline(self) -> None:
        """A safe transaction should produce ALLOW decision and not be flagged."""
        scorer = EnsembleScorer()
        txn = _build_transaction(amount=Decimal("25.00"), hour=14)
        features = _build_features()

        result = await scorer.score(txn, features)

        assert isinstance(result, ScoredTransaction)
        assert result.transaction_id == txn.transaction_id
        assert result.user_id == txn.user_id
        assert result.amount == txn.amount
        assert 0.0 <= result.fraud_score <= 1.0
        assert result.is_flagged is False
        assert result.model_version == "ensemble-v1"
        assert result.scoring_latency_ms >= 0

    async def test_suspicious_transaction_pipeline(self) -> None:
        """A high-risk transaction should be flagged with a reason."""
        scorer = EnsembleScorer()
        txn = _build_transaction(
            amount=Decimal("9999.99"),
            hour=3,
            txn_type=TransactionType.TRANSFER,
        )
        features = _build_features(
            txn_count_1h=15,
            txn_amount_avg_24h=Decimal("50.00"),
            distance_km=2000.0,
            is_new_device=True,
            unique_countries=4,
        )

        result = await scorer.score(txn, features)

        assert result.is_flagged is True
        assert result.flag_reason is not None
        assert result.fraud_score >= 0.5  # at least REVIEW

    async def test_pipeline_stores_result_in_transaction_store(self) -> None:
        """Scored transactions should be retrievable from TransactionStore."""
        scorer = EnsembleScorer()
        store = TransactionStore()

        txn = _build_transaction()
        features = _build_features()

        scored = await scorer.score(txn, features)
        store.add(scored)

        retrieved = store.get(str(txn.transaction_id))
        assert retrieved is not None
        assert retrieved.fraud_score == scored.fraud_score

    async def test_pipeline_total_scored_increments(self) -> None:
        """scorer.total_scored must increase by 1 for each successful call."""
        scorer = EnsembleScorer()
        before = scorer.total_scored

        await scorer.score(_build_transaction(), _build_features())
        assert scorer.total_scored == before + 1

    async def test_pipeline_with_mock_sub_models(self) -> None:
        """Pipeline operates correctly when sub-models are injected mocks."""
        mock_xgb = MagicMock()
        mock_xgb.score.return_value = 0.7

        mock_gnn = AsyncMock()
        mock_gnn.score.return_value = 0.6

        scorer = EnsembleScorer(xgboost_scorer=mock_xgb, gnn_scorer=mock_gnn)
        result = await scorer.score(_build_transaction(), _build_features())

        assert isinstance(result, ScoredTransaction)
        mock_xgb.score.assert_called_once()
        mock_gnn.score.assert_called_once()

    async def test_pipeline_rule_engine_results_embedded(self) -> None:
        """Features fed to RuleEngine directly should match the ensemble path."""
        rule_engine = RuleEngine()
        scorer = EnsembleScorer(rule_engine=rule_engine)

        txn = _build_transaction(amount=Decimal("9999.00"), hour=3)
        features = _build_features(
            txn_count_1h=10,
            txn_amount_avg_24h=Decimal("50.00"),
            distance_km=600.0,
        )

        direct_rule = rule_engine.score(txn, features)
        scored = await scorer.score(txn, features)

        # Rule engine score should influence the ensemble score
        assert scored.fraud_score >= direct_rule.fraud_score * 0.1  # at least some signal


# ---------------------------------------------------------------------------
# Webhook delivery flow
# ---------------------------------------------------------------------------


class TestWebhookDeliveryFlow:
    """Register → trigger → inspect delivery record."""

    async def test_webhook_registered_and_listed(self) -> None:
        """Registered webhook should appear in list_webhooks."""
        manager = WebhookManager()
        wh = manager.register(
            url="https://example.com/hook",
            events=["transaction.flagged"],
            description="Test webhook",
        )
        webhooks = manager.list_webhooks()
        ids = [w.webhook_id for w in webhooks]
        assert wh.webhook_id in ids

    async def test_webhook_delivered_on_matching_event(self) -> None:
        """When event matches subscription, delivery is attempted."""
        manager = WebhookManager()
        manager.register(
            url="https://example.com/hook",
            events=[WebhookEvent.TRANSACTION_FLAGGED],
        )

        # Patch the HTTP call so we don't hit the network
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            deliveries = await manager.deliver(
                WebhookEvent.TRANSACTION_FLAGGED,
                {"transaction_id": str(_TXN_ID), "fraud_score": 0.9},
            )

        assert len(deliveries) == 1
        assert deliveries[0].success is True

    async def test_webhook_not_delivered_for_wrong_event(self) -> None:
        """Webhook subscribed to one event must not receive a different event."""
        manager = WebhookManager()
        manager.register(
            url="https://example.com/hook",
            events=[WebhookEvent.CASE_CREATED],
        )

        deliveries = await manager.deliver(
            WebhookEvent.TRANSACTION_FLAGGED,
            {"transaction_id": str(_TXN_ID)},
        )
        assert len(deliveries) == 0

    async def test_failed_delivery_increments_failure_count(self) -> None:
        """A failed HTTP delivery must increment the webhook failure_count."""
        manager = WebhookManager()
        wh = manager.register(url="https://example.com/hook", events=["transaction.flagged"])

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = ConnectionError("connection refused")
            await manager.deliver("transaction.flagged", {"transaction_id": "abc"})

        updated = manager.get(wh.webhook_id)
        assert updated is not None
        assert updated.failure_count >= 1

    async def test_webhook_auto_paused_after_max_failures(self) -> None:
        """Webhook should be auto-paused after max_consecutive_failures failures."""
        manager = WebhookManager(max_consecutive_failures=3)
        wh = manager.register(url="https://bad.example.com/hook", events=["transaction.flagged"])

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = ConnectionError("always fails")
            for _ in range(3):
                await manager.deliver("transaction.flagged", {"data": "x"})

        updated = manager.get(wh.webhook_id)
        assert updated is not None
        assert updated.status == WebhookStatus.FAILED

    async def test_unregister_removes_webhook(self) -> None:
        """Unregistered webhook should no longer be listed or receive events."""
        manager = WebhookManager()
        wh = manager.register(url="https://example.com/hook", events=["transaction.flagged"])
        removed = manager.unregister(wh.webhook_id)
        assert removed is True
        assert manager.get(wh.webhook_id) is None


# ---------------------------------------------------------------------------
# Case lifecycle
# ---------------------------------------------------------------------------


class TestCaseLifecycle:
    """create → assign → investigate → resolve."""

    def test_create_open_case(self) -> None:
        """Newly created case must have OPEN status."""
        mgr = CaseManager()
        case = mgr.create_case(
            transaction_id=str(_TXN_ID),
            user_id=str(_USER_ID),
            fraud_score=0.85,
            amount=500.0,
            priority="high",
        )
        assert case.status == CaseStatus.OPEN
        assert case.fraud_score == pytest.approx(0.85)
        assert case.priority.value == "high"

    def test_assign_case(self) -> None:
        """Assigned analyst should be reflected in the case."""
        mgr = CaseManager()
        case = mgr.create_case(str(_TXN_ID), str(_USER_ID))
        updated = mgr.assign(case.case_id, analyst="alice", actor="manager")
        assert updated is not None
        assert updated.assigned_to == "alice"

    def test_transition_to_investigating(self) -> None:
        """Status should change to investigating."""
        mgr = CaseManager()
        case = mgr.create_case(str(_TXN_ID), str(_USER_ID))
        updated = mgr.update_status(case.case_id, "investigating", actor="alice")
        assert updated is not None
        assert updated.status == CaseStatus.INVESTIGATING

    def test_resolve_as_fraud(self) -> None:
        """Resolving as fraud sets is_fraud=True and resolved_at."""
        mgr = CaseManager()
        case = mgr.create_case(str(_TXN_ID), str(_USER_ID))
        mgr.update_status(case.case_id, "investigating")
        resolved = mgr.update_status(case.case_id, "resolved_fraud", actor="bob")
        assert resolved is not None
        assert resolved.is_fraud is True
        assert resolved.resolved_at is not None

    def test_resolve_as_legitimate(self) -> None:
        """Resolving as legitimate sets is_fraud=False."""
        mgr = CaseManager()
        case = mgr.create_case(str(_TXN_ID), str(_USER_ID))
        resolved = mgr.update_status(case.case_id, "resolved_legitimate")
        assert resolved is not None
        assert resolved.is_fraud is False

    def test_add_note_to_case(self) -> None:
        """Notes added to a case should be persisted in case.notes."""
        mgr = CaseManager()
        case = mgr.create_case(str(_TXN_ID), str(_USER_ID))
        updated = mgr.add_note(case.case_id, author="analyst", content="Looks suspicious.")
        assert updated is not None
        note_contents = [n.content for n in updated.notes]
        assert "Looks suspicious." in note_contents

    def test_audit_trail_recorded(self) -> None:
        """Every state change must produce a CaseEvent in the events list."""
        mgr = CaseManager()
        case = mgr.create_case(str(_TXN_ID), str(_USER_ID))
        initial_events = len(case.events)

        mgr.update_status(case.case_id, "investigating", actor="tester")
        mgr.assign(case.case_id, analyst="carol", actor="manager")

        refreshed = mgr.get_case(case.case_id)
        assert refreshed is not None
        assert len(refreshed.events) > initial_events

    def test_case_stats_after_lifecycle(self) -> None:
        """Stats must reflect resolved cases correctly."""
        mgr = CaseManager()
        txn1, txn2 = str(uuid4()), str(uuid4())
        c1 = mgr.create_case(txn1, "user_a", priority="high")
        _c2 = mgr.create_case(txn2, "user_b", priority="low")
        mgr.update_status(c1.case_id, "resolved_fraud")

        stats = mgr.stats()
        assert stats["total"] == 2
        assert "resolved_fraud" in stats["by_status"]
        assert stats["by_status"]["resolved_fraud"] >= 1

    def test_list_cases_with_status_filter(self) -> None:
        """list_cases with status filter returns only matching cases."""
        mgr = CaseManager()
        c1 = mgr.create_case(str(uuid4()), "u1")
        c2 = mgr.create_case(str(uuid4()), "u2")
        mgr.update_status(c1.case_id, "investigating")

        cases, total = mgr.list_cases(status="investigating")
        ids = [c.case_id for c in cases]
        assert c1.case_id in ids
        assert c2.case_id not in ids


# ---------------------------------------------------------------------------
# Feedback → model retraining flow
# ---------------------------------------------------------------------------


class TestFeedbackToRetrainingFlow:
    """submit feedback → inspect stats → export training data."""

    def test_feedback_recorded_and_retrievable(self) -> None:
        """Submitted feedback should be retrievable by transaction_id."""
        store = FeedbackStore()
        txn_id = str(uuid4())
        entry = store.add(
            transaction_id=txn_id,
            is_fraud=True,
            analyst="alice",
            original_score=0.9,
            original_decision="BLOCK",
        )
        retrieved = store.get(txn_id)
        assert retrieved is not None
        assert retrieved.feedback_id == entry.feedback_id
        assert retrieved.is_fraud is True

    def test_feedback_stats_agreement_rate(self) -> None:
        """When model and analyst agree, agreement_rate should be 1.0."""
        store = FeedbackStore()
        for _ in range(5):
            store.add(
                transaction_id=str(uuid4()),
                is_fraud=True,
                analyst="carol",
                original_score=0.9,
                original_decision="BLOCK",
            )
        stats = store.stats()
        assert stats.agreement_rate == pytest.approx(1.0)

    def test_feedback_stats_false_positive_rate(self) -> None:
        """False positive rate should reflect model flagging legitimate txns."""
        store = FeedbackStore()
        # Model said BLOCK but analyst says legitimate
        for _ in range(2):
            store.add(
                transaction_id=str(uuid4()),
                is_fraud=False,
                analyst="dave",
                original_decision="BLOCK",
            )
        # Model said BLOCK and analyst agrees
        for _ in range(2):
            store.add(
                transaction_id=str(uuid4()),
                is_fraud=True,
                analyst="dave",
                original_decision="BLOCK",
            )
        stats = store.stats()
        assert stats.false_positive_rate == pytest.approx(0.5)

    def test_training_data_export_format(self) -> None:
        """Exported training data should have correct keys per entry."""
        store = FeedbackStore()
        txn_id = str(uuid4())
        store.add(
            transaction_id=txn_id,
            is_fraud=True,
            analyst="eve",
            original_score=0.88,
        )
        data = store.export_training_data()
        assert len(data) >= 1
        entry = next(d for d in data if d["transaction_id"] == txn_id)
        assert "transaction_id" in entry
        assert "is_fraud" in entry
        assert "original_score" in entry
        assert "analyst" in entry
        assert "created_at" in entry

    def test_training_data_labels_accurate(self) -> None:
        """Exported labels must match what was submitted."""
        store = FeedbackStore()
        fraud_id = str(uuid4())
        legit_id = str(uuid4())
        store.add(transaction_id=fraud_id, is_fraud=True, analyst="frank")
        store.add(transaction_id=legit_id, is_fraud=False, analyst="frank")

        data = store.export_training_data()
        by_txn = {d["transaction_id"]: d for d in data}

        assert by_txn[fraud_id]["is_fraud"] is True
        assert by_txn[legit_id]["is_fraud"] is False

    def test_feedback_list_filter_by_analyst(self) -> None:
        """Filtering by analyst should only return that analyst's entries."""
        store = FeedbackStore()
        store.add(str(uuid4()), is_fraud=True, analyst="alice")
        store.add(str(uuid4()), is_fraud=False, analyst="bob")
        store.add(str(uuid4()), is_fraud=True, analyst="alice")

        entries, total = store.list_entries(analyst="alice")
        assert total == 2
        assert all(e.analyst == "alice" for e in entries)

    def test_feedback_empty_store_stats(self) -> None:
        """Stats on empty store should return zeroed-out values."""
        store = FeedbackStore()
        stats = store.stats()
        assert stats.total_feedback == 0
        assert stats.agreement_rate == pytest.approx(0.0)

    async def test_end_to_end_score_then_feedback_then_export(self) -> None:
        """Full pipeline: score a transaction, record feedback, export for retraining."""
        scorer = EnsembleScorer()
        store = FeedbackStore()
        txn_store = TransactionStore()

        txn = _build_transaction(amount=Decimal("50.00"), hour=14)
        features = _build_features()

        # Step 1: Score
        scored = await scorer.score(txn, features)
        txn_store.add(scored)

        # Step 2: Analyst reviews and records feedback
        decision = "BLOCK" if scored.fraud_score >= 0.8 else "REVIEW" if scored.fraud_score >= 0.5 else "ALLOW"
        store.add(
            transaction_id=str(txn.transaction_id),
            is_fraud=scored.is_flagged,
            analyst="integration_analyst",
            original_score=scored.fraud_score,
            original_decision=decision,
        )

        # Step 3: Export for retraining
        training_data = store.export_training_data()
        assert len(training_data) == 1
        assert training_data[0]["transaction_id"] == str(txn.transaction_id)
        assert "is_fraud" in training_data[0]
