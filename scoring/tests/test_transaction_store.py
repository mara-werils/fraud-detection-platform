"""Tests for TransactionStore enhancements."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from scoring.services.transaction_store import TransactionStore
from shared.schemas import ScoredTransaction, TransactionType


def _make_scored_txn(fraud_score: float = 0.3, is_flagged: bool = False, scored_at: datetime | None = None) -> ScoredTransaction:
    return ScoredTransaction(
        transaction_id=uuid4(),
        user_id=uuid4(),
        amount=Decimal("100.00"),
        currency="USD",
        transaction_type=TransactionType.PURCHASE,
        timestamp=datetime.now(UTC),
        fraud_score=fraud_score,
        model_version="test-v1",
        scoring_latency_ms=5.0,
        scored_at=scored_at or datetime.now(UTC),
        is_flagged=is_flagged,
    )


class TestTransactionStore:
    def test_add_and_get(self):
        store = TransactionStore()
        txn = _make_scored_txn()
        store.add(txn)
        assert store.size == 1
        assert store.get(str(txn.transaction_id)) is not None

    def test_clear(self):
        store = TransactionStore()
        for _ in range(5):
            store.add(_make_scored_txn())
        assert store.size == 5
        store.clear()
        assert store.size == 0

    def test_get_flagged(self):
        store = TransactionStore()
        store.add(_make_scored_txn(fraud_score=0.9, is_flagged=True))
        store.add(_make_scored_txn(fraud_score=0.2, is_flagged=False))
        store.add(_make_scored_txn(fraud_score=0.85, is_flagged=True))
        flagged = store.get_flagged()
        assert len(flagged) == 2
        assert all(t.is_flagged for t in flagged)

    def test_evict_older_than(self):
        store = TransactionStore()
        old_time = datetime.now(UTC) - timedelta(hours=2)
        recent_time = datetime.now(UTC)

        store.add(_make_scored_txn(scored_at=old_time))
        store.add(_make_scored_txn(scored_at=old_time))
        store.add(_make_scored_txn(scored_at=recent_time))

        evicted = store.evict_older_than(timedelta(hours=1))
        assert evicted == 2
        assert store.size == 1

    def test_stats_empty(self):
        store = TransactionStore()
        stats = store.stats()
        assert stats["total"] == 0
