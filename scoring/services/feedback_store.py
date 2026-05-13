"""Feedback store for analyst decisions on scored transactions.

Collects analyst feedback (is_fraud/legitimate) to create labeled
training data for model retraining and performance monitoring.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from threading import Lock
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class FeedbackEntry(BaseModel):
    """A single feedback entry from an analyst."""

    feedback_id: str = Field(default_factory=lambda: str(uuid4()))
    transaction_id: str
    is_fraud: bool
    analyst: str = "unknown"
    notes: str | None = None
    original_score: float | None = None
    original_decision: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackStats(BaseModel):
    """Aggregate feedback statistics."""

    total_feedback: int
    confirmed_fraud: int
    confirmed_legitimate: int
    false_positive_rate: float
    false_negative_rate: float
    agreement_rate: float
    by_analyst: dict[str, int]


class FeedbackStore:
    """In-memory feedback store for analyst decisions.

    Tracks analyst feedback on scored transactions to:
    1. Measure model accuracy (false positive/negative rates)
    2. Generate labeled training data for retraining
    3. Track analyst activity and agreement rates
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: list[FeedbackEntry] = []
        self._by_transaction: dict[str, FeedbackEntry] = {}
        self._by_analyst: dict[str, list[int]] = defaultdict(list)

    def add(
        self,
        transaction_id: str,
        is_fraud: bool,
        analyst: str = "unknown",
        notes: str | None = None,
        original_score: float | None = None,
        original_decision: str | None = None,
    ) -> FeedbackEntry:
        """Record analyst feedback for a transaction.

        Args:
            transaction_id: The transaction being reviewed.
            is_fraud: Analyst's determination (True = fraud).
            analyst: Name of the reviewing analyst.
            notes: Optional notes about the decision.
            original_score: The model's fraud score for this transaction.
            original_decision: The model's decision (BLOCK/REVIEW/ALLOW).

        Returns:
            The created FeedbackEntry.
        """
        with self._lock:
            entry = FeedbackEntry(
                transaction_id=transaction_id,
                is_fraud=is_fraud,
                analyst=analyst,
                notes=notes,
                original_score=original_score,
                original_decision=original_decision,
            )

            idx = len(self._entries)
            self._entries.append(entry)
            self._by_transaction[transaction_id] = entry
            self._by_analyst[analyst].append(idx)

            logger.info(
                "feedback_recorded",
                transaction_id=transaction_id,
                is_fraud=is_fraud,
                analyst=analyst,
                original_score=original_score,
            )

            return entry

    def get(self, transaction_id: str) -> FeedbackEntry | None:
        """Get feedback for a specific transaction."""
        return self._by_transaction.get(transaction_id)

    def list_entries(
        self,
        analyst: str | None = None,
        is_fraud: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[FeedbackEntry], int]:
        """List feedback entries with filtering."""
        with self._lock:
            entries = list(self._entries)

        if analyst:
            entries = [e for e in entries if e.analyst == analyst]
        if is_fraud is not None:
            entries = [e for e in entries if e.is_fraud == is_fraud]

        entries.sort(key=lambda e: e.created_at, reverse=True)
        total = len(entries)
        return entries[offset : offset + limit], total

    def stats(self) -> FeedbackStats:
        """Compute feedback statistics including model accuracy metrics."""
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return FeedbackStats(
                total_feedback=0,
                confirmed_fraud=0,
                confirmed_legitimate=0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                agreement_rate=0.0,
                by_analyst={},
            )

        confirmed_fraud = sum(1 for e in entries if e.is_fraud)
        confirmed_legit = sum(1 for e in entries if not e.is_fraud)

        # False positive: model flagged, analyst says legitimate
        false_positives = sum(
            1 for e in entries
            if not e.is_fraud and e.original_decision in ("BLOCK", "REVIEW")
        )
        # False negative: model allowed, analyst says fraud
        false_negatives = sum(
            1 for e in entries
            if e.is_fraud and e.original_decision == "ALLOW"
        )

        flagged = sum(
            1 for e in entries
            if e.original_decision in ("BLOCK", "REVIEW")
        )
        allowed = sum(
            1 for e in entries
            if e.original_decision == "ALLOW"
        )

        fp_rate = false_positives / flagged if flagged > 0 else 0.0
        fn_rate = false_negatives / allowed if allowed > 0 else 0.0

        # Agreement rate: model decision matches analyst
        agreed = sum(
            1 for e in entries
            if (e.is_fraud and e.original_decision in ("BLOCK", "REVIEW"))
            or (not e.is_fraud and e.original_decision == "ALLOW")
        )
        agreement_rate = agreed / len(entries) if entries else 0.0

        by_analyst: dict[str, int] = defaultdict(int)
        for e in entries:
            by_analyst[e.analyst] += 1

        return FeedbackStats(
            total_feedback=len(entries),
            confirmed_fraud=confirmed_fraud,
            confirmed_legitimate=confirmed_legit,
            false_positive_rate=round(fp_rate, 4),
            false_negative_rate=round(fn_rate, 4),
            agreement_rate=round(agreement_rate, 4),
            by_analyst=dict(by_analyst),
        )

    def export_training_data(self) -> list[dict[str, Any]]:
        """Export feedback as labeled training data.

        Returns:
            List of dicts with transaction_id and is_fraud label.
        """
        with self._lock:
            return [
                {
                    "transaction_id": e.transaction_id,
                    "is_fraud": e.is_fraud,
                    "original_score": e.original_score,
                    "analyst": e.analyst,
                    "created_at": e.created_at.isoformat(),
                }
                for e in self._entries
            ]
