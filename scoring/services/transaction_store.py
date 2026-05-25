"""In-memory transaction store for scored transactions.

In production, this would be backed by ClickHouse or PostgreSQL.
The in-memory store is suitable for development and demonstration.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from threading import Lock
from typing import Any

import structlog

from shared.schemas import ScoredTransaction

logger = structlog.get_logger(__name__)

MAX_STORE_SIZE = 100_000


class TransactionStore:
    """Thread-safe in-memory store for scored transactions.

    Supports filtering by score range, user, time range, and flagged status.
    Maintains indexes for efficient querying.
    """

    def __init__(self, max_size: int = MAX_STORE_SIZE) -> None:
        self._max_size = max_size
        self._lock = Lock()
        self._transactions: list[ScoredTransaction] = []
        self._by_user: dict[str, list[int]] = defaultdict(list)
        self._by_txn_id: dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self._transactions)

    def add(self, txn: ScoredTransaction) -> None:
        """Add a scored transaction to the store."""
        with self._lock:
            if len(self._transactions) >= self._max_size:
                self._evict_oldest(self._max_size // 10)

            idx = len(self._transactions)
            self._transactions.append(txn)
            self._by_user[str(txn.user_id)].append(idx)
            self._by_txn_id[str(txn.transaction_id)] = idx

    def get(self, transaction_id: str) -> ScoredTransaction | None:
        """Get a transaction by ID."""
        idx = self._by_txn_id.get(transaction_id)
        if idx is not None and idx < len(self._transactions):
            return self._transactions[idx]
        return None

    def search(
        self,
        *,
        user_id: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        is_flagged: bool | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        decision: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ScoredTransaction], int]:
        """Search transactions with filtering.

        Args:
            user_id: Filter by user ID.
            min_score: Minimum fraud score.
            max_score: Maximum fraud score.
            is_flagged: Filter by flagged status.
            start_time: Start of time range.
            end_time: End of time range.
            decision: Filter by decision (BLOCK/REVIEW/ALLOW).
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            Tuple of (matching transactions, total count).
        """
        with self._lock:
            # Determine candidate indices
            if user_id:
                candidates = [
                    self._transactions[i]
                    for i in self._by_user.get(user_id, [])
                    if i < len(self._transactions)
                ]
            else:
                candidates = list(self._transactions)

        # Apply filters
        results: list[ScoredTransaction] = []
        for txn in candidates:
            if min_score is not None and txn.fraud_score < min_score:
                continue
            if max_score is not None and txn.fraud_score > max_score:
                continue
            if is_flagged is not None and txn.is_flagged != is_flagged:
                continue
            if start_time is not None and txn.timestamp < start_time:
                continue
            if end_time is not None and txn.timestamp > end_time:
                continue
            if decision is not None:
                txn_decision = self._get_decision(txn.fraud_score)
                if txn_decision.lower() != decision.lower():
                    continue
            results.append(txn)

        # Sort by timestamp descending (newest first)
        results.sort(key=lambda t: t.scored_at, reverse=True)

        total = len(results)
        paginated = results[offset : offset + limit]

        return paginated, total

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics."""
        with self._lock:
            if not self._transactions:
                return {
                    "total": 0,
                    "flagged": 0,
                    "avg_score": 0.0,
                    "max_score": 0.0,
                    "unique_users": 0,
                }

            scores = [t.fraud_score for t in self._transactions]
            return {
                "total": len(self._transactions),
                "flagged": sum(1 for t in self._transactions if t.is_flagged),
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "unique_users": len(self._by_user),
            }

    def _evict_oldest(self, count: int) -> None:
        """Remove the oldest N transactions."""
        self._transactions = self._transactions[count:]
        # Rebuild indexes
        self._by_user.clear()
        self._by_txn_id.clear()
        for i, txn in enumerate(self._transactions):
            self._by_user[str(txn.user_id)].append(i)
            self._by_txn_id[str(txn.transaction_id)] = i

    @staticmethod
    def _get_decision(score: float) -> str:
        if score >= 0.8:
            return "BLOCK"
        if score >= 0.5:
            return "REVIEW"
        return "ALLOW"
