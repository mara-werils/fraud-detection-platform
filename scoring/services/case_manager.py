"""Case management service for fraud investigation workflows.

Provides CRUD operations for fraud cases with status tracking,
assignment, audit trails, and priority management.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class CaseStatus(StrEnum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    ESCALATED = "escalated"
    RESOLVED_FRAUD = "resolved_fraud"
    RESOLVED_LEGITIMATE = "resolved_legitimate"
    CLOSED = "closed"


class CasePriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CaseNote(BaseModel):
    """A note on a fraud case."""

    note_id: str = Field(default_factory=lambda: str(uuid4()))
    author: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CaseEvent(BaseModel):
    """An audit event on a case."""

    event_type: str
    actor: str
    details: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FraudCase(BaseModel):
    """A fraud investigation case."""

    case_id: str = Field(default_factory=lambda: str(uuid4()))
    transaction_id: str
    user_id: str
    fraud_score: float = 0.0
    amount: float = 0.0
    currency: str = "USD"
    status: CaseStatus = CaseStatus.OPEN
    priority: CasePriority = CasePriority.MEDIUM
    assigned_to: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    notes: list[CaseNote] = Field(default_factory=list)
    events: list[CaseEvent] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    is_fraud: bool | None = None  # Final determination


class CaseManager:
    """In-memory case management system.

    In production, this would be backed by a database. The in-memory
    implementation is suitable for development and demonstration.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._cases: dict[str, FraudCase] = {}
        self._by_transaction: dict[str, str] = {}
        self._by_user: dict[str, list[str]] = defaultdict(list)
        self._by_status: dict[str, list[str]] = defaultdict(list)

    def create_case(
        self,
        transaction_id: str,
        user_id: str,
        fraud_score: float = 0.0,
        amount: float = 0.0,
        currency: str = "USD",
        priority: str = "medium",
        assigned_to: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
    ) -> FraudCase:
        """Create a new fraud investigation case.

        Args:
            transaction_id: Associated transaction ID.
            user_id: User being investigated.
            fraud_score: Original fraud score.
            amount: Transaction amount.
            currency: Transaction currency.
            priority: Case priority level.
            assigned_to: Analyst to assign.
            notes: Initial investigation notes.
            tags: Case tags for categorization.

        Returns:
            The created FraudCase.
        """
        with self._lock:
            # Check if a case already exists for this transaction
            if transaction_id in self._by_transaction:
                existing_id = self._by_transaction[transaction_id]
                return self._cases[existing_id]

            case = FraudCase(
                transaction_id=transaction_id,
                user_id=user_id,
                fraud_score=fraud_score,
                amount=amount,
                currency=currency,
                priority=CasePriority(priority),
                assigned_to=assigned_to,
                tags=tags or [],
            )

            if notes:
                case.notes.append(CaseNote(
                    author=assigned_to or "system",
                    content=notes,
                ))

            case.events.append(CaseEvent(
                event_type="created",
                actor=assigned_to or "system",
                details=f"Case created with priority {priority}",
            ))

            self._cases[case.case_id] = case
            self._by_transaction[transaction_id] = case.case_id
            self._by_user[user_id].append(case.case_id)
            self._by_status[case.status.value].append(case.case_id)

            logger.info(
                "case_created",
                case_id=case.case_id,
                transaction_id=transaction_id,
                priority=priority,
            )

            return case

    def get_case(self, case_id: str) -> FraudCase | None:
        """Get a case by ID."""
        return self._cases.get(case_id)

    def get_case_by_transaction(self, transaction_id: str) -> FraudCase | None:
        """Get a case by transaction ID."""
        case_id = self._by_transaction.get(transaction_id)
        if case_id:
            return self._cases.get(case_id)
        return None

    def update_status(
        self,
        case_id: str,
        status: str,
        actor: str = "system",
        notes: str | None = None,
    ) -> FraudCase | None:
        """Update a case's status.

        Args:
            case_id: Case ID to update.
            status: New status value.
            actor: Who is making the change.
            notes: Optional notes about the status change.

        Returns:
            Updated FraudCase or None if not found.
        """
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return None

            old_status = case.status.value

            # Remove from old status index
            if case_id in self._by_status.get(old_status, []):
                self._by_status[old_status].remove(case_id)

            case.status = CaseStatus(status)
            case.updated_at = datetime.now(UTC)

            # Track resolution time
            if status in ("resolved_fraud", "resolved_legitimate", "closed"):
                case.resolved_at = datetime.now(UTC)
                if status == "resolved_fraud":
                    case.is_fraud = True
                elif status == "resolved_legitimate":
                    case.is_fraud = False

            # Add to new status index
            self._by_status[status].append(case_id)

            # Record event
            case.events.append(CaseEvent(
                event_type="status_changed",
                actor=actor,
                details=f"Status changed from {old_status} to {status}",
            ))

            if notes:
                case.notes.append(CaseNote(author=actor, content=notes))

            logger.info(
                "case_status_updated",
                case_id=case_id,
                old_status=old_status,
                new_status=status,
                actor=actor,
            )

            return case

    def assign(self, case_id: str, analyst: str, actor: str = "system") -> FraudCase | None:
        """Assign a case to an analyst."""
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return None

            old_assignee = case.assigned_to
            case.assigned_to = analyst
            case.updated_at = datetime.now(UTC)

            case.events.append(CaseEvent(
                event_type="assigned",
                actor=actor,
                details=f"Assigned from {old_assignee} to {analyst}",
            ))

            return case

    def add_note(self, case_id: str, author: str, content: str) -> FraudCase | None:
        """Add a note to a case."""
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return None

            case.notes.append(CaseNote(author=author, content=content))
            case.updated_at = datetime.now(UTC)
            return case

    def list_cases(
        self,
        status: str | None = None,
        priority: str | None = None,
        assigned_to: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[FraudCase], int]:
        """List cases with optional filtering.

        Returns:
            Tuple of (cases, total_count).
        """
        with self._lock:
            candidates = list(self._cases.values())

        # Apply filters
        if status:
            candidates = [c for c in candidates if c.status.value == status]
        if priority:
            candidates = [c for c in candidates if c.priority.value == priority]
        if assigned_to:
            candidates = [c for c in candidates if c.assigned_to == assigned_to]
        if user_id:
            candidates = [c for c in candidates if c.user_id == user_id]

        # Sort by priority (critical first) then by created_at (newest first)
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        candidates.sort(
            key=lambda c: (priority_order.get(c.priority.value, 4), -c.created_at.timestamp())
        )

        total = len(candidates)
        return candidates[offset : offset + limit], total

    def bulk_update_status(
        self,
        case_ids: list[str],
        status: str,
        actor: str = "system",
    ) -> tuple[int, int]:
        """Update status for multiple cases at once.

        Args:
            case_ids: List of case IDs to update.
            status: New status value for all cases.
            actor: Who is making the change.

        Returns:
            Tuple of (success_count, not_found_count).
        """
        success = 0
        not_found = 0
        for case_id in case_ids:
            result = self.update_status(case_id, status, actor)
            if result is not None:
                success += 1
            else:
                not_found += 1
        logger.info(
            "bulk_status_update",
            total=len(case_ids),
            success=success,
            not_found=not_found,
            status=status,
            actor=actor,
        )
        return success, not_found

    def stats(self) -> dict[str, Any]:
        """Return case statistics."""
        with self._lock:
            total = len(self._cases)
            if total == 0:
                return {"total": 0, "by_status": {}, "by_priority": {}, "avg_resolution_hours": 0}

            by_status: dict[str, int] = defaultdict(int)
            by_priority: dict[str, int] = defaultdict(int)
            resolution_times: list[float] = []

            for case in self._cases.values():
                by_status[case.status.value] += 1
                by_priority[case.priority.value] += 1

                if case.resolved_at and case.created_at:
                    hours = (case.resolved_at - case.created_at).total_seconds() / 3600
                    resolution_times.append(hours)

            avg_resolution = (
                sum(resolution_times) / len(resolution_times)
                if resolution_times
                else 0.0
            )

            return {
                "total": total,
                "by_status": dict(by_status),
                "by_priority": dict(by_priority),
                "avg_resolution_hours": round(avg_resolution, 2),
            }


__all__ = ["CaseEvent", "CaseManager", "CaseNote", "CasePriority", "CaseStatus", "FraudCase"]
