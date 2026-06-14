"""Case investigation workflow automation.

Provides configurable workflow states, auto-assignment based on analyst
workload and expertise, SLA tracking with breach notifications, priority
scoring, case notes/timeline, bulk operations, case templates, and
similarity search for related cases.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Lock
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Workflow states
# ---------------------------------------------------------------------------

class WorkflowState(StrEnum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


# Valid transitions between states
VALID_TRANSITIONS: dict[WorkflowState, set[WorkflowState]] = {
    WorkflowState.OPEN: {WorkflowState.INVESTIGATING, WorkflowState.ESCALATED, WorkflowState.CLOSED},
    WorkflowState.INVESTIGATING: {WorkflowState.ESCALATED, WorkflowState.RESOLVED, WorkflowState.CLOSED},
    WorkflowState.ESCALATED: {WorkflowState.INVESTIGATING, WorkflowState.RESOLVED, WorkflowState.CLOSED},
    WorkflowState.RESOLVED: {WorkflowState.CLOSED, WorkflowState.INVESTIGATING},
    WorkflowState.CLOSED: set(),
}


# ---------------------------------------------------------------------------
# Priority helpers
# ---------------------------------------------------------------------------

class CasePriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


PRIORITY_ORDER: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class CustomerTier(StrEnum):
    STANDARD = "standard"
    PREMIUM = "premium"
    VIP = "vip"


TIER_WEIGHT: dict[str, float] = {"standard": 1.0, "premium": 1.5, "vip": 2.0}


def compute_priority(
    amount: float,
    risk_score: float,
    customer_tier: str = "standard",
) -> CasePriority:
    """Score case priority from transaction amount, risk score, and customer tier.

    The composite score is ``(normalised_amount + risk_score) * tier_weight``.
    Thresholds:
        >= 1.6  -> critical
        >= 1.1  -> high
        >= 0.6  -> medium
        else    -> low
    """
    tier_w = TIER_WEIGHT.get(customer_tier, 1.0)
    # Normalise amount with a soft cap at 10 000
    norm_amount = min(amount / 10_000.0, 1.0)
    composite = (norm_amount + risk_score) * tier_w

    if composite >= 1.6:
        return CasePriority.CRITICAL
    if composite >= 1.1:
        return CasePriority.HIGH
    if composite >= 0.6:
        return CasePriority.MEDIUM
    return CasePriority.LOW


# ---------------------------------------------------------------------------
# SLA configuration
# ---------------------------------------------------------------------------

DEFAULT_SLA_HOURS: dict[CasePriority, float] = {
    CasePriority.CRITICAL: 1.0,
    CasePriority.HIGH: 4.0,
    CasePriority.MEDIUM: 24.0,
    CasePriority.LOW: 72.0,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TimelineEntry(BaseModel):
    """Single entry on a case timeline."""

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    actor: str
    details: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CaseNote(BaseModel):
    """A note attached to a case."""

    note_id: str = Field(default_factory=lambda: str(uuid4()))
    author: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkflowCase(BaseModel):
    """A fraud investigation case with full workflow tracking."""

    case_id: str = Field(default_factory=lambda: str(uuid4()))
    transaction_id: str
    user_id: str
    fraud_type: str = ""
    amount: float = 0.0
    risk_score: float = 0.0
    customer_tier: str = "standard"
    state: WorkflowState = WorkflowState.OPEN
    priority: CasePriority = CasePriority.MEDIUM
    assigned_to: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    sla_deadline: datetime | None = None
    sla_breached: bool = False
    notes: list[CaseNote] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    template_id: str | None = None


# ---------------------------------------------------------------------------
# Analyst model
# ---------------------------------------------------------------------------

class Analyst(BaseModel):
    """Represents an analyst available for case assignment."""

    analyst_id: str
    name: str
    expertise: list[str] = Field(default_factory=list)
    max_cases: int = 10
    active: bool = True


# ---------------------------------------------------------------------------
# Case template
# ---------------------------------------------------------------------------

class CaseTemplate(BaseModel):
    """Reusable template for common fraud types."""

    template_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    fraud_type: str
    default_tags: list[str] = Field(default_factory=list)
    initial_notes: str = ""
    checklist: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SLA breach notification
# ---------------------------------------------------------------------------

class SLABreachNotification(BaseModel):
    """Notification emitted when a case breaches its SLA."""

    case_id: str
    assigned_to: str | None
    priority: str
    deadline: datetime
    breached_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# CaseWorkflowEngine
# ---------------------------------------------------------------------------

class CaseWorkflowEngine:
    """Orchestrates case lifecycle, assignment, SLAs, and search.

    All state is held in-memory; a production deployment would persist to a
    database.
    """

    def __init__(
        self,
        sla_hours: dict[CasePriority, float] | None = None,
    ) -> None:
        self._lock = Lock()
        self._cases: dict[str, WorkflowCase] = {}
        self._analysts: dict[str, Analyst] = {}
        self._templates: dict[str, CaseTemplate] = {}
        self._sla_hours: dict[CasePriority, float] = sla_hours or dict(DEFAULT_SLA_HOURS)
        self._sla_notifications: list[SLABreachNotification] = []

    # -- Analyst management --------------------------------------------------

    def register_analyst(self, analyst: Analyst) -> None:
        """Register an analyst for auto-assignment."""
        with self._lock:
            self._analysts[analyst.analyst_id] = analyst

    def remove_analyst(self, analyst_id: str) -> bool:
        with self._lock:
            return self._analysts.pop(analyst_id, None) is not None

    def get_analyst_workload(self, analyst_id: str) -> int:
        """Return number of open/investigating/escalated cases for an analyst."""
        active_states = {WorkflowState.OPEN, WorkflowState.INVESTIGATING, WorkflowState.ESCALATED}
        count = 0
        for case in self._cases.values():
            if case.assigned_to == analyst_id and case.state in active_states:
                count += 1
        return count

    # -- Template management -------------------------------------------------

    def register_template(self, template: CaseTemplate) -> None:
        with self._lock:
            self._templates[template.template_id] = template

    def get_template(self, template_id: str) -> CaseTemplate | None:
        return self._templates.get(template_id)

    def list_templates(self) -> list[CaseTemplate]:
        return list(self._templates.values())

    # -- Case creation -------------------------------------------------------

    def create_case(
        self,
        transaction_id: str,
        user_id: str,
        amount: float = 0.0,
        risk_score: float = 0.0,
        customer_tier: str = "standard",
        fraud_type: str = "",
        tags: list[str] | None = None,
        template_id: str | None = None,
        actor: str = "system",
    ) -> WorkflowCase:
        """Create a new case, auto-computing priority and SLA deadline."""
        priority = compute_priority(amount, risk_score, customer_tier)
        sla_h = self._sla_hours.get(priority, 24.0)
        now = datetime.now(UTC)
        deadline = now + timedelta(hours=sla_h)

        case = WorkflowCase(
            transaction_id=transaction_id,
            user_id=user_id,
            amount=amount,
            risk_score=risk_score,
            customer_tier=customer_tier,
            fraud_type=fraud_type,
            priority=priority,
            sla_deadline=deadline,
            tags=list(tags or []),
            template_id=template_id,
            created_at=now,
            updated_at=now,
        )

        # Apply template defaults if provided
        if template_id:
            tmpl = self._templates.get(template_id)
            if tmpl:
                case.fraud_type = case.fraud_type or tmpl.fraud_type
                case.tags = list(set(case.tags) | set(tmpl.default_tags))
                if tmpl.initial_notes:
                    case.notes.append(CaseNote(author="system", content=tmpl.initial_notes))
                case.template_id = template_id

        case.timeline.append(TimelineEntry(
            event_type="created",
            actor=actor,
            details=f"Case created with priority {priority.value}",
        ))

        with self._lock:
            self._cases[case.case_id] = case

        logger.info("workflow_case_created", case_id=case.case_id, priority=priority.value)
        return case

    # -- State transitions ---------------------------------------------------

    def transition(
        self,
        case_id: str,
        new_state: str,
        actor: str = "system",
        notes: str | None = None,
    ) -> WorkflowCase:
        """Move a case to *new_state*, validating the transition.

        Raises ``ValueError`` if the transition is not allowed.
        """
        target = WorkflowState(new_state)

        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise KeyError(f"Case {case_id} not found")

            allowed = VALID_TRANSITIONS.get(case.state, set())
            if target not in allowed:
                raise ValueError(
                    f"Transition from {case.state.value} to {target.value} is not allowed"
                )

            old_state = case.state
            case.state = target
            case.updated_at = datetime.now(UTC)

            if target in (WorkflowState.RESOLVED, WorkflowState.CLOSED):
                case.resolved_at = datetime.now(UTC)

            case.timeline.append(TimelineEntry(
                event_type="state_changed",
                actor=actor,
                details=f"State changed from {old_state.value} to {target.value}",
            ))

            if notes:
                case.notes.append(CaseNote(author=actor, content=notes))

        logger.info(
            "workflow_state_changed",
            case_id=case_id,
            old=old_state.value,
            new=target.value,
        )
        return case

    # -- Assignment ----------------------------------------------------------

    def auto_assign(self, case_id: str) -> WorkflowCase:
        """Assign a case to the best-fit analyst based on workload and expertise.

        Selection strategy:
        1. Filter active analysts below their max case limit.
        2. Prefer analysts whose expertise matches the case fraud_type.
        3. Among matches, pick the one with the lowest current workload.
        4. If no expertise match, pick the analyst with lowest workload.
        """
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise KeyError(f"Case {case_id} not found")

            candidates: list[tuple[str, int, bool]] = []
            for a in self._analysts.values():
                if not a.active:
                    continue
                load = self.get_analyst_workload(a.analyst_id)
                if load >= a.max_cases:
                    continue
                has_expertise = case.fraud_type in a.expertise if case.fraud_type else False
                candidates.append((a.analyst_id, load, has_expertise))

            if not candidates:
                raise RuntimeError("No available analyst for assignment")

            # Sort: expertise match first (True > False → reverse), then lowest load
            candidates.sort(key=lambda c: (not c[2], c[1]))
            chosen_id = candidates[0][0]

            case.assigned_to = chosen_id
            case.updated_at = datetime.now(UTC)
            case.timeline.append(TimelineEntry(
                event_type="auto_assigned",
                actor="system",
                details=f"Auto-assigned to {chosen_id}",
            ))

        logger.info("workflow_auto_assigned", case_id=case_id, analyst=chosen_id)
        return case

    def assign(self, case_id: str, analyst_id: str, actor: str = "system") -> WorkflowCase:
        """Manually assign a case to an analyst."""
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise KeyError(f"Case {case_id} not found")

            old = case.assigned_to
            case.assigned_to = analyst_id
            case.updated_at = datetime.now(UTC)
            case.timeline.append(TimelineEntry(
                event_type="assigned",
                actor=actor,
                details=f"Reassigned from {old} to {analyst_id}",
            ))

        return case

    # -- SLA tracking --------------------------------------------------------

    def check_sla_breaches(self, now: datetime | None = None) -> list[SLABreachNotification]:
        """Scan all active cases and emit notifications for SLA breaches.

        Returns newly detected breaches only.
        """
        now = now or datetime.now(UTC)
        active_states = {WorkflowState.OPEN, WorkflowState.INVESTIGATING, WorkflowState.ESCALATED}
        new_breaches: list[SLABreachNotification] = []

        with self._lock:
            for case in self._cases.values():
                if case.state not in active_states:
                    continue
                if case.sla_breached:
                    continue
                if case.sla_deadline and now >= case.sla_deadline:
                    case.sla_breached = True
                    notif = SLABreachNotification(
                        case_id=case.case_id,
                        assigned_to=case.assigned_to,
                        priority=case.priority.value,
                        deadline=case.sla_deadline,
                        breached_at=now,
                    )
                    new_breaches.append(notif)
                    self._sla_notifications.append(notif)
                    case.timeline.append(TimelineEntry(
                        event_type="sla_breached",
                        actor="system",
                        details=f"SLA breached at {now.isoformat()}",
                        timestamp=now,
                    ))

        if new_breaches:
            logger.warning("sla_breaches_detected", count=len(new_breaches))
        return new_breaches

    def get_sla_notifications(self) -> list[SLABreachNotification]:
        return list(self._sla_notifications)

    # -- Notes and timeline --------------------------------------------------

    def add_note(self, case_id: str, author: str, content: str) -> WorkflowCase:
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise KeyError(f"Case {case_id} not found")
            case.notes.append(CaseNote(author=author, content=content))
            case.updated_at = datetime.now(UTC)
            case.timeline.append(TimelineEntry(
                event_type="note_added",
                actor=author,
                details=f"Note added by {author}",
            ))
        return case

    def get_timeline(self, case_id: str) -> list[TimelineEntry]:
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(f"Case {case_id} not found")
        return list(case.timeline)

    # -- Bulk operations -----------------------------------------------------

    def bulk_assign(
        self, case_ids: list[str], analyst_id: str, actor: str = "system",
    ) -> tuple[int, int]:
        """Assign multiple cases. Returns (success_count, not_found_count)."""
        success = 0
        not_found = 0
        for cid in case_ids:
            try:
                self.assign(cid, analyst_id, actor)
                success += 1
            except KeyError:
                not_found += 1
        return success, not_found

    def bulk_escalate(
        self, case_ids: list[str], actor: str = "system", notes: str | None = None,
    ) -> tuple[int, int]:
        """Escalate multiple cases. Returns (success_count, failure_count)."""
        success = 0
        failed = 0
        for cid in case_ids:
            try:
                self.transition(cid, WorkflowState.ESCALATED.value, actor, notes)
                success += 1
            except (KeyError, ValueError):
                failed += 1
        return success, failed

    def bulk_close(
        self, case_ids: list[str], actor: str = "system", notes: str | None = None,
    ) -> tuple[int, int]:
        """Close multiple cases. Returns (success_count, failure_count)."""
        success = 0
        failed = 0
        for cid in case_ids:
            try:
                self.transition(cid, WorkflowState.CLOSED.value, actor, notes)
                success += 1
            except (KeyError, ValueError):
                failed += 1
        return success, failed

    # -- Case similarity search ----------------------------------------------

    def find_similar_cases(
        self,
        case_id: str,
        max_results: int = 5,
    ) -> list[tuple[str, float]]:
        """Find cases similar to the given case.

        Similarity is computed from shared attributes:
        - Same user_id (weight 0.4)
        - Same fraud_type (weight 0.3)
        - Overlapping tags (weight 0.15)
        - Close amount (weight 0.15, exponential decay)

        Returns list of (case_id, similarity_score) sorted by descending score.
        """
        ref = self._cases.get(case_id)
        if ref is None:
            raise KeyError(f"Case {case_id} not found")

        scores: list[tuple[str, float]] = []
        for other in self._cases.values():
            if other.case_id == case_id:
                continue
            score = 0.0

            # User match
            if other.user_id == ref.user_id:
                score += 0.4

            # Fraud type match
            if ref.fraud_type and other.fraud_type == ref.fraud_type:
                score += 0.3

            # Tag overlap (Jaccard-ish)
            if ref.tags and other.tags:
                ref_set = set(ref.tags)
                other_set = set(other.tags)
                union = ref_set | other_set
                if union:
                    score += 0.15 * len(ref_set & other_set) / len(union)

            # Amount proximity (exponential decay, half-life at 1000 difference)
            if ref.amount > 0 and other.amount > 0:
                diff = abs(ref.amount - other.amount)
                score += 0.15 * math.exp(-diff / 1000.0)

            if score > 0:
                scores.append((other.case_id, round(score, 4)))

        scores.sort(key=lambda s: s[1], reverse=True)
        return scores[:max_results]

    # -- Query helpers -------------------------------------------------------

    def get_case(self, case_id: str) -> WorkflowCase | None:
        return self._cases.get(case_id)

    def list_cases(
        self,
        state: str | None = None,
        priority: str | None = None,
        assigned_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[WorkflowCase], int]:
        cases = list(self._cases.values())
        if state:
            cases = [c for c in cases if c.state.value == state]
        if priority:
            cases = [c for c in cases if c.priority.value == priority]
        if assigned_to:
            cases = [c for c in cases if c.assigned_to == assigned_to]

        cases.sort(key=lambda c: (PRIORITY_ORDER.get(c.priority.value, 4), -c.created_at.timestamp()))
        total = len(cases)
        return cases[offset: offset + limit], total
