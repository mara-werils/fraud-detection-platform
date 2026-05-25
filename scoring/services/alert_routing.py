"""Configurable alert routing and escalation engine.

Routes fraud alerts to the correct channels and teams based on priority rules,
supports duplicate suppression within configurable windows, and drives a
full escalation chain (L1 analyst → L4 executive) when alerts go unacknowledged.

Routing rules can be authored in code or loaded from YAML config files so that
on-call teams can tune thresholds without a redeploy.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any

import structlog
import yaml
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

ALERTS_ROUTED_TOTAL = Counter(
    "alerts_routed_total",
    "Total number of alerts that were routed to at least one channel",
    ["priority", "channel"],
)

ALERTS_ESCALATED_TOTAL = Counter(
    "alerts_escalated_total",
    "Total number of alert escalations fired",
    ["from_level", "to_level"],
)

ALERT_ROUTING_LATENCY = Histogram(
    "alert_routing_latency_seconds",
    "End-to-end latency of the routing decision (seconds)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

ALERTS_SUPPRESSED_TOTAL = Counter(
    "alerts_suppressed_total",
    "Alerts dropped due to duplicate suppression",
    ["priority"],
)

# ── Enums ─────────────────────────────────────────────────────────────────────


class AlertChannel(StrEnum):
    SLACK = "slack"
    TELEGRAM = "telegram"
    EMAIL = "email"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SMS = "sms"


class AlertPriority(StrEnum):
    P1_CRITICAL = "P1_CRITICAL"
    P2_HIGH = "P2_HIGH"
    P3_MEDIUM = "P3_MEDIUM"
    P4_LOW = "P4_LOW"
    P5_INFO = "P5_INFO"

    @property
    def numeric(self) -> int:
        """Lower number == higher urgency (P1 = 1)."""
        return int(self.value[1])

    def escalate_one(self) -> AlertPriority:
        """Return the next higher priority, capped at P1_CRITICAL."""
        next_num = max(1, self.numeric - 1)
        mapping = {
            1: AlertPriority.P1_CRITICAL,
            2: AlertPriority.P2_HIGH,
            3: AlertPriority.P3_MEDIUM,
            4: AlertPriority.P4_LOW,
            5: AlertPriority.P5_INFO,
        }
        return mapping[next_num]


class EscalationLevel(StrEnum):
    L1_ANALYST = "L1_ANALYST"
    L2_SENIOR = "L2_SENIOR"
    L3_MANAGER = "L3_MANAGER"
    L4_EXECUTIVE = "L4_EXECUTIVE"


class ConditionOperator(StrEnum):
    GTE = ">="
    LTE = "<="
    GT = ">"
    LT = "<"
    EQ = "=="
    NEQ = "!="
    CONTAINS = "contains"
    IN = "in"


# ── Pydantic models ───────────────────────────────────────────────────────────


class Condition(BaseModel):
    """A single field-level predicate used inside a RoutingRule."""

    field: str
    operator: ConditionOperator
    value: Any


class RoutingRule(BaseModel):
    """Declarative rule that maps alert attributes to channels and priority."""

    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    enabled: bool = True
    conditions: list[Condition] = Field(default_factory=list)
    channels: list[AlertChannel] = Field(default_factory=list)
    priority: AlertPriority = AlertPriority.P3_MEDIUM
    escalation_after_minutes: int = 30
    assignee_group: str = "fraud-ops"
    suppress_duplicates_minutes: int = 15


class EscalationStep(BaseModel):
    """One step in an escalation ladder."""

    level: EscalationLevel
    notify_after_minutes: int
    channels: list[AlertChannel] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)


class EscalationPolicy(BaseModel):
    """Ordered sequence of escalation steps for a given alert type."""

    policy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    levels: list[EscalationStep] = Field(default_factory=list)


class AlertEvent(BaseModel):
    """Inbound alert event produced by the scoring pipeline."""

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transaction_id: str
    score: float = Field(ge=0.0, le=1.0)
    amount: float = 0.0
    currency: str = "USD"
    entity_id: str = ""
    entity_type: str = ""
    sanctioned_entity: bool = False
    rule_triggers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_field(self, field_name: str) -> Any:
        """Retrieve a top-level or metadata field by name."""
        if hasattr(self, field_name):
            return getattr(self, field_name)
        return self.metadata.get(field_name)


class RoutingDecision(BaseModel):
    """Result produced by AlertRoutingEngine.route_alert()."""

    alert_id: str
    matched_rules: list[str]       # rule IDs that fired
    channels: list[AlertChannel]
    priority: AlertPriority
    assignee_group: str
    suppressed: bool = False
    escalation_policy_id: str | None = None
    routed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EscalationResult(BaseModel):
    """Result of a single escalation step invocation."""

    alert_id: str
    previous_level: EscalationLevel | None
    new_level: EscalationLevel | None
    channels_notified: list[AlertChannel]
    assignees_notified: list[str]
    escalated: bool
    reason: str = ""


# ── Suppression state (in-memory; can be swapped for Redis) ──────────────────


class _SuppressionStore:
    """Tracks the first-seen timestamp for each (alert fingerprint) key."""

    def __init__(self) -> None:
        self._store: dict[str, float] = {}
        self._lock = Lock()

    def is_suppressed(self, fingerprint: str, window_minutes: int) -> bool:
        window_seconds = window_minutes * 60
        now = time.monotonic()
        with self._lock:
            first_seen = self._store.get(fingerprint)
            if first_seen is None:
                self._store[fingerprint] = now
                return False
            if now - first_seen < window_seconds:
                return True
            # Window expired — reset so the alert is fresh.
            self._store[fingerprint] = now
            return False

    def evict_expired(self, max_age_seconds: int = 3600) -> int:
        """Remove stale entries to prevent unbounded memory growth."""
        now = time.monotonic()
        with self._lock:
            stale = [k for k, ts in self._store.items() if now - ts > max_age_seconds]
            for k in stale:
                del self._store[k]
        return len(stale)


# ── Escalation tracker ────────────────────────────────────────────────────────


class _EscalationTracker:
    """Tracks which escalation step each active alert is currently at."""

    def __init__(self) -> None:
        self._state: dict[str, int] = {}   # alert_id -> current step index
        self._lock = Lock()

    def current_step_index(self, alert_id: str) -> int:
        with self._lock:
            return self._state.get(alert_id, 0)

    def advance(self, alert_id: str) -> int:
        with self._lock:
            self._state[alert_id] = self._state.get(alert_id, 0) + 1
            return self._state[alert_id]

    def clear(self, alert_id: str) -> None:
        with self._lock:
            self._state.pop(alert_id, None)


# ── Default rules factory ─────────────────────────────────────────────────────


def _build_default_rules() -> list[RoutingRule]:
    """Return the platform's out-of-the-box routing rules."""
    return [
        RoutingRule(
            rule_id="default-p1-score",
            name="Critical score threshold (≥0.95)",
            description="Route to PagerDuty, Slack, and Email for extremely high-confidence fraud.",
            conditions=[Condition(field="score", operator=ConditionOperator.GTE, value=0.95)],
            channels=[AlertChannel.PAGERDUTY, AlertChannel.SLACK, AlertChannel.EMAIL],
            priority=AlertPriority.P1_CRITICAL,
            escalation_after_minutes=5,
            assignee_group="fraud-oncall",
            suppress_duplicates_minutes=5,
        ),
        RoutingRule(
            rule_id="default-p2-score",
            name="High score threshold (≥0.80)",
            description="Route to Slack and Email for high-confidence fraud.",
            conditions=[Condition(field="score", operator=ConditionOperator.GTE, value=0.80)],
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            priority=AlertPriority.P2_HIGH,
            escalation_after_minutes=15,
            assignee_group="fraud-ops",
            suppress_duplicates_minutes=10,
        ),
        RoutingRule(
            rule_id="default-p3-score",
            name="Medium score threshold (≥0.50)",
            description="Route to Slack for moderate-confidence fraud requiring analyst review.",
            conditions=[Condition(field="score", operator=ConditionOperator.GTE, value=0.50)],
            channels=[AlertChannel.SLACK],
            priority=AlertPriority.P3_MEDIUM,
            escalation_after_minutes=60,
            assignee_group="fraud-ops",
            suppress_duplicates_minutes=15,
        ),
        RoutingRule(
            rule_id="default-high-amount",
            name="High-value transaction escalation (≥$50,000)",
            description="Escalate priority by one level when transaction amount is very large.",
            conditions=[Condition(field="amount", operator=ConditionOperator.GTE, value=50_000)],
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            priority=AlertPriority.P2_HIGH,   # bumped further at decision time
            escalation_after_minutes=10,
            assignee_group="fraud-oncall",
            suppress_duplicates_minutes=10,
        ),
        RoutingRule(
            rule_id="default-sanctions",
            name="Sanctioned entity match",
            description="Immediate P1 alert whenever a transaction touches a sanctioned entity.",
            conditions=[Condition(field="sanctioned_entity", operator=ConditionOperator.EQ, value=True)],
            channels=[AlertChannel.PAGERDUTY, AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.SMS],
            priority=AlertPriority.P1_CRITICAL,
            escalation_after_minutes=2,
            assignee_group="compliance-oncall",
            suppress_duplicates_minutes=0,
        ),
    ]


def _build_default_escalation_policy() -> EscalationPolicy:
    return EscalationPolicy(
        policy_id="default-escalation",
        name="Default four-tier escalation ladder",
        levels=[
            EscalationStep(
                level=EscalationLevel.L1_ANALYST,
                notify_after_minutes=0,
                channels=[AlertChannel.SLACK],
                assignees=["analyst-on-duty"],
            ),
            EscalationStep(
                level=EscalationLevel.L2_SENIOR,
                notify_after_minutes=15,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                assignees=["senior-fraud-analyst"],
            ),
            EscalationStep(
                level=EscalationLevel.L3_MANAGER,
                notify_after_minutes=30,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.SMS],
                assignees=["fraud-manager"],
            ),
            EscalationStep(
                level=EscalationLevel.L4_EXECUTIVE,
                notify_after_minutes=60,
                channels=[AlertChannel.PAGERDUTY, AlertChannel.EMAIL, AlertChannel.SMS],
                assignees=["vp-risk", "ciso"],
            ),
        ],
    )


# ── Main engine ───────────────────────────────────────────────────────────────


class AlertRoutingEngine:
    """Configurable alert routing and escalation engine.

    Thread-safe: all mutable state is protected by a lock.

    Typical usage::

        engine = AlertRoutingEngine()
        engine.load_rules_from_yaml("/etc/fraud/routing_rules.yaml")

        decision = engine.route_alert(alert_event)
        # decision.channels, decision.priority, decision.suppressed …

        result = engine.escalate(alert_event.alert_id)
    """

    def __init__(
        self,
        load_defaults: bool = True,
        escalation_policy: EscalationPolicy | None = None,
    ) -> None:
        self._rules: dict[str, RoutingRule] = {}
        self._lock = Lock()
        self._suppression = _SuppressionStore()
        self._escalation_tracker = _EscalationTracker()
        self._escalation_policy: EscalationPolicy = (
            escalation_policy or _build_default_escalation_policy()
        )

        # Stats counters (in addition to Prometheus)
        self._stats: dict[str, int] = defaultdict(int)

        if load_defaults:
            for rule in _build_default_rules():
                self.add_rule(rule)

        logger.info(
            "alert_routing_engine_initialised",
            rule_count=len(self._rules),
            escalation_policy=self._escalation_policy.name,
        )

    # ── Rule management ───────────────────────────────────────────────────────

    def add_rule(self, rule: RoutingRule) -> None:
        """Register or replace a routing rule."""
        with self._lock:
            self._rules[rule.rule_id] = rule
        logger.debug("routing_rule_added", rule_id=rule.rule_id, name=rule.name)

    def remove_rule(self, rule_id: str) -> None:
        """Remove a routing rule by ID. No-op if not found."""
        with self._lock:
            removed = self._rules.pop(rule_id, None)
        if removed:
            logger.debug("routing_rule_removed", rule_id=rule_id)
        else:
            logger.warning("routing_rule_not_found", rule_id=rule_id)

    def list_rules(self) -> list[RoutingRule]:
        with self._lock:
            return list(self._rules.values())

    # ── Condition evaluation ──────────────────────────────────────────────────

    def evaluate_conditions(self, alert: AlertEvent, conditions: list[Condition]) -> bool:
        """Return True if ALL conditions are satisfied (AND semantics)."""
        for cond in conditions:
            actual = alert.get_field(cond.field)
            if actual is None:
                return False
            try:
                matched = self._eval_single(actual, cond.operator, cond.value)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "condition_eval_error",
                    field=cond.field,
                    operator=cond.operator,
                    error=str(exc),
                )
                return False
            if not matched:
                return False
        return True

    @staticmethod
    def _eval_single(actual: Any, operator: ConditionOperator, expected: Any) -> bool:
        match operator:
            case ConditionOperator.GTE:
                return float(actual) >= float(expected)
            case ConditionOperator.LTE:
                return float(actual) <= float(expected)
            case ConditionOperator.GT:
                return float(actual) > float(expected)
            case ConditionOperator.LT:
                return float(actual) < float(expected)
            case ConditionOperator.EQ:
                return actual == expected
            case ConditionOperator.NEQ:
                return actual != expected
            case ConditionOperator.CONTAINS:
                return str(expected) in str(actual)
            case ConditionOperator.IN:
                return actual in expected
            case _:
                raise ValueError(f"Unknown operator: {operator}")

    # ── Suppression ───────────────────────────────────────────────────────────

    def suppress_check(self, alert: AlertEvent, window_minutes: int = 15) -> bool:
        """Return True if this alert should be suppressed as a duplicate.

        The fingerprint is (transaction_id, score-bucket, entity_id) so that
        two alerts from the same transaction with very different scores are still
        treated as distinct events.
        """
        if window_minutes <= 0:
            return False
        score_bucket = round(alert.score, 1)
        fingerprint = f"{alert.transaction_id}:{score_bucket}:{alert.entity_id}"
        return self._suppression.is_suppressed(fingerprint, window_minutes)

    # ── Core routing ──────────────────────────────────────────────────────────

    def route_alert(self, alert: AlertEvent) -> RoutingDecision:
        """Evaluate all enabled rules and return a consolidated routing decision."""
        start = time.perf_counter()

        matched_rules: list[RoutingRule] = []

        with self._lock:
            rules_snapshot = list(self._rules.values())

        for rule in rules_snapshot:
            if not rule.enabled:
                continue
            if self.evaluate_conditions(alert, rule.conditions):
                matched_rules.append(rule)

        if not matched_rules:
            # Fallback: route to Slack at P5 so nothing is silently dropped.
            decision = RoutingDecision(
                alert_id=alert.alert_id,
                matched_rules=[],
                channels=[AlertChannel.SLACK],
                priority=AlertPriority.P5_INFO,
                assignee_group="fraud-ops",
            )
            logger.info(
                "alert_no_rule_match",
                alert_id=alert.alert_id,
                score=alert.score,
            )
            self._stats["no_match"] += 1
            return decision

        # Determine highest priority (lowest numeric value) across all matches.
        best_priority = min(matched_rules, key=lambda r: r.priority.numeric).priority

        # Bump priority one level if a high-amount rule matched alongside other rules.
        high_amount_rule_ids = {"default-high-amount"}
        high_amount_matched = any(r.rule_id in high_amount_rule_ids for r in matched_rules)
        non_amount_matched = any(r.rule_id not in high_amount_rule_ids for r in matched_rules)
        if high_amount_matched and non_amount_matched:
            best_priority = best_priority.escalate_one()

        # Union of all channels from matched rules.
        channel_set: set[AlertChannel] = set()
        for rule in matched_rules:
            channel_set.update(rule.channels)

        # Assignee group from highest-priority rule.
        top_rule = min(matched_rules, key=lambda r: r.priority.numeric)

        # Duplicate suppression using top rule's window.
        suppressed = self.suppress_check(alert, top_rule.suppress_duplicates_minutes)
        if suppressed:
            ALERTS_SUPPRESSED_TOTAL.labels(priority=best_priority.value).inc()
            self._stats["suppressed"] += 1
            logger.info(
                "alert_suppressed",
                alert_id=alert.alert_id,
                transaction_id=alert.transaction_id,
                priority=best_priority,
            )
            return RoutingDecision(
                alert_id=alert.alert_id,
                matched_rules=[r.rule_id for r in matched_rules],
                channels=list(channel_set),
                priority=best_priority,
                assignee_group=top_rule.assignee_group,
                suppressed=True,
            )

        for channel in channel_set:
            ALERTS_ROUTED_TOTAL.labels(
                priority=best_priority.value, channel=channel.value
            ).inc()

        self._stats["routed"] += 1

        elapsed = time.perf_counter() - start
        ALERT_ROUTING_LATENCY.observe(elapsed)

        decision = RoutingDecision(
            alert_id=alert.alert_id,
            matched_rules=[r.rule_id for r in matched_rules],
            channels=sorted(channel_set),
            priority=best_priority,
            assignee_group=top_rule.assignee_group,
            escalation_policy_id=self._escalation_policy.policy_id,
        )

        logger.info(
            "alert_routed",
            alert_id=alert.alert_id,
            transaction_id=alert.transaction_id,
            priority=best_priority,
            channels=[c.value for c in decision.channels],
            matched_rules=decision.matched_rules,
            latency_ms=round(elapsed * 1000, 2),
        )
        return decision

    # ── Escalation ────────────────────────────────────────────────────────────

    def get_escalation_chain(self, alert_id: str) -> list[EscalationStep]:
        """Return escalation steps starting from the current position."""
        current_idx = self._escalation_tracker.current_step_index(alert_id)
        return self._escalation_policy.levels[current_idx:]

    def escalate(self, alert_id: str) -> EscalationResult:
        """Fire the next escalation step for the given alert.

        Each call advances the tracker by one level. Returns an EscalationResult
        with ``escalated=False`` once the top of the ladder is reached.
        """
        current_idx = self._escalation_tracker.current_step_index(alert_id)
        levels = self._escalation_policy.levels

        if current_idx >= len(levels):
            logger.warning("escalation_exhausted", alert_id=alert_id)
            return EscalationResult(
                alert_id=alert_id,
                previous_level=levels[-1].level if levels else None,
                new_level=None,
                channels_notified=[],
                assignees_notified=[],
                escalated=False,
                reason="Escalation ladder exhausted — no further levels available.",
            )

        current_step = levels[current_idx]
        previous_level = levels[current_idx - 1].level if current_idx > 0 else None

        new_idx = self._escalation_tracker.advance(alert_id)
        next_level = levels[new_idx].level if new_idx < len(levels) else None

        ALERTS_ESCALATED_TOTAL.labels(
            from_level=previous_level.value if previous_level else "none",
            to_level=current_step.level.value,
        ).inc()

        self._stats["escalated"] += 1

        logger.info(
            "alert_escalated",
            alert_id=alert_id,
            from_level=previous_level,
            to_level=current_step.level,
            channels=[c.value for c in current_step.channels],
            assignees=current_step.assignees,
        )

        return EscalationResult(
            alert_id=alert_id,
            previous_level=previous_level,
            new_level=current_step.level,
            channels_notified=current_step.channels,
            assignees_notified=current_step.assignees,
            escalated=True,
        )

    def acknowledge(self, alert_id: str) -> None:
        """Mark an alert as acknowledged, resetting its escalation state."""
        self._escalation_tracker.clear(alert_id)
        logger.info("alert_acknowledged", alert_id=alert_id)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_routing_stats(self) -> dict[str, Any]:
        """Return current engine statistics."""
        with self._lock:
            rule_count = len(self._rules)
            enabled_count = sum(1 for r in self._rules.values() if r.enabled)
        return {
            "rule_count": rule_count,
            "enabled_rules": enabled_count,
            "alerts_routed": self._stats["routed"],
            "alerts_suppressed": self._stats["suppressed"],
            "alerts_escalated": self._stats["escalated"],
            "no_rule_match": self._stats["no_match"],
            "escalation_policy": self._escalation_policy.name,
            "escalation_levels": len(self._escalation_policy.levels),
        }

    # ── YAML persistence ──────────────────────────────────────────────────────

    def load_rules_from_yaml(self, path: str) -> int:
        """Load routing rules from a YAML file.  Returns the number of rules loaded.

        Expected YAML structure::

            rules:
              - rule_id: my-rule
                name: "High velocity"
                conditions:
                  - field: score
                    operator: ">="
                    value: 0.9
                channels: [slack, pagerduty]
                priority: P1_CRITICAL
                escalation_after_minutes: 10
                assignee_group: fraud-oncall
                suppress_duplicates_minutes: 5
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Routing rules file not found: {path}")

        with file_path.open() as fh:
            raw = yaml.safe_load(fh)

        rules_data: list[dict[str, Any]] = raw.get("rules", [])
        loaded = 0
        for entry in rules_data:
            try:
                rule = RoutingRule.model_validate(entry)
                self.add_rule(rule)
                loaded += 1
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "routing_rule_yaml_parse_error",
                    entry=entry,
                    error=str(exc),
                )

        logger.info("routing_rules_loaded_from_yaml", path=path, count=loaded)
        return loaded

    def export_rules_to_yaml(self, path: str) -> None:
        """Persist the current rule set to a YAML file."""
        with self._lock:
            rules_data = [r.model_dump(mode="json") for r in self._rules.values()]

        output = {"rules": rules_data}
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w") as fh:
            yaml.safe_dump(output, fh, default_flow_style=False, sort_keys=False)

        logger.info("routing_rules_exported_to_yaml", path=path, count=len(rules_data))
