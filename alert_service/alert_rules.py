"""Flexible alerting rules engine for the fraud detection platform.

Supports threshold-based, anomaly-based, trend-based, and composite alert
rules with severity levels, cooldown/suppression, escalation chains, and
alert grouping to prevent alert fatigue.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


class Severity(StrEnum):
    """Alert severity levels ordered from lowest to highest."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.CRITICAL: 2,
    Severity.EMERGENCY: 3,
}


def severity_gte(a: Severity, b: Severity) -> bool:
    """Return True if severity *a* is greater than or equal to *b*."""
    return _SEVERITY_ORDER[a] >= _SEVERITY_ORDER[b]


# ---------------------------------------------------------------------------
# Alert data classes
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """Represents a fired alert."""

    alert_id: UUID = field(default_factory=uuid4)
    rule_name: str = ""
    severity: Severity = Severity.INFO
    message: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    acknowledged: bool = False
    acknowledged_at: float | None = None
    group_key: str | None = None
    escalation_level: int = 0


@dataclass
class EscalationStep:
    """A single step in an escalation chain.

    Attributes:
        delay_seconds: Time in seconds to wait before escalating.
        target: Identifier for the escalation target (e.g. team name, channel).
        severity_override: Optional severity to set when this step triggers.
    """

    delay_seconds: float
    target: str
    severity_override: Severity | None = None


# ---------------------------------------------------------------------------
# Evaluation context  -- the data passed into each rule
# ---------------------------------------------------------------------------


@dataclass
class EvalContext:
    """Data provided to alert rules for evaluation.

    Attributes:
        fraud_rate: Current fraud rate as a fraction (0.0 - 1.0).
        transaction_count: Number of transactions in the current window.
        historical_rates: Recent fraud rate observations (newest last).
        historical_counts: Recent transaction count observations (newest last).
        extras: Arbitrary key/value pairs rules may inspect.
        current_time: Monotonic timestamp (overridable for testing).
    """

    fraud_rate: float = 0.0
    transaction_count: int = 0
    historical_rates: list[float] = field(default_factory=list)
    historical_counts: list[int] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    current_time: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Base rule
# ---------------------------------------------------------------------------


class AlertRule(ABC):
    """Abstract base class for all alert rules."""

    def __init__(
        self,
        name: str,
        severity: Severity = Severity.WARNING,
        cooldown_seconds: float = 0.0,
        group_key: str | None = None,
    ) -> None:
        self.name = name
        self.severity = severity
        self.cooldown_seconds = cooldown_seconds
        self.group_key = group_key
        self._last_fired: float | None = None

    # --- public API ---

    def evaluate(self, ctx: EvalContext) -> Alert | None:
        """Evaluate the rule against the given context.

        Returns an Alert if the rule fires, otherwise None.
        Respects cooldown / suppression automatically.
        """
        if self._is_suppressed(ctx.current_time):
            return None
        if not self._condition(ctx):
            return None
        self._last_fired = ctx.current_time
        return Alert(
            rule_name=self.name,
            severity=self.severity,
            message=self._build_message(ctx),
            context=self._build_context(ctx),
            created_at=ctx.current_time,
            group_key=self.group_key or self.name,
        )

    def reset_cooldown(self) -> None:
        """Manually reset the cooldown timer."""
        self._last_fired = None

    # --- internal helpers ---

    def _is_suppressed(self, now: float) -> bool:
        if self.cooldown_seconds <= 0 or self._last_fired is None:
            return False
        return (now - self._last_fired) < self.cooldown_seconds

    @abstractmethod
    def _condition(self, ctx: EvalContext) -> bool:
        """Return True when the rule should fire."""
        ...

    def _build_message(self, ctx: EvalContext) -> str:  # noqa: ARG002
        return f"Rule '{self.name}' triggered"

    def _build_context(self, ctx: EvalContext) -> dict[str, Any]:  # noqa: ARG002
        return {}


# ---------------------------------------------------------------------------
# Concrete rules
# ---------------------------------------------------------------------------


class ThresholdRule(AlertRule):
    """Fires when the fraud rate exceeds a fixed threshold.

    Example: alert if fraud rate > 5%.
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        severity: Severity = Severity.WARNING,
        cooldown_seconds: float = 0.0,
        group_key: str | None = None,
    ) -> None:
        super().__init__(name, severity, cooldown_seconds, group_key)
        self.threshold = threshold

    def _condition(self, ctx: EvalContext) -> bool:
        return ctx.fraud_rate > self.threshold

    def _build_message(self, ctx: EvalContext) -> str:
        return (
            f"Fraud rate {ctx.fraud_rate:.2%} exceeds threshold "
            f"{self.threshold:.2%}"
        )

    def _build_context(self, ctx: EvalContext) -> dict[str, Any]:
        return {"fraud_rate": ctx.fraud_rate, "threshold": self.threshold}


class AnomalyRule(AlertRule):
    """Fires on a sudden spike in transaction volume.

    Computes the mean of *historical_counts* and fires when the current
    transaction_count exceeds ``mean * spike_factor``.
    """

    def __init__(
        self,
        name: str,
        spike_factor: float = 3.0,
        min_history: int = 3,
        severity: Severity = Severity.WARNING,
        cooldown_seconds: float = 0.0,
        group_key: str | None = None,
    ) -> None:
        super().__init__(name, severity, cooldown_seconds, group_key)
        self.spike_factor = spike_factor
        self.min_history = min_history

    def _condition(self, ctx: EvalContext) -> bool:
        if len(ctx.historical_counts) < self.min_history:
            return False
        mean = sum(ctx.historical_counts) / len(ctx.historical_counts)
        if mean == 0:
            return ctx.transaction_count > 0
        return ctx.transaction_count > mean * self.spike_factor

    def _build_message(self, ctx: EvalContext) -> str:
        mean = sum(ctx.historical_counts) / max(len(ctx.historical_counts), 1)
        return (
            f"Transaction spike detected: {ctx.transaction_count} txns "
            f"vs historical mean {mean:.1f} (factor {self.spike_factor}x)"
        )

    def _build_context(self, ctx: EvalContext) -> dict[str, Any]:
        mean = sum(ctx.historical_counts) / max(len(ctx.historical_counts), 1)
        return {
            "transaction_count": ctx.transaction_count,
            "historical_mean": mean,
            "spike_factor": self.spike_factor,
        }


class TrendRule(AlertRule):
    """Fires when the fraud rate is trending upward over a time window.

    Uses a simple linear-slope check: the fraud rate must increase by at
    least ``min_increase`` across the last ``window_size`` observations in
    ``historical_rates``.
    """

    def __init__(
        self,
        name: str,
        window_size: int = 5,
        min_increase: float = 0.02,
        severity: Severity = Severity.WARNING,
        cooldown_seconds: float = 0.0,
        group_key: str | None = None,
    ) -> None:
        super().__init__(name, severity, cooldown_seconds, group_key)
        self.window_size = window_size
        self.min_increase = min_increase

    def _condition(self, ctx: EvalContext) -> bool:
        rates = ctx.historical_rates
        if len(rates) < self.window_size:
            return False
        window = rates[-self.window_size:]
        # simple slope: difference between last and first in the window
        increase = window[-1] - window[0]
        # also verify that the trend is monotonically non-decreasing
        monotonic = all(window[i] <= window[i + 1] for i in range(len(window) - 1))
        return increase >= self.min_increase and monotonic

    def _build_message(self, ctx: EvalContext) -> str:
        window = ctx.historical_rates[-self.window_size:]
        increase = window[-1] - window[0]
        return (
            f"Fraud rate trending up: +{increase:.2%} over last "
            f"{self.window_size} observations"
        )

    def _build_context(self, ctx: EvalContext) -> dict[str, Any]:
        window = ctx.historical_rates[-self.window_size:]
        return {
            "window": window,
            "increase": window[-1] - window[0],
            "min_increase": self.min_increase,
        }


class CompositeRule(AlertRule):
    """Fires only when **all** sub-rules would fire (logical AND).

    Sub-rule cooldowns are bypassed; only the composite-level cooldown
    and severity are used.
    """

    def __init__(
        self,
        name: str,
        rules: list[AlertRule],
        severity: Severity = Severity.CRITICAL,
        cooldown_seconds: float = 0.0,
        group_key: str | None = None,
    ) -> None:
        super().__init__(name, severity, cooldown_seconds, group_key)
        self.rules = rules

    def _condition(self, ctx: EvalContext) -> bool:
        return all(rule._condition(ctx) for rule in self.rules)

    def _build_message(self, ctx: EvalContext) -> str:
        triggered = [r.name for r in self.rules if r._condition(ctx)]
        return (
            f"Composite rule '{self.name}' triggered: "
            f"all {len(triggered)} sub-rules fired"
        )

    def _build_context(self, ctx: EvalContext) -> dict[str, Any]:
        return {"sub_rules": [r.name for r in self.rules]}


# ---------------------------------------------------------------------------
# Escalation chain
# ---------------------------------------------------------------------------


class EscalationChain:
    """Manages time-based escalation for unacknowledged alerts.

    If an alert is not acknowledged within *delay_seconds* of a step,
    the alert is escalated to the next target in the chain.
    """

    def __init__(self, steps: list[EscalationStep]) -> None:
        if not steps:
            raise ValueError("EscalationChain requires at least one step")
        self.steps = steps

    def evaluate(self, alert: Alert, now: float | None = None) -> EscalationStep | None:
        """Return the next escalation step if the alert is overdue.

        Returns None if the alert is acknowledged or all steps have been
        exhausted.
        """
        if alert.acknowledged:
            return None
        now = now if now is not None else time.monotonic()
        elapsed = now - alert.created_at
        for idx, step in enumerate(self.steps):
            if idx <= alert.escalation_level - 1:
                continue  # already escalated past this step
            if elapsed >= step.delay_seconds:
                alert.escalation_level = idx + 1
                if step.severity_override is not None:
                    alert.severity = step.severity_override
                return step
        return None


# ---------------------------------------------------------------------------
# Alert grouping
# ---------------------------------------------------------------------------


class AlertGroup:
    """Groups related alerts to prevent alert fatigue.

    Alerts sharing the same ``group_key`` are collected into a single
    group.  The group exposes a summary instead of individual alerts when
    the count exceeds ``max_individual``.
    """

    def __init__(self, group_key: str, max_individual: int = 3) -> None:
        self.group_key = group_key
        self.max_individual = max_individual
        self._alerts: list[Alert] = []

    @property
    def alerts(self) -> list[Alert]:
        return list(self._alerts)

    @property
    def count(self) -> int:
        return len(self._alerts)

    @property
    def is_grouped(self) -> bool:
        """True when there are more alerts than *max_individual*."""
        return len(self._alerts) > self.max_individual

    @property
    def highest_severity(self) -> Severity:
        if not self._alerts:
            return Severity.INFO
        return max(self._alerts, key=lambda a: _SEVERITY_ORDER[a.severity]).severity

    def add(self, alert: Alert) -> None:
        self._alerts.append(alert)

    def summary(self) -> str:
        if not self._alerts:
            return f"[{self.group_key}] No alerts"
        if not self.is_grouped:
            msgs = [a.message for a in self._alerts]
            return f"[{self.group_key}] {len(msgs)} alert(s):\n" + "\n".join(
                f"  - {m}" for m in msgs
            )
        return (
            f"[{self.group_key}] {len(self._alerts)} alerts grouped "
            f"(highest severity: {self.highest_severity.value})"
        )


# ---------------------------------------------------------------------------
# Rules engine
# ---------------------------------------------------------------------------


class RulesEngine:
    """Orchestrates rule evaluation, grouping, and escalation.

    Usage::

        engine = RulesEngine()
        engine.add_rule(ThresholdRule("high_fraud", threshold=0.05))
        engine.set_escalation("high_fraud", EscalationChain([...]))

        alerts = engine.evaluate(ctx)
        engine.check_escalations()
    """

    def __init__(self, group_max_individual: int = 3) -> None:
        self._rules: list[AlertRule] = []
        self._escalations: dict[str, EscalationChain] = {}
        self._pending_alerts: list[Alert] = []
        self._groups: dict[str, AlertGroup] = {}
        self._group_max_individual = group_max_individual

    # --- configuration ---

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def set_escalation(self, rule_name: str, chain: EscalationChain) -> None:
        self._escalations[rule_name] = chain

    # --- evaluation ---

    def evaluate(self, ctx: EvalContext) -> list[Alert]:
        """Evaluate all rules and return fired alerts."""
        fired: list[Alert] = []
        for rule in self._rules:
            alert = rule.evaluate(ctx)
            if alert is not None:
                fired.append(alert)
                self._pending_alerts.append(alert)
                # group
                gk = alert.group_key or alert.rule_name
                if gk not in self._groups:
                    self._groups[gk] = AlertGroup(
                        gk, max_individual=self._group_max_individual
                    )
                self._groups[gk].add(alert)
        return fired

    def check_escalations(self, now: float | None = None) -> list[tuple[Alert, EscalationStep]]:
        """Check pending alerts for escalation and return escalated pairs."""
        escalated: list[tuple[Alert, EscalationStep]] = []
        for alert in self._pending_alerts:
            chain = self._escalations.get(alert.rule_name)
            if chain is None:
                continue
            step = chain.evaluate(alert, now=now)
            if step is not None:
                escalated.append((alert, step))
        return escalated

    def acknowledge(self, alert_id: UUID, now: float | None = None) -> bool:
        """Acknowledge an alert by ID. Returns True if found."""
        for alert in self._pending_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = now if now is not None else time.monotonic()
                return True
        return False

    def get_group(self, group_key: str) -> AlertGroup | None:
        return self._groups.get(group_key)

    @property
    def pending_alerts(self) -> list[Alert]:
        return [a for a in self._pending_alerts if not a.acknowledged]

    def clear_acknowledged(self) -> int:
        """Remove acknowledged alerts from the pending list. Returns count removed."""
        before = len(self._pending_alerts)
        self._pending_alerts = [a for a in self._pending_alerts if not a.acknowledged]
        return before - len(self._pending_alerts)
