"""Tests for the alerting rules engine."""

from __future__ import annotations

import pytest

from alert_service.alert_rules import (
    Alert,
    AlertGroup,
    AnomalyRule,
    CompositeRule,
    EscalationChain,
    EscalationStep,
    EvalContext,
    RulesEngine,
    Severity,
    ThresholdRule,
    TrendRule,
    severity_gte,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = 1_000_000.0


def _ctx(
    *,
    fraud_rate: float = 0.0,
    transaction_count: int = 100,
    historical_rates: list[float] | None = None,
    historical_counts: list[int] | None = None,
    current_time: float = _BASE_TIME,
    extras: dict | None = None,
) -> EvalContext:
    return EvalContext(
        fraud_rate=fraud_rate,
        transaction_count=transaction_count,
        historical_rates=historical_rates or [],
        historical_counts=historical_counts or [],
        current_time=current_time,
        extras=extras or {},
    )


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


class TestSeverity:
    def test_ordering(self) -> None:
        assert severity_gte(Severity.EMERGENCY, Severity.INFO)
        assert severity_gte(Severity.CRITICAL, Severity.WARNING)
        assert severity_gte(Severity.WARNING, Severity.WARNING)
        assert not severity_gte(Severity.INFO, Severity.WARNING)

    def test_str_values(self) -> None:
        assert Severity.INFO.value == "info"
        assert Severity.EMERGENCY.value == "emergency"


# ---------------------------------------------------------------------------
# ThresholdRule
# ---------------------------------------------------------------------------


class TestThresholdRule:
    def test_fires_above_threshold(self) -> None:
        rule = ThresholdRule("high_fraud", threshold=0.05, severity=Severity.WARNING)
        alert = rule.evaluate(_ctx(fraud_rate=0.06))
        assert alert is not None
        assert alert.severity == Severity.WARNING
        assert "5.00%" in alert.message or "threshold" in alert.message

    def test_does_not_fire_below_threshold(self) -> None:
        rule = ThresholdRule("high_fraud", threshold=0.05)
        assert rule.evaluate(_ctx(fraud_rate=0.04)) is None

    def test_does_not_fire_at_exact_threshold(self) -> None:
        rule = ThresholdRule("high_fraud", threshold=0.05)
        assert rule.evaluate(_ctx(fraud_rate=0.05)) is None

    def test_context_contains_threshold(self) -> None:
        rule = ThresholdRule("high_fraud", threshold=0.05)
        alert = rule.evaluate(_ctx(fraud_rate=0.10))
        assert alert is not None
        assert alert.context["threshold"] == 0.05
        assert alert.context["fraud_rate"] == 0.10


# ---------------------------------------------------------------------------
# AnomalyRule
# ---------------------------------------------------------------------------


class TestAnomalyRule:
    def test_fires_on_spike(self) -> None:
        rule = AnomalyRule("spike", spike_factor=2.0, min_history=3)
        ctx = _ctx(
            transaction_count=300,
            historical_counts=[100, 90, 110],
        )
        alert = rule.evaluate(ctx)
        assert alert is not None
        assert "spike" in alert.message.lower()

    def test_no_fire_normal_volume(self) -> None:
        rule = AnomalyRule("spike", spike_factor=2.0, min_history=3)
        ctx = _ctx(
            transaction_count=150,
            historical_counts=[100, 90, 110],
        )
        assert rule.evaluate(ctx) is None

    def test_insufficient_history(self) -> None:
        rule = AnomalyRule("spike", spike_factor=2.0, min_history=3)
        ctx = _ctx(transaction_count=9999, historical_counts=[10, 20])
        assert rule.evaluate(ctx) is None

    def test_zero_mean_with_positive_count(self) -> None:
        rule = AnomalyRule("spike", spike_factor=2.0, min_history=3)
        ctx = _ctx(transaction_count=1, historical_counts=[0, 0, 0])
        alert = rule.evaluate(ctx)
        assert alert is not None

    def test_zero_mean_with_zero_count(self) -> None:
        rule = AnomalyRule("spike", spike_factor=2.0, min_history=3)
        ctx = _ctx(transaction_count=0, historical_counts=[0, 0, 0])
        assert rule.evaluate(ctx) is None


# ---------------------------------------------------------------------------
# TrendRule
# ---------------------------------------------------------------------------


class TestTrendRule:
    def test_fires_on_upward_trend(self) -> None:
        rule = TrendRule("trend_up", window_size=5, min_increase=0.02)
        rates = [0.01, 0.02, 0.03, 0.04, 0.05]
        alert = rule.evaluate(_ctx(historical_rates=rates))
        assert alert is not None
        assert "trending" in alert.message.lower()

    def test_no_fire_flat_trend(self) -> None:
        rule = TrendRule("trend_up", window_size=5, min_increase=0.02)
        rates = [0.03, 0.03, 0.03, 0.03, 0.03]
        assert rule.evaluate(_ctx(historical_rates=rates)) is None

    def test_no_fire_downward_trend(self) -> None:
        rule = TrendRule("trend_up", window_size=5, min_increase=0.02)
        rates = [0.05, 0.04, 0.03, 0.02, 0.01]
        assert rule.evaluate(_ctx(historical_rates=rates)) is None

    def test_insufficient_window(self) -> None:
        rule = TrendRule("trend_up", window_size=5, min_increase=0.02)
        rates = [0.01, 0.02, 0.03]
        assert rule.evaluate(_ctx(historical_rates=rates)) is None

    def test_non_monotonic_rejected(self) -> None:
        rule = TrendRule("trend_up", window_size=5, min_increase=0.02)
        # overall increase is enough, but there's a dip
        rates = [0.01, 0.03, 0.02, 0.04, 0.05]
        assert rule.evaluate(_ctx(historical_rates=rates)) is None

    def test_uses_last_n_observations(self) -> None:
        rule = TrendRule("trend_up", window_size=3, min_increase=0.02)
        rates = [0.10, 0.01, 0.02, 0.04]  # only last 3 matter
        alert = rule.evaluate(_ctx(historical_rates=rates))
        assert alert is not None


# ---------------------------------------------------------------------------
# CompositeRule
# ---------------------------------------------------------------------------


class TestCompositeRule:
    def test_fires_when_all_sub_rules_fire(self) -> None:
        r1 = ThresholdRule("t1", threshold=0.05)
        r2 = AnomalyRule("a1", spike_factor=2.0, min_history=3)
        composite = CompositeRule("both", rules=[r1, r2], severity=Severity.CRITICAL)
        ctx = _ctx(
            fraud_rate=0.10,
            transaction_count=300,
            historical_counts=[100, 90, 110],
        )
        alert = composite.evaluate(ctx)
        assert alert is not None
        assert alert.severity == Severity.CRITICAL

    def test_does_not_fire_when_one_sub_rule_fails(self) -> None:
        r1 = ThresholdRule("t1", threshold=0.05)
        r2 = AnomalyRule("a1", spike_factor=2.0, min_history=3)
        composite = CompositeRule("both", rules=[r1, r2])
        ctx = _ctx(
            fraud_rate=0.10,
            transaction_count=100,  # not a spike
            historical_counts=[100, 90, 110],
        )
        assert composite.evaluate(ctx) is None

    def test_context_lists_sub_rules(self) -> None:
        r1 = ThresholdRule("t1", threshold=0.01)
        r2 = ThresholdRule("t2", threshold=0.01)
        composite = CompositeRule("both", rules=[r1, r2])
        alert = composite.evaluate(_ctx(fraud_rate=0.10))
        assert alert is not None
        assert set(alert.context["sub_rules"]) == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Cooldown / suppression
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_suppressed_within_cooldown(self) -> None:
        rule = ThresholdRule("cd", threshold=0.01, cooldown_seconds=60.0)
        ctx1 = _ctx(fraud_rate=0.05, current_time=_BASE_TIME)
        assert rule.evaluate(ctx1) is not None

        ctx2 = _ctx(fraud_rate=0.05, current_time=_BASE_TIME + 30)
        assert rule.evaluate(ctx2) is None  # still within cooldown

    def test_fires_after_cooldown(self) -> None:
        rule = ThresholdRule("cd", threshold=0.01, cooldown_seconds=60.0)
        ctx1 = _ctx(fraud_rate=0.05, current_time=_BASE_TIME)
        assert rule.evaluate(ctx1) is not None

        ctx2 = _ctx(fraud_rate=0.05, current_time=_BASE_TIME + 61)
        assert rule.evaluate(ctx2) is not None

    def test_reset_cooldown(self) -> None:
        rule = ThresholdRule("cd", threshold=0.01, cooldown_seconds=600.0)
        rule.evaluate(_ctx(fraud_rate=0.05, current_time=_BASE_TIME))
        rule.reset_cooldown()
        alert = rule.evaluate(_ctx(fraud_rate=0.05, current_time=_BASE_TIME + 1))
        assert alert is not None


# ---------------------------------------------------------------------------
# Escalation chains
# ---------------------------------------------------------------------------


class TestEscalationChain:
    def _make_alert(self, t: float = _BASE_TIME) -> Alert:
        return Alert(rule_name="test", severity=Severity.WARNING, created_at=t)

    def test_escalation_triggers_on_timeout(self) -> None:
        chain = EscalationChain([
            EscalationStep(delay_seconds=60, target="on-call"),
            EscalationStep(delay_seconds=300, target="manager"),
        ])
        alert = self._make_alert()
        # before first step
        assert chain.evaluate(alert, now=_BASE_TIME + 30) is None
        # after first step
        step = chain.evaluate(alert, now=_BASE_TIME + 61)
        assert step is not None
        assert step.target == "on-call"
        assert alert.escalation_level == 1

    def test_second_escalation(self) -> None:
        chain = EscalationChain([
            EscalationStep(delay_seconds=60, target="on-call"),
            EscalationStep(delay_seconds=300, target="manager", severity_override=Severity.EMERGENCY),
        ])
        alert = self._make_alert()
        chain.evaluate(alert, now=_BASE_TIME + 61)  # escalate to level 1
        step = chain.evaluate(alert, now=_BASE_TIME + 301)
        assert step is not None
        assert step.target == "manager"
        assert alert.escalation_level == 2
        assert alert.severity == Severity.EMERGENCY

    def test_acknowledged_stops_escalation(self) -> None:
        chain = EscalationChain([
            EscalationStep(delay_seconds=60, target="on-call"),
        ])
        alert = self._make_alert()
        alert.acknowledged = True
        assert chain.evaluate(alert, now=_BASE_TIME + 600) is None

    def test_empty_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            EscalationChain([])

    def test_severity_override(self) -> None:
        chain = EscalationChain([
            EscalationStep(delay_seconds=10, target="team", severity_override=Severity.CRITICAL),
        ])
        alert = self._make_alert()
        assert alert.severity == Severity.WARNING
        chain.evaluate(alert, now=_BASE_TIME + 11)
        assert alert.severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Alert grouping
# ---------------------------------------------------------------------------


class TestAlertGroup:
    def test_not_grouped_under_threshold(self) -> None:
        group = AlertGroup("fraud_alerts", max_individual=3)
        for _ in range(3):
            group.add(Alert(rule_name="r1", severity=Severity.INFO, message="test"))
        assert not group.is_grouped
        assert group.count == 3

    def test_grouped_above_threshold(self) -> None:
        group = AlertGroup("fraud_alerts", max_individual=3)
        for _ in range(4):
            group.add(Alert(rule_name="r1", severity=Severity.INFO, message="test"))
        assert group.is_grouped
        assert group.count == 4

    def test_highest_severity(self) -> None:
        group = AlertGroup("g")
        group.add(Alert(severity=Severity.INFO))
        group.add(Alert(severity=Severity.CRITICAL))
        group.add(Alert(severity=Severity.WARNING))
        assert group.highest_severity == Severity.CRITICAL

    def test_summary_grouped(self) -> None:
        group = AlertGroup("g", max_individual=1)
        group.add(Alert(severity=Severity.INFO, message="a"))
        group.add(Alert(severity=Severity.WARNING, message="b"))
        summary = group.summary()
        assert "2 alerts grouped" in summary

    def test_summary_individual(self) -> None:
        group = AlertGroup("g", max_individual=5)
        group.add(Alert(severity=Severity.INFO, message="alert one"))
        summary = group.summary()
        assert "alert one" in summary

    def test_empty_summary(self) -> None:
        group = AlertGroup("g")
        assert "No alerts" in group.summary()

    def test_highest_severity_empty(self) -> None:
        group = AlertGroup("g")
        assert group.highest_severity == Severity.INFO


# ---------------------------------------------------------------------------
# RulesEngine integration
# ---------------------------------------------------------------------------


class TestRulesEngine:
    def test_evaluate_returns_fired_alerts(self) -> None:
        engine = RulesEngine()
        engine.add_rule(ThresholdRule("t1", threshold=0.05))
        engine.add_rule(ThresholdRule("t2", threshold=0.20))
        alerts = engine.evaluate(_ctx(fraud_rate=0.10))
        assert len(alerts) == 1
        assert alerts[0].rule_name == "t1"

    def test_pending_alerts_exclude_acknowledged(self) -> None:
        engine = RulesEngine()
        engine.add_rule(ThresholdRule("t1", threshold=0.01))
        alerts = engine.evaluate(_ctx(fraud_rate=0.10))
        assert len(engine.pending_alerts) == 1
        engine.acknowledge(alerts[0].alert_id)
        assert len(engine.pending_alerts) == 0

    def test_acknowledge_returns_false_for_unknown(self) -> None:
        engine = RulesEngine()
        from uuid import uuid4
        assert engine.acknowledge(uuid4()) is False

    def test_clear_acknowledged(self) -> None:
        engine = RulesEngine()
        engine.add_rule(ThresholdRule("t1", threshold=0.01))
        alerts = engine.evaluate(_ctx(fraud_rate=0.10))
        engine.acknowledge(alerts[0].alert_id)
        removed = engine.clear_acknowledged()
        assert removed == 1

    def test_escalation_integration(self) -> None:
        engine = RulesEngine()
        engine.add_rule(ThresholdRule("t1", threshold=0.01))
        engine.set_escalation(
            "t1",
            EscalationChain([
                EscalationStep(delay_seconds=60, target="on-call"),
            ]),
        )
        alerts = engine.evaluate(_ctx(fraud_rate=0.10, current_time=_BASE_TIME))
        # no escalation yet
        assert engine.check_escalations(now=_BASE_TIME + 30) == []
        # after timeout
        escalated = engine.check_escalations(now=_BASE_TIME + 61)
        assert len(escalated) == 1
        assert escalated[0][1].target == "on-call"

    def test_grouping_integration(self) -> None:
        engine = RulesEngine(group_max_individual=2)
        rule = ThresholdRule("t1", threshold=0.01, group_key="fraud")
        engine.add_rule(rule)
        for i in range(3):
            engine.evaluate(_ctx(fraud_rate=0.10, current_time=_BASE_TIME + i * 1000))
        group = engine.get_group("fraud")
        assert group is not None
        assert group.count == 3
        assert group.is_grouped

    def test_get_group_unknown(self) -> None:
        engine = RulesEngine()
        assert engine.get_group("nonexistent") is None

    def test_multiple_rules_different_severities(self) -> None:
        engine = RulesEngine()
        engine.add_rule(ThresholdRule("low", threshold=0.01, severity=Severity.INFO))
        engine.add_rule(ThresholdRule("high", threshold=0.10, severity=Severity.EMERGENCY))
        alerts = engine.evaluate(_ctx(fraud_rate=0.15))
        severities = {a.severity for a in alerts}
        assert severities == {Severity.INFO, Severity.EMERGENCY}
