"""Tests for the pre-built fraud detection rule template library.

Covers every built-in template's evaluation logic, the TemplateRegistry
operations (register, unregister, filter, instantiate), and edge cases.
"""

from __future__ import annotations

import pytest

from scoring.models.rule_templates import (
    CARD_TESTING_PATTERN,
    DORMANT_ACCOUNT_ACTIVATION,
    FIRST_TIME_MERCHANT,
    GEOGRAPHIC_ANOMALY,
    HIGH_VALUE_TRANSACTION,
    RAPID_SUCCESSION,
    ROUND_AMOUNT_DETECTION,
    UNUSUAL_HOURS,
    Condition,
    RuleCategory,
    RuleInstance,
    RuleTemplate,
    TemplateRegistry,
    _BUILTIN_TEMPLATES,
)


# ---------------------------------------------------------------------------
# High-value transaction rule
# ---------------------------------------------------------------------------


class TestHighValueTransaction:
    def test_triggers_on_absolute_amount(self) -> None:
        txn = {"amount": 15000, "avg_transaction_amount": 0}
        assert HIGH_VALUE_TRANSACTION.evaluate(txn, HIGH_VALUE_TRANSACTION.default_thresholds)

    def test_triggers_on_relative_amount(self) -> None:
        txn = {"amount": 400, "avg_transaction_amount": 100}
        assert HIGH_VALUE_TRANSACTION.evaluate(txn, {"absolute_amount": 10000, "multiplier": 3.0})

    def test_no_trigger_below_thresholds(self) -> None:
        txn = {"amount": 200, "avg_transaction_amount": 100}
        assert not HIGH_VALUE_TRANSACTION.evaluate(txn, HIGH_VALUE_TRANSACTION.default_thresholds)

    def test_no_trigger_when_avg_is_zero_and_below_absolute(self) -> None:
        txn = {"amount": 50, "avg_transaction_amount": 0}
        assert not HIGH_VALUE_TRANSACTION.evaluate(txn, HIGH_VALUE_TRANSACTION.default_thresholds)


# ---------------------------------------------------------------------------
# Rapid succession rule
# ---------------------------------------------------------------------------


class TestRapidSuccession:
    def test_triggers_above_max(self) -> None:
        txn = {"transaction_count_window": 8}
        assert RAPID_SUCCESSION.evaluate(txn, RAPID_SUCCESSION.default_thresholds)

    def test_no_trigger_at_max(self) -> None:
        txn = {"transaction_count_window": 5}
        assert not RAPID_SUCCESSION.evaluate(txn, RAPID_SUCCESSION.default_thresholds)

    def test_no_trigger_below_max(self) -> None:
        txn = {"transaction_count_window": 2}
        assert not RAPID_SUCCESSION.evaluate(txn, RAPID_SUCCESSION.default_thresholds)

    def test_custom_threshold(self) -> None:
        txn = {"transaction_count_window": 3}
        assert RAPID_SUCCESSION.evaluate(txn, {"max_transactions": 2, "window_minutes": 5})


# ---------------------------------------------------------------------------
# Geographic anomaly rule
# ---------------------------------------------------------------------------


class TestGeographicAnomaly:
    def test_triggers_on_impossible_speed(self) -> None:
        txn = {"distance_from_last_txn_km": 800, "hours_since_last_txn": 0.5}
        assert GEOGRAPHIC_ANOMALY.evaluate(txn, GEOGRAPHIC_ANOMALY.default_thresholds)

    def test_triggers_on_extreme_distance(self) -> None:
        # > 2x max distance always triggers regardless of speed
        txn = {"distance_from_last_txn_km": 1200, "hours_since_last_txn": 10}
        assert GEOGRAPHIC_ANOMALY.evaluate(txn, GEOGRAPHIC_ANOMALY.default_thresholds)

    def test_no_trigger_within_range(self) -> None:
        txn = {"distance_from_last_txn_km": 200, "hours_since_last_txn": 2}
        assert not GEOGRAPHIC_ANOMALY.evaluate(txn, GEOGRAPHIC_ANOMALY.default_thresholds)

    def test_no_trigger_moderate_distance_reasonable_time(self) -> None:
        txn = {"distance_from_last_txn_km": 600, "hours_since_last_txn": 5}
        # speed = 120 km/h, under 900 threshold; distance under 1000 (2x500)
        assert not GEOGRAPHIC_ANOMALY.evaluate(txn, GEOGRAPHIC_ANOMALY.default_thresholds)


# ---------------------------------------------------------------------------
# Round amount detection rule
# ---------------------------------------------------------------------------


class TestRoundAmountDetection:
    def test_triggers_on_round_amount(self) -> None:
        txn = {"amount": 500}
        assert ROUND_AMOUNT_DETECTION.evaluate(txn, ROUND_AMOUNT_DETECTION.default_thresholds)

    def test_no_trigger_below_min(self) -> None:
        txn = {"amount": 50}
        assert not ROUND_AMOUNT_DETECTION.evaluate(txn, ROUND_AMOUNT_DETECTION.default_thresholds)

    def test_no_trigger_non_round(self) -> None:
        txn = {"amount": 347.52}
        assert not ROUND_AMOUNT_DETECTION.evaluate(txn, ROUND_AMOUNT_DETECTION.default_thresholds)

    def test_custom_multiple(self) -> None:
        txn = {"amount": 250}
        assert ROUND_AMOUNT_DETECTION.evaluate(txn, {"min_amount": 100, "round_multiple": 50})


# ---------------------------------------------------------------------------
# First-time merchant rule
# ---------------------------------------------------------------------------


class TestFirstTimeMerchant:
    def test_triggers_first_merchant_high_amount(self) -> None:
        txn = {"is_first_merchant_transaction": True, "amount": 600}
        assert FIRST_TIME_MERCHANT.evaluate(txn, FIRST_TIME_MERCHANT.default_thresholds)

    def test_no_trigger_known_merchant(self) -> None:
        txn = {"is_first_merchant_transaction": False, "amount": 600}
        assert not FIRST_TIME_MERCHANT.evaluate(txn, FIRST_TIME_MERCHANT.default_thresholds)

    def test_no_trigger_first_merchant_low_amount(self) -> None:
        txn = {"is_first_merchant_transaction": True, "amount": 50}
        assert not FIRST_TIME_MERCHANT.evaluate(txn, FIRST_TIME_MERCHANT.default_thresholds)


# ---------------------------------------------------------------------------
# Dormant account activation rule
# ---------------------------------------------------------------------------


class TestDormantAccountActivation:
    def test_triggers_dormant_high_amount(self) -> None:
        txn = {"days_since_last_transaction": 120, "amount": 300}
        assert DORMANT_ACCOUNT_ACTIVATION.evaluate(
            txn, DORMANT_ACCOUNT_ACTIVATION.default_thresholds
        )

    def test_no_trigger_active_account(self) -> None:
        txn = {"days_since_last_transaction": 10, "amount": 300}
        assert not DORMANT_ACCOUNT_ACTIVATION.evaluate(
            txn, DORMANT_ACCOUNT_ACTIVATION.default_thresholds
        )

    def test_no_trigger_dormant_low_amount(self) -> None:
        txn = {"days_since_last_transaction": 120, "amount": 50}
        assert not DORMANT_ACCOUNT_ACTIVATION.evaluate(
            txn, DORMANT_ACCOUNT_ACTIVATION.default_thresholds
        )


# ---------------------------------------------------------------------------
# Card testing pattern rule
# ---------------------------------------------------------------------------


class TestCardTestingPattern:
    def test_triggers_classic_pattern(self) -> None:
        txn = {
            "small_transaction_count_24h": 5,
            "latest_small_amount": 1.00,
            "amount": 800,
        }
        assert CARD_TESTING_PATTERN.evaluate(txn, CARD_TESTING_PATTERN.default_thresholds)

    def test_no_trigger_without_small_probes(self) -> None:
        txn = {
            "small_transaction_count_24h": 1,
            "latest_small_amount": 1.00,
            "amount": 800,
        }
        assert not CARD_TESTING_PATTERN.evaluate(txn, CARD_TESTING_PATTERN.default_thresholds)

    def test_no_trigger_when_current_amount_small(self) -> None:
        txn = {
            "small_transaction_count_24h": 5,
            "latest_small_amount": 1.00,
            "amount": 20,
        }
        assert not CARD_TESTING_PATTERN.evaluate(txn, CARD_TESTING_PATTERN.default_thresholds)

    def test_no_trigger_when_small_amounts_too_large(self) -> None:
        txn = {
            "small_transaction_count_24h": 5,
            "latest_small_amount": 50.00,
            "amount": 800,
        }
        assert not CARD_TESTING_PATTERN.evaluate(txn, CARD_TESTING_PATTERN.default_thresholds)


# ---------------------------------------------------------------------------
# Unusual hours rule
# ---------------------------------------------------------------------------


class TestUnusualHours:
    def test_triggers_at_2am(self) -> None:
        txn = {"transaction_hour": 2}
        assert UNUSUAL_HOURS.evaluate(txn, UNUSUAL_HOURS.default_thresholds)

    def test_no_trigger_at_noon(self) -> None:
        txn = {"transaction_hour": 12}
        assert not UNUSUAL_HOURS.evaluate(txn, UNUSUAL_HOURS.default_thresholds)

    def test_boundary_start(self) -> None:
        txn = {"transaction_hour": 0}
        assert UNUSUAL_HOURS.evaluate(txn, UNUSUAL_HOURS.default_thresholds)

    def test_boundary_end_exclusive(self) -> None:
        txn = {"transaction_hour": 5}
        assert not UNUSUAL_HOURS.evaluate(txn, UNUSUAL_HOURS.default_thresholds)

    def test_wrap_around_midnight(self) -> None:
        thresholds = {"start_hour": 22, "end_hour": 5}
        assert UNUSUAL_HOURS.evaluate({"transaction_hour": 23}, thresholds)
        assert UNUSUAL_HOURS.evaluate({"transaction_hour": 3}, thresholds)
        assert not UNUSUAL_HOURS.evaluate({"transaction_hour": 10}, thresholds)


# ---------------------------------------------------------------------------
# TemplateRegistry
# ---------------------------------------------------------------------------


class TestTemplateRegistry:
    def test_default_registry_contains_all_builtins(self) -> None:
        reg = TemplateRegistry()
        assert len(reg) == len(_BUILTIN_TEMPLATES)
        for tpl in _BUILTIN_TEMPLATES:
            assert tpl.name in reg

    def test_empty_registry(self) -> None:
        reg = TemplateRegistry(include_builtins=False)
        assert len(reg) == 0

    def test_list_templates_sorted(self) -> None:
        reg = TemplateRegistry()
        names = [t.name for t in reg.list_templates()]
        assert names == sorted(names)

    def test_get_existing_template(self) -> None:
        reg = TemplateRegistry()
        tpl = reg.get("high_value_transaction")
        assert tpl is HIGH_VALUE_TRANSACTION

    def test_get_missing_raises(self) -> None:
        reg = TemplateRegistry()
        with pytest.raises(KeyError, match="not_here"):
            reg.get("not_here")

    def test_register_custom_template(self) -> None:
        reg = TemplateRegistry(include_builtins=False)
        custom = RuleTemplate(
            name="custom",
            description="A custom rule",
            category=RuleCategory.AMOUNT,
            conditions=[],
            default_thresholds={"x": 1},
            default_score=0.5,
            evaluate=lambda t, th: True,
        )
        reg.register(custom)
        assert "custom" in reg
        assert reg.get("custom") is custom

    def test_register_rejects_non_template(self) -> None:
        reg = TemplateRegistry(include_builtins=False)
        with pytest.raises(TypeError):
            reg.register("not a template")  # type: ignore[arg-type]

    def test_unregister(self) -> None:
        reg = TemplateRegistry()
        reg.unregister("unusual_hours")
        assert "unusual_hours" not in reg

    def test_unregister_missing_raises(self) -> None:
        reg = TemplateRegistry()
        with pytest.raises(KeyError):
            reg.unregister("does_not_exist")

    def test_filter_by_category(self) -> None:
        reg = TemplateRegistry()
        amount_rules = reg.filter_by_category(RuleCategory.AMOUNT)
        assert len(amount_rules) >= 2
        assert all(t.category == RuleCategory.AMOUNT for t in amount_rules)

    def test_filter_by_category_empty(self) -> None:
        reg = TemplateRegistry(include_builtins=False)
        assert reg.filter_by_category(RuleCategory.AMOUNT) == []


# ---------------------------------------------------------------------------
# RuleInstance (instantiation)
# ---------------------------------------------------------------------------


class TestRuleInstance:
    def test_instantiate_with_defaults(self) -> None:
        reg = TemplateRegistry()
        inst = reg.instantiate("high_value_transaction")
        assert inst.name == "high_value_transaction"
        assert inst.thresholds == HIGH_VALUE_TRANSACTION.default_thresholds
        assert inst.score == HIGH_VALUE_TRANSACTION.default_score
        assert inst.enabled is True

    def test_instantiate_with_overrides(self) -> None:
        reg = TemplateRegistry()
        inst = reg.instantiate(
            "rapid_succession",
            threshold_overrides={"max_transactions": 10},
            score=0.5,
        )
        assert inst.thresholds["max_transactions"] == 10
        # window_minutes should still have its default
        assert inst.thresholds["window_minutes"] == 10
        assert inst.score == 0.5

    def test_instantiate_disabled(self) -> None:
        reg = TemplateRegistry()
        inst = reg.instantiate("unusual_hours", enabled=False)
        assert inst.enabled is False
        # Disabled instance should never fire
        assert inst.evaluate({"transaction_hour": 2}) is False

    def test_instance_evaluate_delegates_to_template(self) -> None:
        reg = TemplateRegistry()
        inst = reg.instantiate("round_amount_detection")
        assert inst.evaluate({"amount": 500}) is True
        assert inst.evaluate({"amount": 123.45}) is False

    def test_instance_defaults_from_template_when_empty(self) -> None:
        inst = RuleInstance(template=HIGH_VALUE_TRANSACTION)
        assert inst.thresholds == HIGH_VALUE_TRANSACTION.default_thresholds
        assert inst.score == HIGH_VALUE_TRANSACTION.default_score


# ---------------------------------------------------------------------------
# Data model integrity
# ---------------------------------------------------------------------------


class TestDataModelIntegrity:
    def test_all_templates_have_unique_names(self) -> None:
        names = [t.name for t in _BUILTIN_TEMPLATES]
        assert len(names) == len(set(names))

    def test_all_templates_have_descriptions(self) -> None:
        for tpl in _BUILTIN_TEMPLATES:
            assert tpl.description, f"{tpl.name} is missing a description"

    def test_all_templates_have_at_least_one_condition(self) -> None:
        for tpl in _BUILTIN_TEMPLATES:
            assert len(tpl.conditions) >= 1, f"{tpl.name} has no conditions"

    def test_all_templates_have_positive_default_score(self) -> None:
        for tpl in _BUILTIN_TEMPLATES:
            assert tpl.default_score > 0, f"{tpl.name} has non-positive score"

    def test_builtin_count_at_least_eight(self) -> None:
        assert len(_BUILTIN_TEMPLATES) >= 8

    def test_condition_is_frozen(self) -> None:
        c = Condition(field="amount", operator="gt", description="test")
        with pytest.raises(AttributeError):
            c.field = "other"  # type: ignore[misc]

    def test_rule_template_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            HIGH_VALUE_TRANSACTION.name = "hacked"  # type: ignore[misc]

    def test_rule_category_values(self) -> None:
        expected = {"amount", "velocity", "geographic", "temporal", "behavioural", "pattern"}
        actual = {c.value for c in RuleCategory}
        assert actual == expected
