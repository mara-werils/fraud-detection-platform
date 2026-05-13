"""YAML-based configurable rule engine for fraud detection.

Allows defining fraud detection rules in YAML files that can be
hot-reloaded without restarting the service.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml

from shared.schemas import FeatureVector, Transaction

logger = structlog.get_logger(__name__)

# Supported comparison operators
OPERATORS: dict[str, Any] = {
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "eq": operator.eq,
    "neq": operator.ne,
}


@dataclass(frozen=True)
class RuleCondition:
    """A single condition within a rule."""

    field: str
    source: str  # "transaction" or "features"
    operator: str
    value: Any


@dataclass(frozen=True)
class YAMLRule:
    """A fraud detection rule loaded from YAML."""

    name: str
    description: str
    score: float
    enabled: bool
    conditions: list[RuleCondition]
    match_mode: str  # "all" or "any"


@dataclass
class YAMLRuleResult:
    """Result from evaluating a YAML rule."""

    rule_name: str
    score: float
    triggered: bool
    detail: str


class YAMLRuleEngine:
    """Rule engine that loads rules from YAML configuration.

    Rules are defined as YAML files with conditions that reference
    transaction or feature fields. Supports hot-reloading via reload().

    Example YAML rule:
        rules:
          - name: high_amount_new_device
            description: "High amount from a new device"
            score: 0.4
            enabled: true
            match_mode: all
            conditions:
              - field: amount
                source: transaction
                operator: gt
                value: 50000
              - field: is_new_device
                source: features
                operator: eq
                value: true
    """

    def __init__(self, rules_path: str | Path | None = None) -> None:
        self._rules: list[YAMLRule] = []
        self._rules_path = Path(rules_path) if rules_path else None

        if self._rules_path and self._rules_path.exists():
            self.load(self._rules_path)

    def load(self, path: Path) -> None:
        """Load rules from a YAML file.

        Args:
            path: Path to the YAML rules file.
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            self._rules = []
            for rule_data in data.get("rules", []):
                conditions = [
                    RuleCondition(
                        field=c["field"],
                        source=c.get("source", "features"),
                        operator=c["operator"],
                        value=c["value"],
                    )
                    for c in rule_data.get("conditions", [])
                ]
                self._rules.append(
                    YAMLRule(
                        name=rule_data["name"],
                        description=rule_data.get("description", ""),
                        score=float(rule_data.get("score", 0.1)),
                        enabled=rule_data.get("enabled", True),
                        conditions=conditions,
                        match_mode=rule_data.get("match_mode", "all"),
                    )
                )

            logger.info("yaml_rules_loaded", count=len(self._rules), path=str(path))

        except Exception:
            logger.exception("yaml_rules_load_error", path=str(path))

    def reload(self) -> None:
        """Reload rules from the configured path."""
        if self._rules_path:
            self.load(self._rules_path)

    @property
    def rules(self) -> list[YAMLRule]:
        """Return the currently loaded rules."""
        return list(self._rules)

    def evaluate(
        self, transaction: Transaction, features: FeatureVector
    ) -> list[YAMLRuleResult]:
        """Evaluate all enabled rules against a transaction.

        Args:
            transaction: The transaction to evaluate.
            features: Computed feature vector.

        Returns:
            List of YAMLRuleResult for each enabled rule.
        """
        results: list[YAMLRuleResult] = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            triggered, detail = self._evaluate_rule(rule, transaction, features)
            results.append(
                YAMLRuleResult(
                    rule_name=rule.name,
                    score=rule.score if triggered else 0.0,
                    triggered=triggered,
                    detail=detail,
                )
            )

        return results

    def total_score(self, transaction: Transaction, features: FeatureVector) -> float:
        """Evaluate all rules and return the total clamped score.

        Args:
            transaction: The transaction to evaluate.
            features: Computed feature vector.

        Returns:
            Total fraud score from YAML rules, clamped to [0, 1].
        """
        results = self.evaluate(transaction, features)
        total = sum(r.score for r in results)
        return min(max(total, 0.0), 1.0)

    def _evaluate_rule(
        self,
        rule: YAMLRule,
        transaction: Transaction,
        features: FeatureVector,
    ) -> tuple[bool, str]:
        """Evaluate a single rule.

        Returns:
            Tuple of (triggered, detail_string).
        """
        condition_results: list[bool] = []
        details: list[str] = []

        for cond in rule.conditions:
            # Get the field value from the appropriate source
            if cond.source == "transaction":
                field_value = self._get_field(transaction, cond.field)
            else:
                field_value = self._get_field(features, cond.field)

            if field_value is None:
                condition_results.append(False)
                details.append(f"{cond.field}: field not found")
                continue

            # Apply the operator
            op_func = OPERATORS.get(cond.operator)
            if op_func is None:
                condition_results.append(False)
                details.append(f"{cond.field}: unknown operator '{cond.operator}'")
                continue

            try:
                # Convert to comparable types
                field_val = float(field_value) if isinstance(cond.value, (int, float)) else field_value
                result = op_func(field_val, cond.value)
                condition_results.append(result)
                details.append(f"{cond.field}={field_value} {cond.operator} {cond.value}: {result}")
            except (TypeError, ValueError):
                condition_results.append(False)
                details.append(f"{cond.field}: comparison error")

        # Determine if the rule triggered based on match_mode
        if rule.match_mode == "any":
            triggered = any(condition_results)
        else:
            triggered = all(condition_results) if condition_results else False

        detail = f"{rule.name}: {'; '.join(details)}"
        return triggered, detail

    @staticmethod
    def _get_field(obj: Any, field_name: str) -> Any:
        """Get a field value from a Pydantic model or dict."""
        if hasattr(obj, field_name):
            return getattr(obj, field_name)
        if isinstance(obj, dict):
            return obj.get(field_name)
        return None
