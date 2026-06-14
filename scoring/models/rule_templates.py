"""Pre-built fraud detection rule template library.

Provides a catalog of reusable rule templates that can be instantiated with
custom thresholds and composed into scoring pipelines. Each template is a
frozen dataclass describing the rule's metadata, default thresholds, and an
``evaluate`` callable that returns whether the rule triggered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Callable


class RuleCategory(Enum):
    """Broad categories for organising rule templates."""

    AMOUNT = "amount"
    VELOCITY = "velocity"
    GEOGRAPHIC = "geographic"
    TEMPORAL = "temporal"
    BEHAVIOURAL = "behavioural"
    PATTERN = "pattern"


@dataclass(frozen=True)
class Condition:
    """A single boolean condition within a rule.

    Attributes:
        field: The transaction / feature field to inspect.
        operator: Comparison operator (``gt``, ``lt``, ``gte``, ``lte``, ``eq``, ``ne``).
        description: Human-readable explanation of this condition.
    """

    field: str
    operator: str
    description: str


@dataclass(frozen=True)
class RuleTemplate:
    """Immutable description of a fraud-detection rule.

    Attributes:
        name: Machine-readable unique identifier for the template.
        description: Human-readable summary of what the rule detects.
        category: High-level category (amount, velocity, etc.).
        conditions: List of conditions that must all be satisfied.
        default_thresholds: Mapping of threshold name to its default value.
        default_score: Score contribution when the rule triggers.
        evaluate: Callable ``(transaction_data, thresholds) -> bool`` that
            returns ``True`` when the rule fires.
    """

    name: str
    description: str
    category: RuleCategory
    conditions: list[Condition]
    default_thresholds: dict[str, Any]
    default_score: float
    evaluate: Callable[[dict[str, Any], dict[str, Any]], bool]


@dataclass
class RuleInstance:
    """A concrete rule created from a template with (possibly overridden) thresholds.

    Attributes:
        template: The originating template.
        thresholds: Active thresholds (defaults merged with overrides).
        score: Score contribution when triggered.
        enabled: Whether this instance is active.
    """

    template: RuleTemplate
    thresholds: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.thresholds:
            self.thresholds = dict(self.template.default_thresholds)
        if self.score == 0.0:
            self.score = self.template.default_score

    @property
    def name(self) -> str:
        return self.template.name

    def evaluate(self, transaction_data: dict[str, Any]) -> bool:
        """Evaluate the rule against *transaction_data* using active thresholds."""
        if not self.enabled:
            return False
        return self.template.evaluate(transaction_data, self.thresholds)


# ---------------------------------------------------------------------------
# Evaluate helpers (pure functions used by the templates)
# ---------------------------------------------------------------------------

def _eval_high_value(txn: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Trigger when the transaction amount exceeds an absolute or relative ceiling."""
    amount = float(txn.get("amount", 0))
    abs_threshold = float(thresholds.get("absolute_amount", 10000))
    avg = float(txn.get("avg_transaction_amount", 0))
    multiplier = float(thresholds.get("multiplier", 3.0))
    if amount >= abs_threshold:
        return True
    if avg > 0 and amount >= avg * multiplier:
        return True
    return False


def _eval_rapid_succession(txn: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Trigger when too many transactions occur in a short window."""
    count = int(txn.get("transaction_count_window", 0))
    max_count = int(thresholds.get("max_transactions", 5))
    window_minutes = int(thresholds.get("window_minutes", 10))
    # The caller is expected to supply the count for the given window.
    _ = window_minutes  # stored for documentation / config purposes
    return count > max_count


def _eval_geographic_anomaly(txn: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Trigger when the distance from the previous transaction is implausible."""
    distance_km = float(txn.get("distance_from_last_txn_km", 0))
    max_distance = float(thresholds.get("max_distance_km", 500))
    time_gap_hours = float(txn.get("hours_since_last_txn", 24))
    min_speed_kmh = float(thresholds.get("min_speed_kmh", 900))
    if distance_km > max_distance:
        if time_gap_hours > 0 and (distance_km / time_gap_hours) > min_speed_kmh:
            return True
        if distance_km > max_distance * 2:
            return True
    return False


def _eval_round_amount(txn: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Trigger when the amount is suspiciously round."""
    amount = float(txn.get("amount", 0))
    min_amount = float(thresholds.get("min_amount", 100))
    round_multiple = float(thresholds.get("round_multiple", 100))
    if amount < min_amount:
        return False
    return amount % round_multiple == 0


def _eval_first_time_merchant(txn: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Trigger when the user has never transacted with this merchant before."""
    is_first = bool(txn.get("is_first_merchant_transaction", False))
    amount = float(txn.get("amount", 0))
    amount_threshold = float(thresholds.get("amount_threshold", 500))
    return is_first and amount >= amount_threshold


def _eval_dormant_account(txn: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Trigger when a previously dormant account suddenly becomes active."""
    days_inactive = int(txn.get("days_since_last_transaction", 0))
    dormant_days = int(thresholds.get("dormant_days", 90))
    amount = float(txn.get("amount", 0))
    amount_threshold = float(thresholds.get("amount_threshold", 200))
    return days_inactive >= dormant_days and amount >= amount_threshold


def _eval_card_testing(txn: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Trigger on card-testing patterns: small probes followed by a large charge."""
    small_txn_count = int(txn.get("small_transaction_count_24h", 0))
    min_small_count = int(thresholds.get("min_small_transactions", 3))
    small_amount_ceiling = float(thresholds.get("small_amount_ceiling", 5.0))
    amount = float(txn.get("amount", 0))
    large_amount_floor = float(thresholds.get("large_amount_floor", 500))
    latest_small_amount = float(txn.get("latest_small_amount", 0))
    has_small_probes = (
        small_txn_count >= min_small_count
        and latest_small_amount <= small_amount_ceiling
    )
    return has_small_probes and amount >= large_amount_floor


def _eval_unusual_hours(txn: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Trigger when the transaction occurs outside normal hours."""
    hour = int(txn.get("transaction_hour", 12))
    start_hour = int(thresholds.get("start_hour", 0))
    end_hour = int(thresholds.get("end_hour", 5))
    if start_hour <= end_hour:
        return start_hour <= hour < end_hour
    # Wraps around midnight, e.g. 22-05
    return hour >= start_hour or hour < end_hour


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

HIGH_VALUE_TRANSACTION = RuleTemplate(
    name="high_value_transaction",
    description=(
        "Flags transactions that exceed an absolute dollar ceiling or are "
        "significantly higher than the user's historical average."
    ),
    category=RuleCategory.AMOUNT,
    conditions=[
        Condition("amount", "gte", "Amount >= absolute threshold"),
        Condition("amount", "gte", "Amount >= multiplier * user average"),
    ],
    default_thresholds={"absolute_amount": 10000, "multiplier": 3.0},
    default_score=0.30,
    evaluate=_eval_high_value,
)

RAPID_SUCCESSION = RuleTemplate(
    name="rapid_succession",
    description=(
        "Detects bursts of transactions within a short time window, a common "
        "indicator of automated fraud."
    ),
    category=RuleCategory.VELOCITY,
    conditions=[
        Condition("transaction_count_window", "gt", "Count in window > max"),
    ],
    default_thresholds={"max_transactions": 5, "window_minutes": 10},
    default_score=0.25,
    evaluate=_eval_rapid_succession,
)

GEOGRAPHIC_ANOMALY = RuleTemplate(
    name="geographic_anomaly",
    description=(
        "Identifies geographically impossible travel between consecutive "
        "transactions based on distance and elapsed time."
    ),
    category=RuleCategory.GEOGRAPHIC,
    conditions=[
        Condition("distance_from_last_txn_km", "gt", "Distance exceeds max"),
        Condition("speed", "gt", "Implied speed exceeds plausible travel"),
    ],
    default_thresholds={"max_distance_km": 500, "min_speed_kmh": 900},
    default_score=0.35,
    evaluate=_eval_geographic_anomaly,
)

ROUND_AMOUNT_DETECTION = RuleTemplate(
    name="round_amount_detection",
    description=(
        "Flags transactions with suspiciously round amounts, which are "
        "statistically uncommon in legitimate purchases."
    ),
    category=RuleCategory.AMOUNT,
    conditions=[
        Condition("amount", "gte", "Amount at or above minimum"),
        Condition("amount", "mod", "Amount is exact multiple of round_multiple"),
    ],
    default_thresholds={"min_amount": 100, "round_multiple": 100},
    default_score=0.10,
    evaluate=_eval_round_amount,
)

FIRST_TIME_MERCHANT = RuleTemplate(
    name="first_time_merchant",
    description=(
        "Flags high-value transactions at a merchant the user has never "
        "transacted with before."
    ),
    category=RuleCategory.BEHAVIOURAL,
    conditions=[
        Condition("is_first_merchant_transaction", "eq", "First time at merchant"),
        Condition("amount", "gte", "Amount >= threshold"),
    ],
    default_thresholds={"amount_threshold": 500},
    default_score=0.20,
    evaluate=_eval_first_time_merchant,
)

DORMANT_ACCOUNT_ACTIVATION = RuleTemplate(
    name="dormant_account_activation",
    description=(
        "Detects sudden activity on accounts that have been inactive for an "
        "extended period, a hallmark of account takeover."
    ),
    category=RuleCategory.BEHAVIOURAL,
    conditions=[
        Condition("days_since_last_transaction", "gte", "Exceeds dormant threshold"),
        Condition("amount", "gte", "Amount >= threshold"),
    ],
    default_thresholds={"dormant_days": 90, "amount_threshold": 200},
    default_score=0.30,
    evaluate=_eval_dormant_account,
)

CARD_TESTING_PATTERN = RuleTemplate(
    name="card_testing_pattern",
    description=(
        "Identifies card-testing behaviour where multiple small transactions "
        "are followed by a single large purchase."
    ),
    category=RuleCategory.PATTERN,
    conditions=[
        Condition("small_transaction_count_24h", "gte", "Enough small probes"),
        Condition("amount", "gte", "Current amount is large"),
    ],
    default_thresholds={
        "min_small_transactions": 3,
        "small_amount_ceiling": 5.0,
        "large_amount_floor": 500,
    },
    default_score=0.35,
    evaluate=_eval_card_testing,
)

UNUSUAL_HOURS = RuleTemplate(
    name="unusual_hours",
    description=(
        "Flags transactions that occur during unusual hours (e.g. between "
        "midnight and 5 AM) when legitimate activity is rare."
    ),
    category=RuleCategory.TEMPORAL,
    conditions=[
        Condition("transaction_hour", "between", "Hour falls within unusual window"),
    ],
    default_thresholds={"start_hour": 0, "end_hour": 5},
    default_score=0.10,
    evaluate=_eval_unusual_hours,
)

# Collect all built-in templates for the registry default set.
_BUILTIN_TEMPLATES: list[RuleTemplate] = [
    HIGH_VALUE_TRANSACTION,
    RAPID_SUCCESSION,
    GEOGRAPHIC_ANOMALY,
    ROUND_AMOUNT_DETECTION,
    FIRST_TIME_MERCHANT,
    DORMANT_ACCOUNT_ACTIVATION,
    CARD_TESTING_PATTERN,
    UNUSUAL_HOURS,
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TemplateRegistry:
    """Central catalog of rule templates.

    The registry ships with all built-in templates and supports adding custom
    ones at runtime.
    """

    def __init__(self, *, include_builtins: bool = True) -> None:
        self._templates: dict[str, RuleTemplate] = {}
        if include_builtins:
            for tpl in _BUILTIN_TEMPLATES:
                self.register(tpl)

    # -- Mutation -----------------------------------------------------------

    def register(self, template: RuleTemplate) -> None:
        """Add or replace a template in the registry.

        Args:
            template: The rule template to register.

        Raises:
            TypeError: If *template* is not a ``RuleTemplate``.
        """
        if not isinstance(template, RuleTemplate):
            raise TypeError(f"Expected RuleTemplate, got {type(template).__name__}")
        self._templates[template.name] = template

    def unregister(self, name: str) -> None:
        """Remove a template by name.

        Args:
            name: Template identifier.

        Raises:
            KeyError: If no template with *name* exists.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found in registry")
        del self._templates[name]

    # -- Query --------------------------------------------------------------

    def list_templates(self) -> list[RuleTemplate]:
        """Return all registered templates sorted by name."""
        return sorted(self._templates.values(), key=lambda t: t.name)

    def get(self, name: str) -> RuleTemplate:
        """Retrieve a template by name.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found in registry")
        return self._templates[name]

    def filter_by_category(self, category: RuleCategory) -> list[RuleTemplate]:
        """Return templates belonging to *category*, sorted by name."""
        return sorted(
            [t for t in self._templates.values() if t.category == category],
            key=lambda t: t.name,
        )

    def instantiate(
        self,
        name: str,
        *,
        threshold_overrides: dict[str, Any] | None = None,
        score: float | None = None,
        enabled: bool = True,
    ) -> RuleInstance:
        """Create a ``RuleInstance`` from the named template.

        Args:
            name: Registered template name.
            threshold_overrides: Values to override on top of the template defaults.
            score: Custom score; uses the template default when ``None``.
            enabled: Whether the instance starts enabled.

        Returns:
            A ready-to-use ``RuleInstance``.
        """
        template = self.get(name)
        thresholds = dict(template.default_thresholds)
        if threshold_overrides:
            thresholds.update(threshold_overrides)
        return RuleInstance(
            template=template,
            thresholds=thresholds,
            score=score if score is not None else template.default_score,
            enabled=enabled,
        )

    def __len__(self) -> int:
        return len(self._templates)

    def __contains__(self, name: str) -> bool:
        return name in self._templates
