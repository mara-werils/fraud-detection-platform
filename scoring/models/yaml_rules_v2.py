"""Enhanced YAML-based rules engine v2 for fraud detection.

Supports hot-reload, nested logical conditions (AND/OR/NOT), additive and
multiplicative score contributions, per-rule statistics, file watching, and
full Pydantic v2 validation of the rule configuration.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog
import yaml
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field, field_validator, model_validator

from shared.schemas import FeatureVector, Transaction

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_RULE_EVALUATIONS = Counter(
    "fraud_yaml_rules_v2_evaluations_total",
    "Total rule evaluations",
    ["rule_id", "triggered"],
)
_RULE_EVAL_LATENCY = Histogram(
    "fraud_yaml_rules_v2_evaluation_seconds",
    "Rule evaluation latency",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
)
_RULESET_RELOAD_COUNTER = Counter(
    "fraud_yaml_rules_v2_reloads_total",
    "Total rule set reloads",
    ["status"],
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Operator(StrEnum):
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    REGEX = "regex"
    CONTAINS = "contains"


class LogicalCombinator(StrEnum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class ScoreMethod(StrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class CombinationMethod(StrEnum):
    SUM = "sum"
    MAX = "max"
    WEIGHTED_AVG = "weighted_avg"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RuleCondition(BaseModel):
    """A single leaf condition that tests one field against a value."""

    field: str = Field(..., description="Dot-separated path to the field, e.g. 'amount' or 'metadata.channel'")
    operator: Operator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    negate: bool = Field(default=False, description="If True, invert the result of this condition")

    @field_validator("value")
    @classmethod
    def validate_between_value(cls, v: Any) -> Any:
        return v


class ConditionGroup(BaseModel):
    """A logical group of conditions or nested groups."""

    combinator: LogicalCombinator = Field(default=LogicalCombinator.AND)
    conditions: list[RuleCondition | ConditionGroup] = Field(default_factory=list)
    negate: bool = Field(default=False, description="If True, negate the whole group result")

    model_config = {"arbitrary_types_allowed": True}


# Make ConditionGroup self-referential — Pydantic v2 requires this call after class definition.
ConditionGroup.model_rebuild()


class RuleDefinition(BaseModel):
    """A single fraud detection rule loaded from YAML."""

    id: str = Field(..., description="Unique rule identifier, e.g. 'HIGH_VALUE_001'")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(default="", description="Explanation of what this rule detects")
    condition_group: ConditionGroup = Field(..., description="Top-level condition group")
    score: float = Field(..., ge=0.0, le=1.0, description="Score contribution when this rule fires")
    score_method: ScoreMethod = Field(default=ScoreMethod.ADDITIVE, description="How the score is applied")
    priority: int = Field(default=50, ge=1, le=100, description="Evaluation priority (higher = earlier)")
    weight: float = Field(default=1.0, gt=0.0, description="Weight for weighted_avg combination")
    enabled: bool = Field(default=True, description="If False the rule is skipped during evaluation")
    tags: list[str] = Field(default_factory=list, description="Grouping tags, e.g. ['velocity', 'geo']")


class RuleSet(BaseModel):
    """The complete collection of rules loaded from one YAML file."""

    name: str = Field(..., description="Name of this rule set")
    version: str = Field(default="1.0.0", description="Rule set version string")
    rules: list[RuleDefinition] = Field(default_factory=list)
    default_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Baseline score before rules are applied")
    combination_method: CombinationMethod = Field(default=CombinationMethod.SUM)

    @model_validator(mode="after")
    def unique_rule_ids(self) -> RuleSet:
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            duplicates = [i for i in ids if ids.count(i) > 1]
            raise ValueError(f"Duplicate rule IDs found: {list(set(duplicates))}")
        return self


class ValidationError(BaseModel):
    """A single validation problem found in a rule set."""

    rule_id: str
    field: str
    message: str


class RuleEvaluationResult(BaseModel):
    """Result of evaluating an entire rule set against one transaction."""

    transaction_id: str
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    triggered_rules: list[str] = Field(default_factory=list)
    rule_scores: dict[str, float] = Field(default_factory=dict)
    rule_details: dict[str, str] = Field(default_factory=dict)
    evaluation_time_ms: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Operator evaluation helpers
# ---------------------------------------------------------------------------


def _apply_operator(op: Operator, field_val: Any, rule_val: Any) -> bool:
    """Apply a single operator and return a boolean result."""

    if op == Operator.GT:
        return float(field_val) > float(rule_val)
    if op == Operator.LT:
        return float(field_val) < float(rule_val)
    if op == Operator.GTE:
        return float(field_val) >= float(rule_val)
    if op == Operator.LTE:
        return float(field_val) <= float(rule_val)
    if op == Operator.EQ:
        # Allow numeric comparison across int/float/Decimal
        try:
            return float(field_val) == float(rule_val)
        except (TypeError, ValueError):
            return str(field_val) == str(rule_val)
    if op == Operator.NE:
        try:
            return float(field_val) != float(rule_val)
        except (TypeError, ValueError):
            return str(field_val) != str(rule_val)
    if op == Operator.IN:
        if not isinstance(rule_val, list):
            raise ValueError(f"'in' operator requires a list value, got {type(rule_val)}")
        return field_val in rule_val
    if op == Operator.NOT_IN:
        if not isinstance(rule_val, list):
            raise ValueError(f"'not_in' operator requires a list value, got {type(rule_val)}")
        return field_val not in rule_val
    if op == Operator.BETWEEN:
        if not (isinstance(rule_val, (list, tuple)) and len(rule_val) == 2):
            raise ValueError(f"'between' operator requires a two-element list, got {rule_val!r}")
        lo, hi = rule_val
        return float(lo) <= float(field_val) <= float(hi)
    if op == Operator.REGEX:
        return bool(re.search(str(rule_val), str(field_val)))
    if op == Operator.CONTAINS:
        return str(rule_val) in str(field_val)

    raise ValueError(f"Unsupported operator: {op}")


def _resolve_field(obj: Any, dotted_path: str) -> Any:
    """Resolve a dot-separated field path from a Pydantic model, dict, or plain object."""
    parts = dotted_path.split(".")
    current: Any = obj
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


# ---------------------------------------------------------------------------
# Per-rule statistics tracker
# ---------------------------------------------------------------------------


class _RuleStats:
    """Thread-safe hit-rate tracker for a single rule."""

    __slots__ = ("_lock", "evaluations", "hits")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.evaluations: int = 0
        self.hits: int = 0

    def record(self, triggered: bool) -> None:
        with self._lock:
            self.evaluations += 1
            if triggered:
                self.hits += 1

    def hit_rate(self) -> float:
        with self._lock:
            if self.evaluations == 0:
                return 0.0
            return self.hits / self.evaluations

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "evaluations": self.evaluations,
                "hits": self.hits,
                "hit_rate": round(self.hits / self.evaluations, 4) if self.evaluations else 0.0,
            }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class YAMLRulesEngine:
    """Hot-reloadable YAML-based fraud detection rule engine (v2).

    Features over v1:
    - Nested AND / OR / NOT condition groups
    - Additive *and* multiplicative score contributions
    - Per-rule hit-rate statistics
    - Background file watcher for automatic hot-reload
    - Full Pydantic v2 validation on load
    - Export back to YAML string

    Usage::

        engine = YAMLRulesEngine()
        rule_set = engine.load_rules("scoring/rules_v2.yaml")
        engine.watch_file("scoring/rules_v2.yaml", callback=lambda: print("reloaded"))

        result = engine.evaluate(transaction, features)
    """

    def __init__(self) -> None:
        self._rule_set: RuleSet | None = None
        self._rules_path: Path | None = None
        self._stats: dict[str, _RuleStats] = {}
        self._lock = threading.RLock()
        self._watcher_thread: threading.Thread | None = None
        self._watcher_stop = threading.Event()
        self._last_file_hash: str | None = None

    # ------------------------------------------------------------------
    # Loading & hot-reload
    # ------------------------------------------------------------------

    def load_rules(self, path: str) -> RuleSet:
        """Load and validate a rule set from a YAML file.

        Args:
            path: Filesystem path to the rules YAML file.

        Returns:
            The validated :class:`RuleSet`.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the YAML fails Pydantic validation.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Rules file not found: {file_path}")

        raw = file_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        rule_set = self._parse_rule_set(data)

        with self._lock:
            self._rule_set = rule_set
            self._rules_path = file_path
            self._last_file_hash = self._hash_content(raw)
            self._stats = {r.id: _RuleStats() for r in rule_set.rules}

        _RULESET_RELOAD_COUNTER.labels(status="success").inc()
        logger.info(
            "yaml_rules_v2_loaded",
            path=str(file_path),
            rule_set=rule_set.name,
            version=rule_set.version,
            rule_count=len(rule_set.rules),
        )
        return rule_set

    def reload_rules(self) -> None:
        """Hot-reload rules from the previously configured path.

        Preserves accumulated per-rule statistics across reloads when a rule
        ID is retained.  New rules start with fresh statistics.
        """
        with self._lock:
            if self._rules_path is None:
                logger.warning("yaml_rules_v2_reload_skipped", reason="no_path_configured")
                return
            path = self._rules_path

        try:
            raw = path.read_text(encoding="utf-8")
            new_hash = self._hash_content(raw)

            with self._lock:
                if new_hash == self._last_file_hash:
                    logger.debug("yaml_rules_v2_reload_skipped", reason="content_unchanged")
                    return

            data = yaml.safe_load(raw)
            rule_set = self._parse_rule_set(data)

            with self._lock:
                old_stats = self._stats
                self._rule_set = rule_set
                self._last_file_hash = new_hash
                # Carry over stats for rules that still exist
                self._stats = {
                    r.id: old_stats.get(r.id, _RuleStats()) for r in rule_set.rules
                }

            _RULESET_RELOAD_COUNTER.labels(status="success").inc()
            logger.info(
                "yaml_rules_v2_reloaded",
                path=str(path),
                rule_count=len(rule_set.rules),
            )

        except Exception:
            _RULESET_RELOAD_COUNTER.labels(status="error").inc()
            logger.exception("yaml_rules_v2_reload_error", path=str(path))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        transaction: Transaction,
        features: FeatureVector,
    ) -> RuleEvaluationResult:
        """Evaluate all enabled rules against a transaction and its feature vector.

        Args:
            transaction: The incoming transaction.
            features: The pre-computed feature vector.

        Returns:
            :class:`RuleEvaluationResult` with the combined fraud score.
        """
        start = time.perf_counter()

        with self._lock:
            rule_set = self._rule_set

        if rule_set is None:
            logger.warning("yaml_rules_v2_evaluate_called_before_load")
            return RuleEvaluationResult(
                transaction_id=str(transaction.transaction_id),
                fraud_score=0.0,
            )

        context: dict[str, Any] = {"transaction": transaction, "features": features}
        triggered_rules: list[str] = []
        rule_scores: dict[str, float] = {}
        rule_details: dict[str, str] = {}

        # Evaluate rules in priority order (descending)
        sorted_rules = sorted(
            [r for r in rule_set.rules if r.enabled],
            key=lambda r: r.priority,
            reverse=True,
        )

        for rule in sorted_rules:
            triggered, detail = self._eval_rule_with_metrics(rule, context)
            rule_details[rule.id] = detail
            if triggered:
                triggered_rules.append(rule.id)
                rule_scores[rule.id] = rule.score

        fraud_score = self._combine_scores(rule_set, rule_scores, sorted_rules)

        elapsed_ms = (time.perf_counter() - start) * 1000
        _RULE_EVAL_LATENCY.observe(elapsed_ms / 1000)

        return RuleEvaluationResult(
            transaction_id=str(transaction.transaction_id),
            fraud_score=fraud_score,
            triggered_rules=triggered_rules,
            rule_scores=rule_scores,
            rule_details=rule_details,
            evaluation_time_ms=round(elapsed_ms, 3),
        )

    def evaluate_rule(
        self,
        rule: RuleDefinition,
        transaction: Transaction,
        features: FeatureVector,
    ) -> bool:
        """Evaluate a single rule definition.

        Args:
            rule: The rule to test.
            transaction: The transaction to test against.
            features: The pre-computed feature vector.

        Returns:
            True if the rule fires.
        """
        context: dict[str, Any] = {"transaction": transaction, "features": features}
        triggered, _ = self._evaluate_condition_group(rule.condition_group, context)
        return triggered

    def evaluate_condition(
        self,
        condition: RuleCondition,
        context: dict[str, Any],
    ) -> bool:
        """Evaluate a single leaf condition.

        Args:
            condition: The leaf condition to evaluate.
            context: A dict containing ``transaction`` and ``features`` objects.

        Returns:
            Boolean result, potentially negated by ``condition.negate``.
        """
        # Search transaction then features for the field
        result = False
        for source_key in ("transaction", "features"):
            source = context.get(source_key)
            if source is None:
                continue
            val = _resolve_field(source, condition.field)
            if val is not None:
                try:
                    result = _apply_operator(condition.operator, val, condition.value)
                except Exception as exc:
                    logger.warning(
                        "yaml_rules_v2_condition_error",
                        field=condition.field,
                        operator=condition.operator,
                        error=str(exc),
                    )
                    result = False
                break

        return (not result) if condition.negate else result

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: RuleDefinition) -> None:
        """Add or replace a rule in the active rule set.

        Args:
            rule: The rule to insert.  If a rule with the same ID already
                  exists it will be replaced.
        """
        with self._lock:
            if self._rule_set is None:
                raise RuntimeError("No rule set loaded; call load_rules() first.")
            rules = [r for r in self._rule_set.rules if r.id != rule.id]
            rules.append(rule)
            self._rule_set = self._rule_set.model_copy(update={"rules": rules})
            if rule.id not in self._stats:
                self._stats[rule.id] = _RuleStats()
        logger.info("yaml_rules_v2_rule_added", rule_id=rule.id)

    def remove_rule(self, rule_id: str) -> None:
        """Remove a rule from the active rule set by ID.

        Args:
            rule_id: The rule identifier to remove.
        """
        with self._lock:
            if self._rule_set is None:
                raise RuntimeError("No rule set loaded; call load_rules() first.")
            rules = [r for r in self._rule_set.rules if r.id != rule_id]
            self._rule_set = self._rule_set.model_copy(update={"rules": rules})
            self._stats.pop(rule_id, None)
        logger.info("yaml_rules_v2_rule_removed", rule_id=rule_id)

    def enable_rule(self, rule_id: str) -> None:
        """Enable a rule that was previously disabled.

        Args:
            rule_id: The rule identifier to enable.
        """
        self._set_rule_enabled(rule_id, enabled=True)

    def disable_rule(self, rule_id: str) -> None:
        """Disable a rule without removing it.

        Args:
            rule_id: The rule identifier to disable.
        """
        self._set_rule_enabled(rule_id, enabled=False)

    # ------------------------------------------------------------------
    # Validation & export
    # ------------------------------------------------------------------

    def validate_rules(self, rule_set: RuleSet) -> list[ValidationError]:
        """Validate a rule set and return a list of problems found.

        Args:
            rule_set: The :class:`RuleSet` to validate.

        Returns:
            A (possibly empty) list of :class:`ValidationError` objects.
        """
        errors: list[ValidationError] = []

        seen_ids: set[str] = set()
        for rule in rule_set.rules:
            if rule.id in seen_ids:
                errors.append(
                    ValidationError(rule_id=rule.id, field="id", message="Duplicate rule ID")
                )
            seen_ids.add(rule.id)

            if rule.score_method == ScoreMethod.MULTIPLICATIVE and not (0.0 < rule.score <= 1.0):
                errors.append(
                    ValidationError(
                        rule_id=rule.id,
                        field="score",
                        message="Multiplicative score must be in (0, 1]",
                    )
                )

            errors.extend(self._validate_condition_group(rule.id, rule.condition_group))

        return errors

    def _validate_condition_group(
        self, rule_id: str, group: ConditionGroup
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []
        if group.combinator == LogicalCombinator.NOT and len(group.conditions) != 1:
            errors.append(
                ValidationError(
                    rule_id=rule_id,
                    field="condition_group.combinator",
                    message="NOT combinator must have exactly one child condition",
                )
            )
        for cond in group.conditions:
            if isinstance(cond, ConditionGroup):
                errors.extend(self._validate_condition_group(rule_id, cond))
            else:
                if cond.operator == Operator.BETWEEN:
                    if not (isinstance(cond.value, list) and len(cond.value) == 2):
                        errors.append(
                            ValidationError(
                                rule_id=rule_id,
                                field=f"condition[{cond.field}].value",
                                message="'between' operator requires a two-element list [lo, hi]",
                            )
                        )
                if cond.operator in (Operator.IN, Operator.NOT_IN):
                    if not isinstance(cond.value, list):
                        errors.append(
                            ValidationError(
                                rule_id=rule_id,
                                field=f"condition[{cond.field}].value",
                                message=f"'{cond.operator}' operator requires a list value",
                            )
                        )
        return errors

    def get_rule_stats(self) -> dict[str, Any]:
        """Return per-rule evaluation statistics.

        Returns:
            A dict mapping rule ID to a stats sub-dict with keys
            ``evaluations``, ``hits``, and ``hit_rate``.
        """
        with self._lock:
            return {rule_id: stats.to_dict() for rule_id, stats in self._stats.items()}

    def export_rules(self) -> str:
        """Serialize the current rule set back to a YAML string.

        Returns:
            The YAML representation of the active :class:`RuleSet`.
        """
        with self._lock:
            if self._rule_set is None:
                return ""
            data = self._rule_set.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # File watching
    # ------------------------------------------------------------------

    def watch_file(self, path: str, callback: Any = None, poll_interval: float = 2.0) -> None:
        """Start a background thread that polls *path* for changes and hot-reloads.

        Args:
            path: The YAML file to watch.
            callback: Optional zero-argument callable invoked after each successful reload.
            poll_interval: Seconds between file-hash checks. Default 2 s.
        """
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            logger.warning("yaml_rules_v2_watcher_already_running")
            return

        self._watcher_stop.clear()

        def _watch() -> None:
            while not self._watcher_stop.is_set():
                self._watcher_stop.wait(poll_interval)
                if self._watcher_stop.is_set():
                    break
                try:
                    file_path = Path(path)
                    if not file_path.exists():
                        continue
                    raw = file_path.read_text(encoding="utf-8")
                    new_hash = self._hash_content(raw)
                    with self._lock:
                        changed = new_hash != self._last_file_hash
                    if changed:
                        self.reload_rules()
                        if callback is not None:
                            try:
                                callback()
                            except Exception:
                                logger.exception("yaml_rules_v2_watch_callback_error")
                except Exception:
                    logger.exception("yaml_rules_v2_watcher_error", path=path)

        self._watcher_thread = threading.Thread(target=_watch, daemon=True, name="yaml-rules-watcher")
        self._watcher_thread.start()
        logger.info("yaml_rules_v2_watcher_started", path=path, poll_interval=poll_interval)

    def stop_watching(self) -> None:
        """Stop the background file-watcher thread if it is running."""
        self._watcher_stop.set()
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=5.0)
            self._watcher_thread = None
        logger.info("yaml_rules_v2_watcher_stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_rule_with_metrics(
        self, rule: RuleDefinition, context: dict[str, Any]
    ) -> tuple[bool, str]:
        triggered, detail = self._evaluate_condition_group(rule.condition_group, context)
        with self._lock:
            if rule.id in self._stats:
                self._stats[rule.id].record(triggered)
        _RULE_EVALUATIONS.labels(rule_id=rule.id, triggered=str(triggered)).inc()
        return triggered, detail

    def _evaluate_condition_group(
        self, group: ConditionGroup, context: dict[str, Any]
    ) -> tuple[bool, str]:
        results: list[bool] = []
        details: list[str] = []

        for item in group.conditions:
            if isinstance(item, ConditionGroup):
                child_result, child_detail = self._evaluate_condition_group(item, context)
                results.append(child_result)
                details.append(f"({child_detail})")
            else:
                leaf_result = self.evaluate_condition(item, context)
                results.append(leaf_result)
                neg_prefix = "NOT " if item.negate else ""
                details.append(f"{neg_prefix}{item.field} {item.operator} {item.value!r} => {leaf_result}")

        if group.combinator == LogicalCombinator.AND:
            combined = all(results) if results else False
        elif group.combinator == LogicalCombinator.OR:
            combined = any(results) if results else False
        else:  # NOT — validated to have exactly one child
            combined = not results[0] if results else True

        if group.negate:
            combined = not combined

        combinator_str = group.combinator.value
        detail = f"{combinator_str}[{'; '.join(details)}]"
        return combined, detail

    def _combine_scores(
        self,
        rule_set: RuleSet,
        rule_scores: dict[str, float],
        sorted_rules: list[RuleDefinition],
    ) -> float:
        if not rule_scores:
            return min(max(rule_set.default_score, 0.0), 1.0)

        rule_map = {r.id: r for r in sorted_rules}

        if rule_set.combination_method == CombinationMethod.MAX:
            raw = max(rule_scores.values())

        elif rule_set.combination_method == CombinationMethod.WEIGHTED_AVG:
            total_weight = sum(rule_map[rid].weight for rid in rule_scores)
            if total_weight == 0:
                raw = 0.0
            else:
                raw = sum(
                    score * rule_map[rid].weight for rid, score in rule_scores.items()
                ) / total_weight

        else:  # SUM (additive default)
            additive_total = rule_set.default_score
            multiplicative_factor = 1.0
            for rid, score in rule_scores.items():
                rule = rule_map.get(rid)
                if rule and rule.score_method == ScoreMethod.MULTIPLICATIVE:
                    multiplicative_factor *= 1.0 + score
                else:
                    additive_total += score
            raw = additive_total * multiplicative_factor

        return round(min(max(raw, 0.0), 1.0), 6)

    def _set_rule_enabled(self, rule_id: str, *, enabled: bool) -> None:
        with self._lock:
            if self._rule_set is None:
                raise RuntimeError("No rule set loaded; call load_rules() first.")
            updated = [
                r.model_copy(update={"enabled": enabled}) if r.id == rule_id else r
                for r in self._rule_set.rules
            ]
            self._rule_set = self._rule_set.model_copy(update={"rules": updated})
        action = "enabled" if enabled else "disabled"
        logger.info(f"yaml_rules_v2_rule_{action}", rule_id=rule_id)

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def _parse_rule_set(data: dict[str, Any]) -> RuleSet:
        """Parse raw YAML data into a validated :class:`RuleSet`."""
        rules_raw = data.get("rules", [])
        rules: list[RuleDefinition] = []
        for raw in rules_raw:
            condition_group = _parse_condition_group(raw.get("condition_group", {}))
            rules.append(
                RuleDefinition(
                    id=raw["id"],
                    name=raw["name"],
                    description=raw.get("description", ""),
                    condition_group=condition_group,
                    score=float(raw["score"]),
                    score_method=ScoreMethod(raw.get("score_method", "additive")),
                    priority=int(raw.get("priority", 50)),
                    weight=float(raw.get("weight", 1.0)),
                    enabled=bool(raw.get("enabled", True)),
                    tags=raw.get("tags", []),
                )
            )
        return RuleSet(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            rules=rules,
            default_score=float(data.get("default_score", 0.0)),
            combination_method=CombinationMethod(data.get("combination_method", "sum")),
        )


# ---------------------------------------------------------------------------
# YAML parsing helpers (recursive)
# ---------------------------------------------------------------------------


def _parse_condition_group(raw: dict[str, Any]) -> ConditionGroup:
    """Recursively parse a condition group dict from YAML into a :class:`ConditionGroup`."""
    combinator = LogicalCombinator(raw.get("combinator", "AND"))
    negate = bool(raw.get("negate", False))
    children: list[RuleCondition | ConditionGroup] = []

    for item in raw.get("conditions", []):
        if "combinator" in item:
            # Nested group
            children.append(_parse_condition_group(item))
        else:
            children.append(
                RuleCondition(
                    field=item["field"],
                    operator=Operator(item["operator"]),
                    value=item["value"],
                    negate=bool(item.get("negate", False)),
                )
            )

    return ConditionGroup(combinator=combinator, conditions=children, negate=negate)
