"""Rule-based fraud scoring engine (MVP implementation)."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from shared.schemas import FeatureVector, Transaction

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RuleResult:
    """Result from a single rule evaluation."""

    rule_name: str
    score: float
    triggered: bool
    detail: str


@dataclass
class ScoringResult:
    """Aggregated result from all rules."""

    fraud_score: float
    rule_scores: dict[str, float]
    triggered_rules: list[str]
    details: list[str]


class RuleEngine:
    """Rule-based fraud scorer with configurable thresholds.

    Each rule is a separate method for testability. The final fraud score
    is the sum of individual rule scores, clamped to [0, 1].
    """

    def __init__(
        self,
        high_amount_multiplier: float = 3.0,
        high_amount_score: float = 0.3,
        new_device_score: float = 0.2,
        velocity_threshold: int = 5,
        velocity_score: float = 0.15,
        geo_distance_km: float = 500.0,
        geo_score: float = 0.25,
        night_start_hour: int = 1,
        night_end_hour: int = 5,
        night_score: float = 0.1,
        international_score: float = 0.15,
    ) -> None:
        self.high_amount_multiplier = high_amount_multiplier
        self.high_amount_score = high_amount_score
        self.new_device_score = new_device_score
        self.velocity_threshold = velocity_threshold
        self.velocity_score = velocity_score
        self.geo_distance_km = geo_distance_km
        self.geo_score = geo_score
        self.night_start_hour = night_start_hour
        self.night_end_hour = night_end_hour
        self.night_score = night_score
        self.international_score = international_score

    def score(self, transaction: Transaction, features: FeatureVector) -> ScoringResult:
        """Score a transaction against all rules.

        Args:
            transaction: The raw transaction to score.
            features: Pre-computed feature vector for the transaction.

        Returns:
            ScoringResult with fraud_score clamped to [0, 1].
        """
        rules = [
            self.rule_high_amount(transaction, features),
            self.rule_new_device(transaction, features),
            self.rule_velocity(transaction, features),
            self.rule_geo_anomaly(transaction, features),
            self.rule_night_transaction(transaction, features),
            self.rule_international(transaction, features),
        ]

        total_score = sum(r.score for r in rules)
        clamped_score = min(max(total_score, 0.0), 1.0)

        rule_scores = {r.rule_name: r.score for r in rules}
        triggered_rules = [r.rule_name for r in rules if r.triggered]
        details = [r.detail for r in rules if r.triggered]

        return ScoringResult(
            fraud_score=clamped_score,
            rule_scores=rule_scores,
            triggered_rules=triggered_rules,
            details=details,
        )

    def rule_high_amount(self, transaction: Transaction, features: FeatureVector) -> RuleResult:
        """High amount rule: triggers if amount > multiplier * user average.

        Args:
            transaction: The transaction being scored.
            features: Feature vector with historical averages.

        Returns:
            RuleResult indicating whether this rule triggered.
        """
        if features.txn_amount_avg_24h <= 0:
            return RuleResult(
                rule_name="high_amount",
                score=0.0,
                triggered=False,
                detail="No historical average available",
            )

        ratio = float(transaction.amount) / float(features.txn_amount_avg_24h)
        triggered = ratio > self.high_amount_multiplier

        return RuleResult(
            rule_name="high_amount",
            score=self.high_amount_score if triggered else 0.0,
            triggered=triggered,
            detail=f"Amount ratio {ratio:.2f}x vs avg (threshold: {self.high_amount_multiplier}x)",
        )

    def rule_new_device(self, transaction: Transaction, features: FeatureVector) -> RuleResult:
        """New device rule: triggers if the device has not been seen before.

        Args:
            transaction: The transaction being scored.
            features: Feature vector with device history.

        Returns:
            RuleResult indicating whether this rule triggered.
        """
        triggered = features.is_new_device

        return RuleResult(
            rule_name="new_device",
            score=self.new_device_score if triggered else 0.0,
            triggered=triggered,
            detail=f"Device {'is new' if triggered else 'is known'}",
        )

    def rule_velocity(self, transaction: Transaction, features: FeatureVector) -> RuleResult:
        """Velocity rule: triggers if >N transactions in the last hour.

        Args:
            transaction: The transaction being scored.
            features: Feature vector with velocity counts.

        Returns:
            RuleResult indicating whether this rule triggered.
        """
        triggered = features.txn_count_1h > self.velocity_threshold

        return RuleResult(
            rule_name="velocity",
            score=self.velocity_score if triggered else 0.0,
            triggered=triggered,
            detail=f"{features.txn_count_1h} txns in 1h (threshold: {self.velocity_threshold})",
        )

    def rule_geo_anomaly(self, transaction: Transaction, features: FeatureVector) -> RuleResult:
        """Geo anomaly rule: triggers if distance from last txn exceeds threshold.

        Args:
            transaction: The transaction being scored.
            features: Feature vector with location data.

        Returns:
            RuleResult indicating whether this rule triggered.
        """
        distance = features.distance_from_last_txn_km
        triggered = distance > self.geo_distance_km

        return RuleResult(
            rule_name="geo_anomaly",
            score=self.geo_score if triggered else 0.0,
            triggered=triggered,
            detail=f"Distance {distance:.1f}km from last txn (threshold: {self.geo_distance_km}km)",
        )

    def rule_night_transaction(
        self, transaction: Transaction, features: FeatureVector
    ) -> RuleResult:
        """Night transaction rule: triggers if transaction is between 1-5 AM.

        Args:
            transaction: The transaction being scored.
            features: Feature vector (unused for this rule).

        Returns:
            RuleResult indicating whether this rule triggered.
        """
        hour = transaction.timestamp.hour
        triggered = self.night_start_hour <= hour < self.night_end_hour

        return RuleResult(
            rule_name="night_transaction",
            score=self.night_score if triggered else 0.0,
            triggered=triggered,
            detail=f"Transaction at {hour:02d}:00 ({'night window' if triggered else 'normal hours'})",
        )

    def rule_international(self, transaction: Transaction, features: FeatureVector) -> RuleResult:
        """International transaction rule: triggers on unexpected international activity.

        Uses unique_countries_24h > 1 as a proxy for unexpected international
        transactions in the MVP. Will be refined with user profile data later.

        Args:
            transaction: The transaction being scored.
            features: Feature vector with country counts.

        Returns:
            RuleResult indicating whether this rule triggered.
        """
        triggered = features.unique_countries_24h > 1

        return RuleResult(
            rule_name="international",
            score=self.international_score if triggered else 0.0,
            triggered=triggered,
            detail=f"{features.unique_countries_24h} countries in 24h",
        )
