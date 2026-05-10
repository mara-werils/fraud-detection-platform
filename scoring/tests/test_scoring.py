"""Tests for the scoring service."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from fastapi.testclient import TestClient

from scoring.models.ensemble import Decision, EnsembleScorer
from scoring.models.rule_engine import RuleEngine
from shared.schemas import FeatureVector, Transaction, TransactionType


class TestRuleEngine:
    """Tests for the rule-based scoring engine."""

    def test_high_amount_triggers(
        self,
        rule_engine: RuleEngine,
        suspicious_transaction: Transaction,
        suspicious_features: FeatureVector,
    ) -> None:
        """High amount rule should trigger when amount far exceeds user average."""
        result = rule_engine.rule_high_amount(suspicious_transaction, suspicious_features)
        assert result.triggered is True
        assert result.score == 0.3
        assert result.rule_name == "high_amount"

    def test_high_amount_does_not_trigger(
        self,
        rule_engine: RuleEngine,
        normal_transaction: Transaction,
        normal_features: FeatureVector,
    ) -> None:
        """High amount rule should not trigger for normal amounts."""
        result = rule_engine.rule_high_amount(normal_transaction, normal_features)
        assert result.triggered is False
        assert result.score == 0.0

    def test_velocity_triggers(
        self,
        rule_engine: RuleEngine,
        suspicious_transaction: Transaction,
        suspicious_features: FeatureVector,
    ) -> None:
        """Velocity rule should trigger when txn count exceeds threshold."""
        result = rule_engine.rule_velocity(suspicious_transaction, suspicious_features)
        assert result.triggered is True
        assert result.score == 0.15

    def test_velocity_does_not_trigger(
        self,
        rule_engine: RuleEngine,
        normal_transaction: Transaction,
        normal_features: FeatureVector,
    ) -> None:
        """Velocity rule should not trigger with normal transaction count."""
        result = rule_engine.rule_velocity(normal_transaction, normal_features)
        assert result.triggered is False
        assert result.score == 0.0

    def test_geo_anomaly_triggers(
        self,
        rule_engine: RuleEngine,
        suspicious_transaction: Transaction,
        suspicious_features: FeatureVector,
    ) -> None:
        """Geo anomaly rule should trigger when distance exceeds threshold."""
        result = rule_engine.rule_geo_anomaly(suspicious_transaction, suspicious_features)
        assert result.triggered is True
        assert result.score == 0.25

    def test_geo_anomaly_does_not_trigger(
        self,
        rule_engine: RuleEngine,
        normal_transaction: Transaction,
        normal_features: FeatureVector,
    ) -> None:
        """Geo anomaly rule should not trigger for nearby transactions."""
        result = rule_engine.rule_geo_anomaly(normal_transaction, normal_features)
        assert result.triggered is False
        assert result.score == 0.0

    def test_new_device_triggers(
        self,
        rule_engine: RuleEngine,
        suspicious_transaction: Transaction,
        suspicious_features: FeatureVector,
    ) -> None:
        """New device rule should trigger when device is new."""
        result = rule_engine.rule_new_device(suspicious_transaction, suspicious_features)
        assert result.triggered is True
        assert result.score == 0.2

    def test_night_transaction_triggers(
        self,
        rule_engine: RuleEngine,
        suspicious_transaction: Transaction,
        suspicious_features: FeatureVector,
    ) -> None:
        """Night rule should trigger for transactions between 1-5 AM."""
        result = rule_engine.rule_night_transaction(suspicious_transaction, suspicious_features)
        assert result.triggered is True
        assert result.score == 0.1

    def test_night_transaction_does_not_trigger(
        self,
        rule_engine: RuleEngine,
        normal_transaction: Transaction,
        normal_features: FeatureVector,
    ) -> None:
        """Night rule should not trigger for daytime transactions."""
        result = rule_engine.rule_night_transaction(normal_transaction, normal_features)
        assert result.triggered is False
        assert result.score == 0.0

    def test_international_triggers(
        self,
        rule_engine: RuleEngine,
        suspicious_transaction: Transaction,
        suspicious_features: FeatureVector,
    ) -> None:
        """International rule should trigger when multiple countries detected."""
        result = rule_engine.rule_international(suspicious_transaction, suspicious_features)
        assert result.triggered is True
        assert result.score == 0.15

    def test_full_score_suspicious(
        self,
        rule_engine: RuleEngine,
        suspicious_transaction: Transaction,
        suspicious_features: FeatureVector,
    ) -> None:
        """Full scoring of a suspicious transaction should yield a high score."""
        result = rule_engine.score(suspicious_transaction, suspicious_features)
        assert result.fraud_score > 0.8
        assert len(result.triggered_rules) >= 4
        assert "high_amount" in result.triggered_rules
        assert "geo_anomaly" in result.triggered_rules

    def test_full_score_legitimate(
        self,
        rule_engine: RuleEngine,
        normal_transaction: Transaction,
        normal_features: FeatureVector,
    ) -> None:
        """Full scoring of a normal transaction should yield a low score."""
        result = rule_engine.score(normal_transaction, normal_features)
        assert result.fraud_score < 0.3
        assert len(result.triggered_rules) == 0

    def test_score_clamped_to_one(self, rule_engine: RuleEngine) -> None:
        """Fraud score should be clamped to a maximum of 1.0."""
        user_id = UUID("12345678-1234-5678-1234-567812345678")
        txn_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

        txn = Transaction(
            transaction_id=txn_id,
            user_id=user_id,
            amount=Decimal("99999.99"),
            transaction_type=TransactionType.WITHDRAWAL,
            timestamp=datetime(2025, 6, 15, 2, 0, 0),
        )
        features = FeatureVector(
            transaction_id=txn_id,
            user_id=user_id,
            txn_count_1h=20,
            txn_amount_avg_24h=Decimal("10.00"),
            distance_from_last_txn_km=5000.0,
            is_new_device=True,
            unique_countries_24h=5,
        )

        result = rule_engine.score(txn, features)
        assert result.fraud_score == 1.0


class TestEnsembleScorer:
    """Tests for the ensemble scorer and decision thresholds."""

    def test_decision_block(self) -> None:
        """Score >= 0.8 should result in BLOCK decision."""
        assert EnsembleScorer.decide(0.85) == Decision.BLOCK
        assert EnsembleScorer.decide(0.8) == Decision.BLOCK
        assert EnsembleScorer.decide(1.0) == Decision.BLOCK

    def test_decision_review(self) -> None:
        """Score in [0.5, 0.8) should result in REVIEW decision."""
        assert EnsembleScorer.decide(0.5) == Decision.REVIEW
        assert EnsembleScorer.decide(0.65) == Decision.REVIEW
        assert EnsembleScorer.decide(0.79) == Decision.REVIEW

    def test_decision_allow(self) -> None:
        """Score < 0.5 should result in ALLOW decision."""
        assert EnsembleScorer.decide(0.0) == Decision.ALLOW
        assert EnsembleScorer.decide(0.3) == Decision.ALLOW
        assert EnsembleScorer.decide(0.49) == Decision.ALLOW

    async def test_scorer_returns_scored_transaction(
        self,
        scorer: EnsembleScorer,
        normal_transaction: Transaction,
        normal_features: FeatureVector,
    ) -> None:
        """Scorer should return a valid ScoredTransaction."""
        result = await scorer.score(normal_transaction, normal_features)
        assert result.transaction_id == normal_transaction.transaction_id
        assert result.model_version == "ensemble-v1"
        assert result.scoring_latency_ms >= 0
        assert 0.0 <= result.fraud_score <= 1.0

    async def test_scorer_flags_suspicious(
        self,
        scorer: EnsembleScorer,
        suspicious_transaction: Transaction,
        suspicious_features: FeatureVector,
    ) -> None:
        """Scorer should flag a suspicious transaction."""
        result = await scorer.score(suspicious_transaction, suspicious_features)
        assert result.is_flagged is True
        assert result.flag_reason is not None

    async def test_scorer_does_not_flag_normal(
        self,
        scorer: EnsembleScorer,
        normal_transaction: Transaction,
        normal_features: FeatureVector,
    ) -> None:
        """Scorer should not flag a normal transaction."""
        result = await scorer.score(normal_transaction, normal_features)
        assert result.is_flagged is False

    async def test_scorer_tracks_counts(
        self,
        scorer: EnsembleScorer,
        normal_transaction: Transaction,
        normal_features: FeatureVector,
    ) -> None:
        """Scorer should track total scored and flagged counts."""
        initial_scored = scorer.total_scored
        await scorer.score(normal_transaction, normal_features)
        assert scorer.total_scored == initial_scored + 1


class TestAPI:
    """Tests for the scoring API endpoints."""

    def test_score_endpoint(self, test_client: TestClient, normal_transaction: Transaction) -> None:
        """POST /score should return a scored transaction."""
        payload = normal_transaction.model_dump(mode="json")
        response = test_client.post("/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_score" in data
        assert "model_version" in data
        assert 0.0 <= data["fraud_score"] <= 1.0

    def test_health_endpoint(self, test_client: TestClient) -> None:
        """GET /health should return component statuses."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "kafka" in data["components"]
        assert "redis" in data["components"]
        assert "model" in data["components"]

    def test_readiness_endpoint(self, test_client: TestClient) -> None:
        """GET /health/ready should return ready status."""
        response = test_client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True

    def test_model_info_endpoint(self, test_client: TestClient) -> None:
        """GET /model/info should return model metadata."""
        response = test_client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "ensemble-v1"
        assert data["model_type"] == "ensemble"

    def test_metrics_endpoint(self, test_client: TestClient) -> None:
        """GET /metrics should return Prometheus metrics."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
