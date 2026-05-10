"""Pytest fixtures for the scoring service tests."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from scoring.models.ensemble import EnsembleScorer
from scoring.models.rule_engine import RuleEngine
from shared.schemas import FeatureVector, Transaction, TransactionType


@pytest.fixture
def rule_engine() -> RuleEngine:
    """Provide a default RuleEngine instance."""
    return RuleEngine()


@pytest.fixture
def scorer() -> EnsembleScorer:
    """Provide a default EnsembleScorer instance."""
    return EnsembleScorer()


@pytest.fixture
def user_id() -> UUID:
    """Provide a stable test user ID."""
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def transaction_id() -> UUID:
    """Provide a stable test transaction ID."""
    return UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


@pytest.fixture
def normal_transaction(user_id: UUID, transaction_id: UUID) -> Transaction:
    """A normal, low-risk transaction."""
    return Transaction(
        transaction_id=transaction_id,
        user_id=user_id,
        amount=Decimal("25.00"),
        currency="USD",
        transaction_type=TransactionType.PURCHASE,
        merchant_id="merchant_001",
        device_id="device_known",
        timestamp=datetime(2025, 6, 15, 14, 30, 0),
    )


@pytest.fixture
def suspicious_transaction(user_id: UUID, transaction_id: UUID) -> Transaction:
    """A high-risk transaction (high amount, night time)."""
    return Transaction(
        transaction_id=transaction_id,
        user_id=user_id,
        amount=Decimal("9999.99"),
        currency="USD",
        transaction_type=TransactionType.TRANSFER,
        device_id="device_new",
        timestamp=datetime(2025, 6, 15, 3, 0, 0),  # 3 AM
    )


@pytest.fixture
def normal_features(user_id: UUID, transaction_id: UUID) -> FeatureVector:
    """Feature vector representing normal user behavior."""
    return FeatureVector(
        transaction_id=transaction_id,
        user_id=user_id,
        txn_count_1h=2,
        txn_count_24h=5,
        txn_amount_avg_24h=Decimal("30.00"),
        distance_from_last_txn_km=5.0,
        is_new_device=False,
        unique_countries_24h=1,
    )


@pytest.fixture
def suspicious_features(user_id: UUID, transaction_id: UUID) -> FeatureVector:
    """Feature vector with multiple anomalous signals."""
    return FeatureVector(
        transaction_id=transaction_id,
        user_id=user_id,
        txn_count_1h=8,
        txn_count_24h=20,
        txn_amount_avg_24h=Decimal("50.00"),
        distance_from_last_txn_km=1200.0,
        is_new_device=True,
        unique_countries_24h=3,
    )


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Provide a mock async Redis client."""
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def mock_producer() -> MagicMock:
    """Provide a mock Kafka producer."""
    producer = MagicMock()
    producer.is_healthy = True
    producer.start = AsyncMock()
    producer.send = AsyncMock()
    producer.stop = AsyncMock()
    return producer


@pytest.fixture
def mock_consumer() -> MagicMock:
    """Provide a mock Kafka consumer."""
    consumer = MagicMock()
    consumer.is_healthy = True
    consumer.start = AsyncMock()
    consumer.consume = AsyncMock()
    consumer.stop = AsyncMock()
    return consumer


@pytest.fixture
def test_client(
    scorer: EnsembleScorer,
    mock_redis: AsyncMock,
    mock_producer: MagicMock,
    mock_consumer: MagicMock,
) -> TestClient:
    """Provide a FastAPI test client with mocked dependencies."""
    from scoring.main import create_app

    app = create_app()

    # Override lifespan by setting state directly
    app.state.scorer = scorer
    app.state.redis = mock_redis
    app.state.producer = mock_producer
    app.state.consumer = mock_consumer

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
