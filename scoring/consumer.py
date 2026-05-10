"""Kafka consumer loop for scoring transactions."""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

import orjson
import structlog
from redis.asyncio import Redis

from scoring.models.ensemble import Decision, EnsembleScorer
from shared.kafka_utils import KafkaProducerWrapper
from shared.metrics import MetricsRegistry
from shared.schemas import (
    AlertEvent,
    AlertSeverity,
    FeatureVector,
    ScoredTransaction,
    Transaction,
)

logger = structlog.get_logger(__name__)


def _severity_from_score(fraud_score: float) -> AlertSeverity:
    """Map a fraud score to an alert severity level.

    Args:
        fraud_score: Fraud probability in [0, 1].

    Returns:
        AlertSeverity based on score ranges.
    """
    if fraud_score >= 0.9:
        return AlertSeverity.CRITICAL
    if fraud_score >= 0.8:
        return AlertSeverity.HIGH
    if fraud_score >= 0.5:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


async def _fetch_features(
    redis: Redis | None,
    user_id: UUID,
    transaction_id: UUID,
) -> FeatureVector:
    """Fetch feature vector from Redis, falling back to defaults.

    Args:
        redis: Async Redis connection (may be None).
        user_id: The user whose features to look up.
        transaction_id: The transaction ID to associate features with.

    Returns:
        FeatureVector from Redis or a default-valued vector.
    """
    if redis is None:
        return FeatureVector(transaction_id=transaction_id, user_id=user_id)

    try:
        key = f"features:{user_id}"
        raw = await redis.get(key)
        if raw is not None:
            data = orjson.loads(raw)
            data["transaction_id"] = str(transaction_id)
            data["user_id"] = str(user_id)
            return FeatureVector.model_validate(data)
    except Exception:
        await logger.aexception("feature_fetch_error", user_id=str(user_id))

    return FeatureVector(transaction_id=transaction_id, user_id=user_id)


async def handle_message(
    message: dict[str, Any],
    scorer: EnsembleScorer,
    producer: KafkaProducerWrapper,
    redis: Redis | None,
    metrics: MetricsRegistry,
    topic_scored: str,
    topic_alerts: str,
) -> ScoredTransaction:
    """Process a single raw transaction message.

    Args:
        message: Deserialized Kafka message payload.
        scorer: The ensemble scorer instance.
        producer: Kafka producer for publishing results.
        redis: Redis connection for feature lookup.
        metrics: Prometheus metrics registry.
        topic_scored: Kafka topic to publish scored transactions.
        topic_alerts: Kafka topic to publish alerts.

    Returns:
        The scored transaction.
    """
    start = time.perf_counter()

    transaction = Transaction.model_validate(message)
    metrics.transactions_received.labels(transaction_type=transaction.transaction_type.value).inc()

    features = await _fetch_features(redis, transaction.user_id, transaction.transaction_id)

    scored = scorer.score(transaction, features)

    metrics.transactions_scored.labels(model_version=scored.model_version).inc()

    scoring_duration = time.perf_counter() - start
    metrics.scoring_latency.labels(model_version=scored.model_version).observe(scoring_duration)

    # Publish scored transaction
    publish_start = time.perf_counter()
    await producer.send(
        topic=topic_scored,
        value=scored.to_json_bytes(),
        key=str(scored.transaction_id),
    )
    metrics.kafka_publish_latency.labels(topic=topic_scored).observe(
        time.perf_counter() - publish_start
    )

    # Publish alert if flagged
    decision = EnsembleScorer.decide(scored.fraud_score)
    if decision in (Decision.BLOCK, Decision.REVIEW):
        severity = _severity_from_score(scored.fraud_score)
        alert = AlertEvent(
            transaction_id=scored.transaction_id,
            user_id=scored.user_id,
            fraud_score=scored.fraud_score,
            severity=severity,
            reason=scored.flag_reason or f"Fraud score {scored.fraud_score:.2f}",
            amount=scored.amount,
            currency=scored.currency,
        )

        alert_publish_start = time.perf_counter()
        await producer.send(
            topic=topic_alerts,
            value=alert.to_json_bytes(),
            key=str(alert.transaction_id),
        )
        metrics.kafka_publish_latency.labels(topic=topic_alerts).observe(
            time.perf_counter() - alert_publish_start
        )
        metrics.transactions_flagged.labels(severity=severity.value).inc()
        metrics.alerts_generated.labels(severity=severity.value).inc()

        await logger.ainfo(
            "transaction_flagged",
            transaction_id=str(scored.transaction_id),
            fraud_score=scored.fraud_score,
            decision=decision.value,
            severity=severity.value,
        )

    metrics.end_to_end_latency.observe(time.perf_counter() - start)
    return scored


def create_consumer_handler(
    scorer: EnsembleScorer,
    producer: KafkaProducerWrapper,
    redis: Redis | None,
    metrics: MetricsRegistry,
    topic_scored: str,
    topic_alerts: str,
) -> Any:
    """Create a message handler closure for the Kafka consumer loop.

    Args:
        scorer: The ensemble scorer.
        producer: Kafka producer for output topics.
        redis: Redis connection for feature lookup.
        metrics: Metrics registry.
        topic_scored: Topic for scored transactions.
        topic_alerts: Topic for alerts.

    Returns:
        An async handler function compatible with KafkaConsumerWrapper.consume().
    """

    async def handler(message: dict[str, Any]) -> None:
        try:
            await handle_message(
                message=message,
                scorer=scorer,
                producer=producer,
                redis=redis,
                metrics=metrics,
                topic_scored=topic_scored,
                topic_alerts=topic_alerts,
            )
        except Exception:
            metrics.transaction_errors.labels(error_type="scoring_error").inc()
            await logger.aexception("scoring_pipeline_error")

    return handler
