"""FastAPI application for the scoring service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis

from shared.kafka_utils import KafkaConsumerWrapper, KafkaProducerWrapper
from shared.logging import setup_logging
from shared.metrics import create_metrics

from scoring.api.middleware import setup_middleware
from scoring.api.routes import router
from scoring.config import ScoringConfig
from scoring.consumer import create_consumer_handler
from scoring.models.ensemble import EnsembleScorer
from scoring.models.rule_engine import RuleEngine

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler.

    Initialises Kafka consumer/producer, Redis connection, and the scoring
    model on startup. Tears everything down gracefully on shutdown.
    """
    config = ScoringConfig()

    setup_logging(
        service_name=config.service_name,
        log_level=config.log_level,
        log_format=config.log_format,
    )

    await logger.ainfo("scoring_service_starting", version=config.service_version)

    # Metrics
    metrics = create_metrics(config.service_name)
    metrics.set_service_info(version=config.service_version)

    # Model
    rule_engine = RuleEngine()
    scorer = EnsembleScorer(
        model_version=config.model_version,
        threshold_block=config.score_threshold_block,
        threshold_review=config.score_threshold_review,
        rule_engine=rule_engine,
    )
    app.state.scorer = scorer

    # Redis
    redis: Redis | None = None
    try:
        redis = Redis.from_url(
            config.redis_url,
            max_connections=config.redis_max_connections,
            socket_timeout=config.redis_socket_timeout,
            decode_responses=False,
        )
        await redis.ping()
        await logger.ainfo("redis_connected", url=config.redis_url)
    except Exception:
        await logger.awarning("redis_connection_failed_starting_without_cache")
        redis = None
    app.state.redis = redis

    # Kafka producer
    producer: KafkaProducerWrapper | None = None
    try:
        producer = KafkaProducerWrapper(
            bootstrap_servers=config.kafka_bootstrap_servers,
            max_retries=config.kafka_max_retries,
            retry_backoff_ms=config.kafka_retry_backoff_ms,
        )
        await producer.start()
    except Exception:
        await logger.awarning("kafka_producer_start_failed")
        producer = None
    app.state.producer = producer

    # Kafka consumer
    consumer: KafkaConsumerWrapper | None = None
    consumer_task: asyncio.Task[None] | None = None
    try:
        consumer = KafkaConsumerWrapper(
            bootstrap_servers=config.kafka_bootstrap_servers,
            topics=[config.kafka_topic_raw_txn],
            group_id=config.kafka_consumer_group,
            max_retries=config.kafka_max_retries,
            retry_backoff_ms=config.kafka_retry_backoff_ms,
        )
        await consumer.start()

        handler = create_consumer_handler(
            scorer=scorer,
            producer=producer,
            redis=redis,
            metrics=metrics,
            topic_scored=config.kafka_topic_scored,
            topic_alerts=config.kafka_topic_alerts,
        )
        consumer_task = asyncio.create_task(
            consumer.consume(handler), name="scoring-consumer"
        )
    except Exception:
        await logger.awarning("kafka_consumer_start_failed")
        consumer = None
    app.state.consumer = consumer

    await logger.ainfo(
        "scoring_service_started",
        model_version=config.model_version,
        kafka="connected" if producer and consumer else "degraded",
        redis="connected" if redis else "degraded",
    )

    yield

    # Shutdown
    await logger.ainfo("scoring_service_stopping")

    if consumer_task is not None:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

    if consumer is not None:
        await consumer.stop()
    if producer is not None:
        await producer.stop()
    if redis is not None:
        await redis.aclose()

    await logger.ainfo("scoring_service_stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Fully configured FastAPI instance.
    """
    app = FastAPI(
        title="Fraud Detection Scoring Service",
        description="Real-time transaction fraud scoring API",
        version="0.1.0",
        lifespan=lifespan,
    )

    setup_middleware(app)
    app.include_router(router)

    return app


app = create_app()
