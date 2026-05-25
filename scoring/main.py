"""FastAPI application for the scoring service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis

from scoring.api.ab_testing import router as ab_router
from scoring.api.analytics import router as analytics_router
from scoring.api.auth import register_api_key
from scoring.api.batch import router as batch_router
from scoring.api.cases import router as cases_router
from scoring.api.drift import router as drift_router
from scoring.api.entity_lists import router as entity_lists_router
from scoring.api.explainability import router as explainability_router
from scoring.api.export import router as export_router
from scoring.api.feedback import router as feedback_router
from scoring.api.middleware import setup_middleware
from scoring.api.routes import router
from scoring.api.sanctions import router as sanctions_router
from scoring.api.transactions import router as transactions_router
from scoring.api.versioning import VersioningMiddleware
from scoring.api.webhooks import router as webhooks_router
from scoring.config import ScoringConfig
from scoring.consumer import create_consumer_handler
from scoring.models.ensemble import EnsembleScorer
from scoring.models.rule_engine import RuleEngine
from scoring.plugins.registry import PluginRegistry
from scoring.services.alert_routing import AlertRoutingEngine
from scoring.services.analytics_engine import AnalyticsEngine
from scoring.services.case_manager import CaseManager
from scoring.services.currency_risk import CurrencyRiskService
from scoring.services.device_fingerprint import DeviceFingerprintService
from scoring.services.drift_detector import DriftDetector
from scoring.services.entity_lists import EntityListService
from scoring.services.event_store import EventStore
from scoring.services.explainability import ExplainabilityService
from scoring.services.feedback_store import FeedbackStore
from scoring.services.graph_analysis import TransactionGraphAnalyzer
from scoring.services.ip_intelligence import IPIntelligenceService
from scoring.services.merchant_risk import MerchantRiskService
from scoring.services.sanctions_screener import SanctionsScreener
from scoring.services.session_analyzer import SessionAnalyzer
from scoring.services.shadow_mode import ShadowModeService
from scoring.services.transaction_store import TransactionStore
from scoring.services.webhook_delivery import WebhookDeliveryService
from scoring.services.webhook_manager import WebhookManager
from shared.kafka_utils import KafkaConsumerWrapper, KafkaProducerWrapper
from shared.logging import setup_logging
from shared.metrics import create_metrics

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler.

    Initialises Kafka consumer/producer, Redis connection, the scoring
    model, and all platform services on startup. Tears everything down
    gracefully on shutdown.
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

    # Authentication
    app.state.auth_enabled = config.api_auth_enabled
    if config.api_key:
        register_api_key(config.api_key, name="default", rate_limit=10000)

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
        consumer_task = asyncio.create_task(consumer.consume(handler), name="scoring-consumer")
    except Exception:
        await logger.awarning("kafka_consumer_start_failed")
        consumer = None
    app.state.consumer = consumer

    # Platform services
    app.state.transaction_store = TransactionStore()
    app.state.case_manager = CaseManager()
    app.state.feedback_store = FeedbackStore()
    app.state.drift_detector = DriftDetector()
    app.state.sanctions_screener = SanctionsScreener()
    app.state.webhook_manager = WebhookManager()
    app.state.analytics_engine = AnalyticsEngine()
    app.state.device_fingerprint = DeviceFingerprintService()
    app.state.ip_intelligence = IPIntelligenceService()
    app.state.graph_analyzer = TransactionGraphAnalyzer()
    app.state.merchant_risk = MerchantRiskService()
    app.state.entity_lists = EntityListService()
    app.state.event_store = EventStore()
    app.state.shadow_mode = ShadowModeService()
    app.state.session_analyzer = SessionAnalyzer()
    app.state.alert_routing = AlertRoutingEngine()
    app.state.webhook_delivery = WebhookDeliveryService()
    app.state.currency_risk = CurrencyRiskService()
    app.state.explainability = ExplainabilityService()
    app.state.plugin_registry = PluginRegistry()

    await logger.ainfo(
        "scoring_service_started",
        model_version=config.model_version,
        kafka="connected" if producer and consumer else "degraded",
        redis="connected" if redis else "degraded",
        services=[
            "transaction_store", "case_manager", "feedback_store",
            "drift_detector", "sanctions_screener", "webhook_manager",
            "analytics_engine", "device_fingerprint", "ip_intelligence",
            "graph_analyzer", "merchant_risk", "entity_lists", "event_store",
            "shadow_mode", "session_analyzer", "alert_routing",
            "webhook_delivery", "currency_risk", "explainability",
            "plugin_registry",
        ],
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
        Fully configured FastAPI instance with all routers and middleware.
    """
    app = FastAPI(
        title="Fraud Detection Platform",
        description=(
            "Real-time ML-powered fraud detection platform with ensemble scoring, "
            "case management, drift detection, and analyst workflows."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    setup_middleware(app)

    # Core routes
    app.include_router(router)

    # API v1 routes
    app.include_router(batch_router)
    app.include_router(transactions_router)
    app.include_router(cases_router)
    app.include_router(feedback_router)
    app.include_router(drift_router)
    app.include_router(ab_router)
    app.include_router(explainability_router)
    app.include_router(sanctions_router)
    app.include_router(webhooks_router)
    app.include_router(export_router)
    app.include_router(analytics_router)
    app.include_router(entity_lists_router)

    # API versioning middleware
    app.add_middleware(VersioningMiddleware)

    return app


app = create_app()
