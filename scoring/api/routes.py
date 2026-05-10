"""API routes for the scoring service."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from prometheus_client import generate_latest
from starlette.responses import Response

from shared.schemas import (
    FeatureVector,
    ScoredTransaction,
    Transaction,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/score", response_model=ScoredTransaction)
async def score_transaction(transaction: Transaction, request: Request) -> ScoredTransaction:
    """Synchronous scoring endpoint.

    Accepts a Transaction JSON payload, scores it using the loaded model,
    and returns a ScoredTransaction response.

    Args:
        transaction: The transaction to score.
        request: The incoming HTTP request (for accessing app state).

    Returns:
        The scored transaction with fraud probability and metadata.
    """
    scorer = request.app.state.scorer
    redis = request.app.state.redis

    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build a default feature vector; in production this would come from Redis
    features = FeatureVector(
        transaction_id=transaction.transaction_id,
        user_id=transaction.user_id,
    )

    if redis is not None:
        try:
            import orjson

            key = f"features:{transaction.user_id}"
            raw = await redis.get(key)
            if raw is not None:
                data = orjson.loads(raw)
                data["transaction_id"] = str(transaction.transaction_id)
                data["user_id"] = str(transaction.user_id)
                features = FeatureVector.model_validate(data)
        except Exception:
            await logger.aexception(
                "feature_fetch_error_api",
                user_id=str(transaction.user_id),
            )

    scored = scorer.score(transaction, features)

    await logger.ainfo(
        "transaction_scored_api",
        transaction_id=str(scored.transaction_id),
        fraud_score=scored.fraud_score,
        is_flagged=scored.is_flagged,
    )

    return scored


@router.get("/health")
async def health_check(request: Request) -> dict[str, Any]:
    """Health check endpoint.

    Reports the status of Kafka connections, Redis, and model loading.

    Args:
        request: The incoming HTTP request.

    Returns:
        Dictionary with component health statuses.
    """
    consumer = request.app.state.consumer
    producer = request.app.state.producer
    redis = request.app.state.redis
    scorer = request.app.state.scorer

    kafka_healthy = (consumer is not None and consumer.is_healthy) and (
        producer is not None and producer.is_healthy
    )

    redis_healthy = False
    if redis is not None:
        try:
            await redis.ping()
            redis_healthy = True
        except Exception:
            redis_healthy = False

    model_loaded = scorer is not None

    overall = kafka_healthy and redis_healthy and model_loaded
    status_code = 200 if overall else 503

    body = {
        "status": "healthy" if overall else "unhealthy",
        "components": {
            "kafka": "connected" if kafka_healthy else "disconnected",
            "redis": "connected" if redis_healthy else "disconnected",
            "model": "loaded" if model_loaded else "not_loaded",
        },
    }

    return Response(
        content=__import__("orjson").dumps(body),
        media_type="application/json",
        status_code=status_code,
    )


@router.get("/health/ready")
async def readiness_probe(request: Request) -> dict[str, Any]:
    """Readiness probe for Kubernetes.

    Returns 200 only when the service is fully ready to handle traffic.

    Args:
        request: The incoming HTTP request.

    Returns:
        Readiness status dictionary.
    """
    scorer = request.app.state.scorer
    ready = scorer is not None

    status_code = 200 if ready else 503
    body = {"ready": ready}

    return Response(
        content=__import__("orjson").dumps(body),
        media_type="application/json",
        status_code=status_code,
    )


@router.get("/model/info")
async def model_info(request: Request) -> dict[str, Any]:
    """Return information about the currently loaded model.

    Args:
        request: The incoming HTTP request.

    Returns:
        Model version, type, and runtime metrics.
    """
    scorer = request.app.state.scorer

    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_version": scorer.model_version,
        "model_type": scorer.model_type,
        "total_scored": scorer.total_scored,
        "total_flagged": scorer.total_flagged,
        "flag_rate": (
            scorer.total_flagged / scorer.total_scored if scorer.total_scored > 0 else 0.0
        ),
    }


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus metrics endpoint.

    Returns:
        Plain-text Prometheus metrics exposition format.
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
