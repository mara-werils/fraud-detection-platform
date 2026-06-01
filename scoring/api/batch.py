"""Batch scoring API for processing multiple transactions at once."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from shared.schemas import FeatureVector, ScoredTransaction, Transaction

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/batch", tags=["batch"])

MAX_BATCH_SIZE = 1000
MAX_CONCURRENT_SCORES = 64  # Limit concurrent scoring to prevent resource exhaustion


class BatchScoreRequest(BaseModel):
    """Request body for batch scoring."""

    transactions: list[Transaction] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of transactions to score (max {MAX_BATCH_SIZE})",
    )


class BatchScoreResponse(BaseModel):
    """Response for batch scoring."""

    results: list[ScoredTransaction] = Field(..., description="Scored transactions")
    total: int = Field(..., description="Total transactions processed")
    failed: int = Field(default=0, description="Number of failed scorings")
    latency_ms: float = Field(..., description="Total batch processing time in ms")
    errors: list[dict[str, Any]] = Field(default_factory=list, description="Error details")


class BatchStatsResponse(BaseModel):
    """Summary statistics for a batch."""

    total: int
    flagged: int
    blocked: int
    review: int
    allowed: int
    avg_score: float
    max_score: float
    min_score: float
    avg_latency_ms: float


@router.post("/score", response_model=BatchScoreResponse)
async def batch_score(batch: BatchScoreRequest, request: Request) -> BatchScoreResponse:
    """Score multiple transactions in a single request.

    Transactions are scored concurrently for maximum throughput.
    Individual failures don't block other transactions.

    Args:
        batch: List of transactions to score.
        request: HTTP request for accessing app state.

    Returns:
        BatchScoreResponse with all results and error details.
    """
    scorer = request.app.state.scorer
    redis = request.app.state.redis

    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    # Score all transactions with bounded concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCORES)

    async def _bounded_score(txn: Transaction) -> ScoredTransaction:
        async with semaphore:
            return await _score_single(txn, scorer, redis)

    tasks = [_bounded_score(txn) for txn in batch.transactions]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[ScoredTransaction] = []
    errors: list[dict[str, Any]] = []
    failed = 0

    for i, outcome in enumerate(outcomes):
        if isinstance(outcome, Exception):
            failed += 1
            errors.append({
                "index": i,
                "transaction_id": str(batch.transactions[i].transaction_id),
                "error": str(outcome),
            })
        else:
            results.append(outcome)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    await logger.ainfo(
        "batch_scored",
        total=len(batch.transactions),
        succeeded=len(results),
        failed=failed,
        latency_ms=round(elapsed_ms, 2),
    )

    return BatchScoreResponse(
        results=results,
        total=len(batch.transactions),
        failed=failed,
        latency_ms=round(elapsed_ms, 2),
        errors=errors,
    )


@router.post("/score/stats", response_model=BatchStatsResponse)
async def batch_score_with_stats(
    batch: BatchScoreRequest, request: Request
) -> BatchStatsResponse:
    """Score a batch and return summary statistics only.

    Useful when you need aggregate metrics without individual results.

    Args:
        batch: List of transactions to score.
        request: HTTP request.

    Returns:
        Summary statistics for the batch.
    """
    response = await batch_score(batch, request)

    scores = [r.fraud_score for r in response.results]
    if not scores:
        raise HTTPException(status_code=500, detail="All transactions failed scoring")

    flagged = sum(1 for r in response.results if r.is_flagged)
    blocked = sum(1 for r in response.results if r.fraud_score >= 0.8)
    review = sum(1 for r in response.results if 0.5 <= r.fraud_score < 0.8)

    return BatchStatsResponse(
        total=len(response.results),
        flagged=flagged,
        blocked=blocked,
        review=review,
        allowed=len(response.results) - flagged,
        avg_score=sum(scores) / len(scores),
        max_score=max(scores),
        min_score=min(scores),
        avg_latency_ms=sum(r.scoring_latency_ms for r in response.results) / len(response.results),
    )


async def _score_single(
    txn: Transaction,
    scorer: Any,
    redis: Any,
) -> ScoredTransaction:
    """Score a single transaction with feature retrieval."""
    features = FeatureVector(
        transaction_id=txn.transaction_id,
        user_id=txn.user_id,
    )

    if redis is not None:
        try:
            import orjson

            key = f"features:{txn.user_id}"
            raw = await redis.get(key)
            if raw is not None:
                data = orjson.loads(raw)
                data["transaction_id"] = str(txn.transaction_id)
                data["user_id"] = str(txn.user_id)
                features = FeatureVector.model_validate(data)
        except Exception:
            pass

    return await scorer.score(txn, features)
