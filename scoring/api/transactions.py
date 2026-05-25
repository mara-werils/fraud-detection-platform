"""Transaction search and filtering API."""

from __future__ import annotations

from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from shared.schemas import ScoredTransaction

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])


class TransactionSearchResponse(BaseModel):
    """Response for transaction search."""

    transactions: list[ScoredTransaction] = Field(..., description="Matching transactions")
    total: int = Field(..., description="Total matching count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")
    has_more: bool = Field(..., description="Whether more results exist")


class TransactionStatsResponse(BaseModel):
    """Aggregate transaction statistics."""

    total: int
    flagged: int
    avg_score: float
    max_score: float
    unique_users: int


@router.get("", response_model=TransactionSearchResponse)
async def search_transactions(
    request: Request,
    user_id: str | None = Query(None, description="Filter by user ID"),
    min_score: float | None = Query(None, ge=0.0, le=1.0, description="Minimum fraud score"),
    max_score: float | None = Query(None, ge=0.0, le=1.0, description="Maximum fraud score"),
    is_flagged: bool | None = Query(None, description="Filter by flagged status"),
    decision: str | None = Query(None, description="Filter by decision (BLOCK/REVIEW/ALLOW)"),
    start_time: datetime | None = Query(None, description="Start of time range (ISO 8601)"),
    end_time: datetime | None = Query(None, description="End of time range (ISO 8601)"),
    limit: int = Query(50, ge=1, le=500, description="Results per page"),
    offset: int = Query(0, ge=0, description="Result offset"),
) -> TransactionSearchResponse:
    """Search and filter scored transactions.

    Supports filtering by user, score range, flagged status, decision,
    and time range. Results are sorted by scoring time (newest first).
    """
    store = getattr(request.app.state, "transaction_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Transaction store not available")

    results, total = store.search(
        user_id=user_id,
        min_score=min_score,
        max_score=max_score,
        is_flagged=is_flagged,
        decision=decision,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )

    return TransactionSearchResponse(
        transactions=results,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + limit) < total,
    )


@router.get("/stats", response_model=TransactionStatsResponse)
async def transaction_stats(request: Request) -> TransactionStatsResponse:
    """Get aggregate statistics for stored transactions."""
    store = getattr(request.app.state, "transaction_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Transaction store not available")

    stats = store.stats()
    return TransactionStatsResponse(**stats)


@router.get("/{transaction_id}", response_model=ScoredTransaction)
async def get_transaction(transaction_id: str, request: Request) -> ScoredTransaction:
    """Get a specific scored transaction by ID."""
    store = getattr(request.app.state, "transaction_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Transaction store not available")

    txn = store.get(transaction_id)
    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    return txn
