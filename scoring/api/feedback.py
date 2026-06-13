"""Feedback API for analyst decisions on scored transactions."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from scoring.services.feedback_store import FeedbackEntry, FeedbackStats

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Request to submit analyst feedback."""

    transaction_id: str = Field(..., description="Transaction ID being reviewed")
    is_fraud: bool = Field(..., description="Analyst determination: True if fraud")
    analyst: str = Field(default="unknown", description="Analyst name")
    notes: str | None = Field(default=None, description="Additional notes")


class FeedbackListResponse(BaseModel):
    """Response for feedback listing."""

    entries: list[FeedbackEntry]
    total: int


@router.post("", response_model=FeedbackEntry)
async def submit_feedback(req: FeedbackRequest, request: Request) -> FeedbackEntry:
    """Submit analyst feedback on a scored transaction.

    This feedback is used to:
    1. Measure model accuracy (false positive/negative rates)
    2. Generate labeled training data for model retraining
    3. Update case status if a case exists
    """
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Feedback store not available")

    # Look up original score and decision
    txn_store = getattr(request.app.state, "transaction_store", None)
    original_score = None
    original_decision = None

    if txn_store:
        txn = txn_store.get(req.transaction_id)
        if txn:
            original_score = txn.fraud_score
            if txn.fraud_score >= 0.8:
                original_decision = "BLOCK"
            elif txn.fraud_score >= 0.5:
                original_decision = "REVIEW"
            else:
                original_decision = "ALLOW"

    entry = store.add(
        transaction_id=req.transaction_id,
        is_fraud=req.is_fraud,
        analyst=req.analyst,
        notes=req.notes,
        original_score=original_score,
        original_decision=original_decision,
    )

    # Auto-update case if one exists
    case_mgr = getattr(request.app.state, "case_manager", None)
    if case_mgr:
        case = case_mgr.get_case_by_transaction(req.transaction_id)
        if case:
            status = "resolved_fraud" if req.is_fraud else "resolved_legitimate"
            case_mgr.update_status(
                case.case_id,
                status=status,
                actor=req.analyst,
                notes=req.notes,
            )

    return entry


@router.get("", response_model=FeedbackListResponse)
async def list_feedback(
    request: Request,
    analyst: str | None = Query(None, description="Filter by analyst"),
    is_fraud: bool | None = Query(None, description="Filter by fraud determination"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> FeedbackListResponse:
    """List feedback entries with optional filtering."""
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Feedback store not available")

    entries, total = store.list_entries(
        analyst=analyst,
        is_fraud=is_fraud,
        limit=limit,
        offset=offset,
    )
    return FeedbackListResponse(entries=entries, total=total)


@router.get("/stats", response_model=FeedbackStats)
async def feedback_stats(request: Request) -> FeedbackStats:
    """Get feedback statistics including model accuracy metrics.

    Returns false positive/negative rates, agreement rate,
    and per-analyst activity counts.
    """
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Feedback store not available")

    return store.stats()


@router.get("/export")
async def export_training_data(request: Request) -> dict[str, Any]:
    """Export feedback as labeled training data for model retraining.

    Returns all feedback entries formatted for ML training pipelines.
    """
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Feedback store not available")

    data = store.export_training_data()
    return {
        "total": len(data),
        "format": "labeled_transactions",
        "data": data,
    }


__all__ = ["FeedbackListResponse", "FeedbackRequest", "router"]
