"""Case management API for fraud investigation workflows."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from scoring.services.case_manager import FraudCase

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/cases", tags=["cases"])


class CreateCaseRequest(BaseModel):
    """Request to create a new case."""

    transaction_id: str = Field(..., description="Transaction ID to investigate")
    priority: str = Field(default="medium", description="Priority: low/medium/high/critical")
    assigned_to: str | None = Field(default=None, description="Analyst to assign")
    notes: str | None = Field(default=None, description="Initial investigation notes")
    tags: list[str] = Field(default_factory=list, description="Case tags")


class UpdateCaseRequest(BaseModel):
    """Request to update a case."""

    status: str | None = Field(default=None, description="New status")
    assigned_to: str | None = Field(default=None, description="New assignee")
    notes: str | None = Field(default=None, description="Additional notes")
    actor: str = Field(default="api", description="Who is making the change")


class CaseListResponse(BaseModel):
    """Response for case listing."""

    cases: list[FraudCase]
    total: int
    limit: int
    offset: int


class CaseStatsResponse(BaseModel):
    """Case statistics."""

    total: int
    by_status: dict[str, int]
    by_priority: dict[str, int]
    avg_resolution_hours: float


@router.post("", response_model=FraudCase)
async def create_case(req: CreateCaseRequest, request: Request) -> FraudCase:
    """Create a new fraud investigation case."""
    case_mgr = getattr(request.app.state, "case_manager", None)
    if case_mgr is None:
        raise HTTPException(status_code=503, detail="Case manager not available")

    # Look up transaction details from store
    txn_store = getattr(request.app.state, "transaction_store", None)
    user_id = ""
    fraud_score = 0.0
    amount = 0.0
    currency = "USD"

    if txn_store:
        txn = txn_store.get(req.transaction_id)
        if txn:
            user_id = str(txn.user_id)
            fraud_score = txn.fraud_score
            amount = float(txn.amount)
            currency = txn.currency

    case = case_mgr.create_case(
        transaction_id=req.transaction_id,
        user_id=user_id,
        fraud_score=fraud_score,
        amount=amount,
        currency=currency,
        priority=req.priority,
        assigned_to=req.assigned_to,
        notes=req.notes,
        tags=req.tags,
    )

    return case


@router.get("", response_model=CaseListResponse)
async def list_cases(
    request: Request,
    status: str | None = Query(None, description="Filter by status"),
    priority: str | None = Query(None, description="Filter by priority"),
    assigned_to: str | None = Query(None, description="Filter by assignee"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> CaseListResponse:
    """List fraud cases with optional filtering."""
    case_mgr = getattr(request.app.state, "case_manager", None)
    if case_mgr is None:
        raise HTTPException(status_code=503, detail="Case manager not available")

    cases, total = case_mgr.list_cases(
        status=status,
        priority=priority,
        assigned_to=assigned_to,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )

    return CaseListResponse(cases=cases, total=total, limit=limit, offset=offset)


@router.get("/stats", response_model=CaseStatsResponse)
async def case_stats(request: Request) -> CaseStatsResponse:
    """Get case statistics."""
    case_mgr = getattr(request.app.state, "case_manager", None)
    if case_mgr is None:
        raise HTTPException(status_code=503, detail="Case manager not available")

    stats = case_mgr.stats()
    return CaseStatsResponse(**stats)


@router.get("/{case_id}", response_model=FraudCase)
async def get_case(case_id: str, request: Request) -> FraudCase:
    """Get a specific case by ID."""
    case_mgr = getattr(request.app.state, "case_manager", None)
    if case_mgr is None:
        raise HTTPException(status_code=503, detail="Case manager not available")

    case = case_mgr.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    return case


@router.patch("/{case_id}", response_model=FraudCase)
async def update_case(
    case_id: str, req: UpdateCaseRequest, request: Request
) -> FraudCase:
    """Update a case's status, assignment, or add notes."""
    case_mgr = getattr(request.app.state, "case_manager", None)
    if case_mgr is None:
        raise HTTPException(status_code=503, detail="Case manager not available")

    case = case_mgr.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    if req.status:
        case = case_mgr.update_status(case_id, req.status, actor=req.actor, notes=req.notes)

    if req.assigned_to:
        case = case_mgr.assign(case_id, req.assigned_to, actor=req.actor)

    if req.notes and not req.status:
        case = case_mgr.add_note(case_id, author=req.actor, content=req.notes)

    if case is None:
        raise HTTPException(status_code=500, detail="Failed to update case")

    return case


__all__ = ["CaseListResponse", "CaseStatsResponse", "CreateCaseRequest", "UpdateCaseRequest", "router"]
