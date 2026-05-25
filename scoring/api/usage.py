"""Usage and billing API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request

from scoring.api.jwt_auth import CurrentUser, require_user
from scoring.api.rbac import require_analyst

router = APIRouter(prefix="/api/v1/usage", tags=["usage"])


@router.get("/me")
async def get_my_usage(
    request: Request,
    days: int = 30,
    current_user: CurrentUser = Depends(require_user),
):
    """Return usage statistics for the current user's organization."""
    usage_meter = getattr(request.app.state, "usage_meter", None)
    if not usage_meter:
        raise HTTPException(status_code=503, detail="Usage metering not available")

    since = datetime.now(UTC) - timedelta(days=days)
    stats = await usage_meter.get_org_usage(current_user.org_id, since=since)
    realtime = await usage_meter.get_realtime_counts(current_user.org_id)
    return {**stats, **realtime, "period_days": days}


@router.get("/org/{org_id}")
async def get_org_usage(
    org_id: str,
    request: Request,
    days: int = 30,
    current_user: CurrentUser = Depends(require_analyst),
):
    """Return usage statistics for a specific organization (analyst+)."""
    usage_meter = getattr(request.app.state, "usage_meter", None)
    if not usage_meter:
        raise HTTPException(status_code=503, detail="Usage metering not available")

    since = datetime.now(UTC) - timedelta(days=days)
    stats = await usage_meter.get_org_usage(org_id, since=since)
    return {**stats, "period_days": days}
