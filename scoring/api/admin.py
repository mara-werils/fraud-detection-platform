"""Admin API — platform-wide management for administrators.

Provides:
  - Platform-wide metrics and statistics
  - Organization management (list, suspend, change plan)
  - User management (list, deactivate, change role)
  - System health overview
  - Model management (get version, trigger retrain)
  - Feature flag management
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select

from scoring.api.jwt_auth import CurrentUser
from scoring.api.rbac import require_admin
from scoring.db.engine import get_db_session
from scoring.db.models import APIKey, Organization, TransactionRecord, User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# ── Platform overview ─────────────────────────────────────────────────────────


@router.get("/stats")
async def platform_stats(
    request: Request,
    current_user: CurrentUser = Depends(require_admin),
):
    """Return platform-wide statistics."""
    async with get_db_session() as session:
        org_count = await session.scalar(select(func.count(Organization.id)))
        active_org_count = await session.scalar(
            select(func.count(Organization.id)).where(Organization.is_active)
        )
        user_count = await session.scalar(select(func.count(User.id)))
        active_user_count = await session.scalar(
            select(func.count(User.id)).where(User.is_active)
        )
        txn_count = await session.scalar(select(func.count(TransactionRecord.id)))
        api_key_count = await session.scalar(
            select(func.count(APIKey.id)).where(APIKey.is_active)
        )

    # Real-time system stats
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    batch_queue = getattr(request.app.state, "batch_queue", None)
    ws_hub = getattr(request.app.state, "ws_hub", None)
    redis_pool = getattr(request.app.state, "redis_pool", None)
    _usage_meter = getattr(request.app.state, "usage_meter", None)
    feature_cache = getattr(request.app.state, "feature_cache", None)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "tenants": {
            "total": org_count or 0,
            "active": active_org_count or 0,
        },
        "users": {
            "total": user_count or 0,
            "active": active_user_count or 0,
        },
        "api_keys": {"active": api_key_count or 0},
        "transactions": {"total_scored": txn_count or 0},
        "slo": slo_monitor.report() if slo_monitor else {},
        "batch_queue": batch_queue.stats() if batch_queue else {},
        "websocket": ws_hub.stats() if ws_hub else {},
        "redis": redis_pool.stats() if redis_pool else {},
        "feature_cache": feature_cache.stats() if feature_cache else {},
    }


# ── Organization management ───────────────────────────────────────────────────


@router.get("/organizations")
async def list_organizations(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    plan: str | None = None,
    is_active: bool | None = None,
    current_user: CurrentUser = Depends(require_admin),
):
    """List all organizations with filters."""
    async with get_db_session() as session:
        q = select(Organization).order_by(Organization.created_at.desc())
        if plan:
            q = q.where(Organization.plan == plan)
        if is_active is not None:
            q = q.where(Organization.is_active == is_active)
        q = q.limit(limit).offset(offset)
        result = await session.execute(q)
        orgs = result.scalars().all()

    return {
        "items": [
            {
                "id": str(o.id),
                "name": o.name,
                "slug": o.slug,
                "plan": o.plan,
                "is_active": o.is_active,
                "rate_limit_rps": o.rate_limit_rps,
                "monthly_quota": o.monthly_quota,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
            for o in orgs
        ],
        "limit": limit,
        "offset": offset,
    }


class OrgSuspendRequest(BaseModel):
    reason: str = ""


@router.post("/organizations/{org_id}/suspend")
async def suspend_organization(
    org_id: str,
    body: OrgSuspendRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_admin),
):
    """Suspend an organization (disables all API access)."""
    async with get_db_session() as session:
        org = await session.get(Organization, uuid.UUID(org_id))
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        org.is_active = False
        session.add(org)

    await logger.awarning(
        "org_suspended",
        org_id=org_id,
        by=current_user.user_id,
        reason=body.reason,
    )
    return {"org_id": org_id, "status": "suspended"}


@router.post("/organizations/{org_id}/activate")
async def activate_organization(
    org_id: str,
    request: Request,
    current_user: CurrentUser = Depends(require_admin),
):
    """Re-activate a suspended organization."""
    async with get_db_session() as session:
        org = await session.get(Organization, uuid.UUID(org_id))
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        org.is_active = True
        session.add(org)

    await logger.ainfo("org_activated", org_id=org_id, by=current_user.user_id)
    return {"org_id": org_id, "status": "active"}


# ── User management ───────────────────────────────────────────────────────────


@router.get("/users")
async def list_users(
    request: Request,
    org_id: str | None = None,
    role: str | None = None,
    limit: int = 50,
    offset: int = 0,
    current_user: CurrentUser = Depends(require_admin),
):
    """List all users across the platform."""
    async with get_db_session() as session:
        q = select(User).order_by(User.created_at.desc())
        if org_id:
            q = q.where(User.org_id == uuid.UUID(org_id))
        if role:
            q = q.where(User.role == role)
        q = q.limit(limit).offset(offset)
        result = await session.execute(q)
        users = result.scalars().all()

    return {
        "items": [
            {
                "id": str(u.id),
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role,
                "org_id": str(u.org_id),
                "is_active": u.is_active,
                "last_login": u.last_login.isoformat() if u.last_login else None,
            }
            for u in users
        ],
        "limit": limit,
        "offset": offset,
    }


class ChangeRoleRequest(BaseModel):
    role: str


@router.patch("/users/{user_id}/role")
async def change_user_role(
    user_id: str,
    body: ChangeRoleRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_admin),
):
    """Change a user's role."""
    valid_roles = {"admin", "analyst", "viewer", "api_user"}
    if body.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of {valid_roles}")

    async with get_db_session() as session:
        user = await session.get(User, uuid.UUID(user_id))
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        old_role = user.role
        user.role = body.role
        session.add(user)

    await logger.ainfo(
        "user_role_changed",
        user_id=user_id,
        old_role=old_role,
        new_role=body.role,
        by=current_user.user_id,
    )
    return {"user_id": user_id, "role": body.role}


# ── Model management ──────────────────────────────────────────────────────────


@router.get("/model/info")
async def model_info(
    request: Request,
    current_user: CurrentUser = Depends(require_admin),
):
    """Return current model version and statistics."""
    scorer = getattr(request.app.state, "scorer", None)
    if not scorer:
        raise HTTPException(status_code=503, detail="Scorer not available")

    return {
        "model_version": getattr(scorer, "model_version", "unknown"),
        "threshold_block": getattr(scorer, "threshold_block", 0.8),
        "threshold_review": getattr(scorer, "threshold_review", 0.5),
    }


# ── Feature flags ─────────────────────────────────────────────────────────────


_FEATURE_FLAGS: dict[str, bool] = {
    "batch_scoring_enabled": True,
    "gnn_model_enabled": False,
    "llm_explanations_enabled": True,
    "websocket_enabled": True,
    "usage_metering_enabled": True,
}


@router.get("/feature-flags")
async def get_feature_flags(
    current_user: CurrentUser = Depends(require_admin),
):
    return _FEATURE_FLAGS


@router.patch("/feature-flags/{flag_name}")
async def set_feature_flag(
    flag_name: str,
    request: Request,
    current_user: CurrentUser = Depends(require_admin),
):
    body = await request.json()
    enabled = body.get("enabled")
    if flag_name not in _FEATURE_FLAGS:
        raise HTTPException(status_code=404, detail=f"Unknown flag: {flag_name}")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="'enabled' must be a boolean")
    _FEATURE_FLAGS[flag_name] = enabled
    await logger.ainfo("feature_flag_changed", flag=flag_name, enabled=enabled, by=current_user.user_id)
    return {"flag": flag_name, "enabled": enabled}
