"""Multi-tenant organization management API."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from scoring.db.engine import get_db_session
from scoring.db.models import Organization

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/organizations", tags=["organizations"])

PLANS = {
    "starter": {"rps": 100, "monthly_quota": 1_000_000},
    "growth": {"rps": 500, "monthly_quota": 10_000_000},
    "enterprise": {"rps": 5000, "monthly_quota": 500_000_000},
}


class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    slug: str = Field(..., min_length=2, max_length=100, pattern=r"^[a-z0-9-]+$")
    plan: str = Field(default="starter")

    @field_validator("plan")
    @classmethod
    def validate_plan(cls, v: str) -> str:
        if v not in PLANS:
            raise ValueError(f"Plan must be one of {list(PLANS.keys())}")
        return v


class OrganizationUpdate(BaseModel):
    name: str | None = None
    plan: str | None = None
    is_active: bool | None = None
    settings: dict | None = None

    @field_validator("plan")
    @classmethod
    def validate_plan(cls, v: str | None) -> str | None:
        if v is not None and v not in PLANS:
            raise ValueError(f"Plan must be one of {list(PLANS.keys())}")
        return v


class OrganizationResponse(BaseModel):
    id: str
    name: str
    slug: str
    plan: str
    is_active: bool
    rate_limit_rps: int
    monthly_quota: int
    created_at: datetime

    model_config = {"from_attributes": True}


@router.post("", response_model=OrganizationResponse, status_code=201)
async def create_organization(body: OrganizationCreate, request: Request):
    """Create a new tenant organization."""
    plan_limits = PLANS[body.plan]
    org = Organization(
        id=uuid.uuid4(),
        name=body.name,
        slug=body.slug,
        plan=body.plan,
        rate_limit_rps=plan_limits["rps"],
        monthly_quota=plan_limits["monthly_quota"],
    )

    async with get_db_session() as session:
        existing = await session.execute(
            select(Organization).where(Organization.slug == body.slug)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail=f"Slug '{body.slug}' already taken")

        session.add(org)

    await logger.ainfo("organization_created", org_id=str(org.id), slug=org.slug, plan=org.plan)
    return OrganizationResponse(
        id=str(org.id),
        name=org.name,
        slug=org.slug,
        plan=org.plan,
        is_active=org.is_active,
        rate_limit_rps=org.rate_limit_rps,
        monthly_quota=org.monthly_quota,
        created_at=org.created_at or datetime.now(UTC),
    )


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(org_id: str, request: Request):
    """Retrieve an organization by ID."""
    async with get_db_session() as session:
        org = await session.get(Organization, uuid.UUID(org_id))
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")

    return OrganizationResponse(
        id=str(org.id),
        name=org.name,
        slug=org.slug,
        plan=org.plan,
        is_active=org.is_active,
        rate_limit_rps=org.rate_limit_rps,
        monthly_quota=org.monthly_quota,
        created_at=org.created_at or datetime.now(UTC),
    )


@router.patch("/{org_id}", response_model=OrganizationResponse)
async def update_organization(org_id: str, body: OrganizationUpdate, request: Request):
    """Update an organization."""
    async with get_db_session() as session:
        org = await session.get(Organization, uuid.UUID(org_id))
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")

        if body.name is not None:
            org.name = body.name
        if body.is_active is not None:
            org.is_active = body.is_active
        if body.settings is not None:
            org.settings = body.settings
        if body.plan is not None:
            plan_limits = PLANS[body.plan]
            org.plan = body.plan
            org.rate_limit_rps = plan_limits["rps"]
            org.monthly_quota = plan_limits["monthly_quota"]

        session.add(org)

    await logger.ainfo("organization_updated", org_id=org_id)
    return OrganizationResponse(
        id=str(org.id),
        name=org.name,
        slug=org.slug,
        plan=org.plan,
        is_active=org.is_active,
        rate_limit_rps=org.rate_limit_rps,
        monthly_quota=org.monthly_quota,
        created_at=org.created_at or datetime.now(UTC),
    )


@router.get("")
async def list_organizations(request: Request, limit: int = 50, offset: int = 0):
    """List all organizations (admin only)."""
    async with get_db_session() as session:
        result = await session.execute(
            select(Organization).order_by(Organization.created_at.desc()).limit(limit).offset(offset)
        )
        orgs = result.scalars().all()

    return {
        "items": [
            OrganizationResponse(
                id=str(o.id),
                name=o.name,
                slug=o.slug,
                plan=o.plan,
                is_active=o.is_active,
                rate_limit_rps=o.rate_limit_rps,
                monthly_quota=o.monthly_quota,
                created_at=o.created_at or datetime.now(UTC),
            )
            for o in orgs
        ],
        "limit": limit,
        "offset": offset,
    }
