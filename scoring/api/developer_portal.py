"""Developer portal API — self-service API key management and docs.

Provides:
  - API key creation/listing/revocation per org
  - SDK quickstart snippets
  - Usage quotas and plan info
  - Webhook test endpoint
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from scoring.api.jwt_auth import CurrentUser, require_user
from scoring.api.rbac import require_permission, PERM_MANAGE_API_KEYS
from scoring.db.engine import get_db_session
from scoring.db.models import APIKey, Organization

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/developer", tags=["developer-portal"])


def _generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        (raw_key, key_hash, key_prefix)
    """
    raw = f"fdp_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    prefix = raw[:12]
    return raw, key_hash, prefix


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    scopes: list[str] = Field(default=["score", "read"])
    rate_limit: int = Field(default=1000, ge=1, le=100_000)
    expires_days: int | None = Field(default=None, ge=1, le=3650)


class APIKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    rate_limit: int
    is_active: bool
    created_at: datetime
    expires_at: datetime | None = None
    # raw_key only returned on creation
    raw_key: str | None = None


class QuickstartResponse(BaseModel):
    python: str
    curl: str
    javascript: str


@router.post("/api-keys", response_model=APIKeyResponse, status_code=201)
async def create_api_key(
    body: CreateAPIKeyRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_permission(PERM_MANAGE_API_KEYS)),
):
    """Create a new API key for the current org. The raw key is shown only once."""
    raw_key, key_hash, key_prefix = _generate_api_key()

    expires_at = None
    if body.expires_days:
        from datetime import timedelta
        expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_days)

    record = APIKey(
        id=uuid.uuid4(),
        org_id=uuid.UUID(current_user.org_id),
        created_by=uuid.UUID(current_user.user_id),
        name=body.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        scopes=body.scopes,
        rate_limit=body.rate_limit,
        is_active=True,
        expires_at=expires_at,
    )

    async with get_db_session() as session:
        session.add(record)

    await logger.ainfo(
        "api_key_created",
        org_id=current_user.org_id,
        key_prefix=key_prefix,
        name=body.name,
    )

    return APIKeyResponse(
        id=str(record.id),
        name=record.name,
        key_prefix=key_prefix,
        scopes=body.scopes,
        rate_limit=body.rate_limit,
        is_active=True,
        created_at=record.created_at or datetime.now(timezone.utc),
        expires_at=expires_at,
        raw_key=raw_key,  # Only shown once
    )


@router.get("/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    request: Request,
    current_user: CurrentUser = Depends(require_user),
):
    """List all API keys for the current org (raw keys never shown)."""
    async with get_db_session() as session:
        result = await session.execute(
            select(APIKey)
            .where(APIKey.org_id == uuid.UUID(current_user.org_id))
            .order_by(APIKey.created_at.desc())
        )
        keys = result.scalars().all()

    return [
        APIKeyResponse(
            id=str(k.id),
            name=k.name,
            key_prefix=k.key_prefix,
            scopes=k.scopes or [],
            rate_limit=k.rate_limit,
            is_active=k.is_active,
            created_at=k.created_at or datetime.now(timezone.utc),
            expires_at=k.expires_at,
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: str,
    request: Request,
    current_user: CurrentUser = Depends(require_permission(PERM_MANAGE_API_KEYS)),
):
    """Revoke (deactivate) an API key."""
    async with get_db_session() as session:
        key = await session.get(APIKey, uuid.UUID(key_id))
        if not key or str(key.org_id) != current_user.org_id:
            raise HTTPException(status_code=404, detail="API key not found")
        key.is_active = False
        session.add(key)

    await logger.ainfo("api_key_revoked", key_id=key_id, org_id=current_user.org_id)


@router.get("/quickstart", response_model=QuickstartResponse)
async def quickstart(
    request: Request,
    current_user: CurrentUser = Depends(require_user),
):
    """Return SDK quickstart code snippets for the current org."""
    base_url = str(request.base_url).rstrip("/")

    python_snippet = f'''import httpx

client = httpx.Client(
    base_url="{base_url}",
    headers={{"X-API-Key": "fdp_YOUR_KEY_HERE"}},
)

response = client.post("/api/v1/score", json={{
    "transaction_id": "txn_001",
    "user_id": "user_123",
    "amount": 150.00,
    "currency": "USD",
    "merchant_id": "merch_abc",
    "timestamp": "2026-05-14T12:00:00Z",
}})
print(response.json())
# {{"transaction_id": "txn_001", "fraud_score": 0.12, "decision": "ALLOW", ...}}
'''

    curl_snippet = f'''curl -X POST "{base_url}/api/v1/score" \\
  -H "X-API-Key: fdp_YOUR_KEY_HERE" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "transaction_id": "txn_001",
    "user_id": "user_123",
    "amount": 150.00,
    "currency": "USD",
    "merchant_id": "merch_abc",
    "timestamp": "2026-05-14T12:00:00Z"
  }}'
'''

    js_snippet = f'''const response = await fetch("{base_url}/api/v1/score", {{
  method: "POST",
  headers: {{
    "X-API-Key": "fdp_YOUR_KEY_HERE",
    "Content-Type": "application/json",
  }},
  body: JSON.stringify({{
    transaction_id: "txn_001",
    user_id: "user_123",
    amount: 150.00,
    currency: "USD",
    merchant_id: "merch_abc",
    timestamp: new Date().toISOString(),
  }}),
}});
const result = await response.json();
console.log(result.fraud_score, result.decision);
'''

    return QuickstartResponse(python=python_snippet, curl=curl_snippet, javascript=js_snippet)


@router.get("/plan")
async def get_plan_info(
    request: Request,
    current_user: CurrentUser = Depends(require_user),
):
    """Return the current org's plan and limits."""
    async with get_db_session() as session:
        org = await session.get(Organization, uuid.UUID(current_user.org_id))

    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    return {
        "org_id": current_user.org_id,
        "plan": org.plan,
        "rate_limit_rps": org.rate_limit_rps,
        "monthly_quota": org.monthly_quota,
        "features": {
            "batch_scoring": org.plan in ("growth", "enterprise"),
            "gnn_model": org.plan == "enterprise",
            "data_export": org.plan in ("growth", "enterprise"),
            "webhook_alerts": True,
            "sla_guarantee": org.plan == "enterprise",
        },
    }
