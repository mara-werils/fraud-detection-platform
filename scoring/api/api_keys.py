"""API key management with rotation, scoped permissions, and usage analytics.

Provides full lifecycle management for API keys including generation with
prefixed tokens, SHA-256 hashing for storage, key rotation with grace periods,
scoped permissions, per-key rate limits, usage tracking, and automatic
expiration/revocation.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


# ── Permission scopes ────────────────────────────────────────────────────────


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


ALL_PERMISSIONS = frozenset(Permission)

# Admin implies read+write
PERMISSION_HIERARCHY: dict[Permission, frozenset[Permission]] = {
    Permission.ADMIN: frozenset({Permission.READ, Permission.WRITE, Permission.ADMIN}),
    Permission.WRITE: frozenset({Permission.READ, Permission.WRITE}),
    Permission.READ: frozenset({Permission.READ}),
}


def expand_permissions(scopes: set[str]) -> frozenset[str]:
    """Expand permissions according to hierarchy (admin implies read+write)."""
    expanded: set[str] = set()
    for scope in scopes:
        try:
            perm = Permission(scope)
            expanded |= PERMISSION_HIERARCHY[perm]
        except (ValueError, KeyError):
            expanded.add(scope)
    return frozenset(expanded)


# ── Key hashing ──────────────────────────────────────────────────────────────


def _hash_key(raw_key: str) -> str:
    """Hash an API key using SHA-256 for secure storage."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _generate_raw_key(prefix: str = "sk_live") -> str:
    """Generate a prefixed API key with 32 random bytes."""
    token = secrets.token_urlsafe(32)
    return f"{prefix}_{token}"


# ── Data models ──────────────────────────────────────────────────────────────


@dataclass
class UsageRecord:
    """Tracks per-endpoint usage for an API key."""

    total_requests: int = 0
    last_used: datetime | None = None
    requests_by_endpoint: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: int = 0


@dataclass
class APIKey:
    """Represents a managed API key with metadata."""

    id: str
    name: str
    key_hash: str
    prefix: str
    scopes: set[str]
    rate_limit_rpm: int
    created_at: datetime
    expires_at: datetime | None
    is_active: bool = True
    revoked_at: datetime | None = None
    rotated_from: str | None = None
    grace_period_end: datetime | None = None
    usage: UsageRecord = field(default_factory=UsageRecord)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    @property
    def is_in_grace_period(self) -> bool:
        if self.grace_period_end is None:
            return False
        return datetime.now(UTC) <= self.grace_period_end

    @property
    def effective_permissions(self) -> frozenset[str]:
        return expand_permissions(self.scopes)

    def has_permission(self, permission: str) -> bool:
        return permission in self.effective_permissions


# ── In-memory store ──────────────────────────────────────────────────────────


class APIKeyStore:
    """Thread-safe in-memory API key store.

    In production, replace with a database-backed implementation.
    """

    def __init__(self) -> None:
        self._keys_by_hash: dict[str, APIKey] = {}
        self._keys_by_id: dict[str, APIKey] = {}
        self._lock = threading.Lock()

    def add(self, key: APIKey) -> None:
        with self._lock:
            self._keys_by_hash[key.key_hash] = key
            self._keys_by_id[key.id] = key

    def get_by_hash(self, key_hash: str) -> APIKey | None:
        return self._keys_by_hash.get(key_hash)

    def get_by_id(self, key_id: str) -> APIKey | None:
        return self._keys_by_id.get(key_id)

    def list_all(self) -> list[APIKey]:
        return list(self._keys_by_id.values())

    def remove(self, key_id: str) -> bool:
        with self._lock:
            key = self._keys_by_id.pop(key_id, None)
            if key is None:
                return False
            self._keys_by_hash.pop(key.key_hash, None)
            return True

    def clear(self) -> None:
        with self._lock:
            self._keys_by_hash.clear()
            self._keys_by_id.clear()


# Global store instance
_store = APIKeyStore()


def get_store() -> APIKeyStore:
    """Return the global API key store (useful for dependency injection)."""
    return _store


# ── Core operations ──────────────────────────────────────────────────────────


def create_api_key(
    name: str,
    scopes: set[str] | None = None,
    rate_limit_rpm: int = 1000,
    expires_in_days: int | None = None,
    prefix: str = "sk_live",
) -> tuple[str, APIKey]:
    """Generate a new API key and store its hash.

    Args:
        name: Human-readable label.
        scopes: Permission scopes (defaults to read-only).
        rate_limit_rpm: Max requests per minute.
        expires_in_days: Days until expiration (None = never).
        prefix: Key prefix (e.g. sk_live, sk_test).

    Returns:
        Tuple of (raw_key, APIKey record). The raw key is only available at
        creation time.
    """
    if scopes is None:
        scopes = {Permission.READ}

    invalid = scopes - ALL_PERMISSIONS
    if invalid:
        raise ValueError(f"Invalid scopes: {invalid}")

    raw_key = _generate_raw_key(prefix)
    key_hash = _hash_key(raw_key)
    now = datetime.now(UTC)

    key_record = APIKey(
        id=str(uuid4()),
        name=name,
        key_hash=key_hash,
        prefix=f"{raw_key[:len(prefix)+1]}...",
        scopes=scopes,
        rate_limit_rpm=rate_limit_rpm,
        created_at=now,
        expires_at=now + timedelta(days=expires_in_days) if expires_in_days else None,
    )

    _store.add(key_record)
    logger.info("api_key_created", key_id=key_record.id, name=name, scopes=list(scopes))
    return raw_key, key_record


def validate_key(raw_key: str) -> APIKey | None:
    """Validate a raw API key. Returns the key record if valid and active."""
    key_hash = _hash_key(raw_key)
    key = _store.get_by_hash(key_hash)

    if key is None:
        return None

    # Check revocation (but allow grace period for rotated keys)
    if not key.is_active and not key.is_in_grace_period:
        return None

    # Check expiration and auto-revoke
    if key.is_expired:
        _auto_revoke(key, reason="expired")
        return None

    return key


def _auto_revoke(key: APIKey, reason: str = "expired") -> None:
    """Automatically revoke an expired or compromised key."""
    key.is_active = False
    key.revoked_at = datetime.now(UTC)
    logger.info("api_key_auto_revoked", key_id=key.id, reason=reason)


def revoke_key(key_id: str) -> bool:
    """Manually revoke an API key by its ID."""
    key = _store.get_by_id(key_id)
    if key is None:
        return False
    key.is_active = False
    key.revoked_at = datetime.now(UTC)
    logger.info("api_key_revoked", key_id=key_id)
    return True


def rotate_key(
    key_id: str,
    grace_period_hours: int = 24,
) -> tuple[str, APIKey] | None:
    """Rotate an API key, keeping the old one valid during a grace period.

    Args:
        key_id: ID of the key to rotate.
        grace_period_hours: Hours the old key remains valid.

    Returns:
        Tuple of (new_raw_key, new_APIKey) or None if key_id not found.
    """
    old_key = _store.get_by_id(key_id)
    if old_key is None:
        return None

    now = datetime.now(UTC)

    # Create the replacement key with the same settings
    new_raw, new_key = create_api_key(
        name=old_key.name,
        scopes=old_key.scopes.copy(),
        rate_limit_rpm=old_key.rate_limit_rpm,
        expires_in_days=(
            int((old_key.expires_at - now).days) if old_key.expires_at and old_key.expires_at > now else None
        ),
        prefix=old_key.prefix.split("_")[0] + "_" + old_key.prefix.split("_")[1].rstrip("."),
    )
    new_key.rotated_from = old_key.id

    # Mark the old key with a grace period instead of immediate revocation
    old_key.is_active = False
    old_key.revoked_at = now
    old_key.grace_period_end = now + timedelta(hours=grace_period_hours)

    logger.info(
        "api_key_rotated",
        old_key_id=old_key.id,
        new_key_id=new_key.id,
        grace_period_hours=grace_period_hours,
    )
    return new_raw, new_key


def record_usage(key: APIKey, endpoint: str, is_error: bool = False) -> None:
    """Record a request against an API key for analytics."""
    key.usage.total_requests += 1
    key.usage.last_used = datetime.now(UTC)
    key.usage.requests_by_endpoint[endpoint] += 1
    if is_error:
        key.usage.errors += 1


def get_usage_analytics(key_id: str) -> dict | None:
    """Return usage analytics for a given key."""
    key = _store.get_by_id(key_id)
    if key is None:
        return None
    return {
        "key_id": key.id,
        "name": key.name,
        "total_requests": key.usage.total_requests,
        "errors": key.usage.errors,
        "last_used": key.usage.last_used.isoformat() if key.usage.last_used else None,
        "requests_by_endpoint": dict(key.usage.requests_by_endpoint),
        "error_rate": (
            round(key.usage.errors / key.usage.total_requests, 4)
            if key.usage.total_requests > 0
            else 0.0
        ),
    }


def cleanup_expired_keys() -> int:
    """Revoke all expired keys and remove keys past their grace period.

    Returns:
        Number of keys cleaned up.
    """
    now = datetime.now(UTC)
    cleaned = 0
    for key in _store.list_all():
        if key.is_expired and key.is_active:
            _auto_revoke(key, reason="expired")
            cleaned += 1
        elif (
            not key.is_active
            and key.grace_period_end is not None
            and now > key.grace_period_end
        ):
            _store.remove(key.id)
            cleaned += 1
    return cleaned


# ── Pydantic schemas for the API ─────────────────────────────────────────────


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    scopes: list[str] = Field(default=["read"])
    rate_limit_rpm: int = Field(default=1000, ge=1, le=100_000)
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


class CreateKeyResponse(BaseModel):
    raw_key: str
    key_id: str
    name: str
    prefix: str
    scopes: list[str]
    rate_limit_rpm: int
    created_at: str
    expires_at: str | None


class KeyInfoResponse(BaseModel):
    key_id: str
    name: str
    prefix: str
    scopes: list[str]
    rate_limit_rpm: int
    is_active: bool
    created_at: str
    expires_at: str | None
    revoked_at: str | None
    rotated_from: str | None


class RotateKeyRequest(BaseModel):
    grace_period_hours: int = Field(default=24, ge=0, le=720)


class RotateKeyResponse(BaseModel):
    new_raw_key: str
    new_key_id: str
    old_key_id: str
    grace_period_end: str


class UsageAnalyticsResponse(BaseModel):
    key_id: str
    name: str
    total_requests: int
    errors: int
    error_rate: float
    last_used: str | None
    requests_by_endpoint: dict[str, int]


class UpdateKeyRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    scopes: list[str] | None = None
    rate_limit_rpm: int | None = Field(default=None, ge=1, le=100_000)


# ── Helper to convert APIKey to response ─────────────────────────────────────


def _key_to_info(key: APIKey) -> KeyInfoResponse:
    return KeyInfoResponse(
        key_id=key.id,
        name=key.name,
        prefix=key.prefix,
        scopes=sorted(key.scopes),
        rate_limit_rpm=key.rate_limit_rpm,
        is_active=key.is_active,
        created_at=key.created_at.isoformat(),
        expires_at=key.expires_at.isoformat() if key.expires_at else None,
        revoked_at=key.revoked_at.isoformat() if key.revoked_at else None,
        rotated_from=key.rotated_from,
    )


# ── FastAPI routes ───────────────────────────────────────────────────────────


@router.post("", response_model=CreateKeyResponse, status_code=201)
async def create_key_endpoint(body: CreateKeyRequest) -> CreateKeyResponse:
    """Create a new API key."""
    try:
        raw_key, key_record = create_api_key(
            name=body.name,
            scopes=set(body.scopes),
            rate_limit_rpm=body.rate_limit_rpm,
            expires_in_days=body.expires_in_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CreateKeyResponse(
        raw_key=raw_key,
        key_id=key_record.id,
        name=key_record.name,
        prefix=key_record.prefix,
        scopes=sorted(key_record.scopes),
        rate_limit_rpm=key_record.rate_limit_rpm,
        created_at=key_record.created_at.isoformat(),
        expires_at=key_record.expires_at.isoformat() if key_record.expires_at else None,
    )


@router.get("", response_model=list[KeyInfoResponse])
async def list_keys_endpoint(
    active_only: Annotated[bool, Query(description="Filter to active keys only")] = False,
) -> list[KeyInfoResponse]:
    """List all API keys (hashes are never exposed)."""
    keys = _store.list_all()
    if active_only:
        keys = [k for k in keys if k.is_active and not k.is_expired]
    return [_key_to_info(k) for k in keys]


@router.get("/{key_id}", response_model=KeyInfoResponse)
async def get_key_endpoint(key_id: str) -> KeyInfoResponse:
    """Get details for a specific API key."""
    key = _store.get_by_id(key_id)
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    return _key_to_info(key)


@router.patch("/{key_id}", response_model=KeyInfoResponse)
async def update_key_endpoint(key_id: str, body: UpdateKeyRequest) -> KeyInfoResponse:
    """Update an API key's name, scopes, or rate limit."""
    key = _store.get_by_id(key_id)
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")

    if body.name is not None:
        key.name = body.name
    if body.scopes is not None:
        invalid = set(body.scopes) - ALL_PERMISSIONS
        if invalid:
            raise HTTPException(status_code=400, detail=f"Invalid scopes: {invalid}")
        key.scopes = set(body.scopes)
    if body.rate_limit_rpm is not None:
        key.rate_limit_rpm = body.rate_limit_rpm

    logger.info("api_key_updated", key_id=key_id)
    return _key_to_info(key)


@router.delete("/{key_id}", status_code=204)
async def delete_key_endpoint(key_id: str) -> None:
    """Revoke and delete an API key."""
    key = _store.get_by_id(key_id)
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    revoke_key(key_id)
    _store.remove(key_id)


@router.post("/{key_id}/rotate", response_model=RotateKeyResponse)
async def rotate_key_endpoint(key_id: str, body: RotateKeyRequest) -> RotateKeyResponse:
    """Rotate an API key, issuing a new key with a grace period for the old one."""
    result = rotate_key(key_id, grace_period_hours=body.grace_period_hours)
    if result is None:
        raise HTTPException(status_code=404, detail="API key not found")

    new_raw, new_key = result
    old_key = _store.get_by_id(key_id)

    return RotateKeyResponse(
        new_raw_key=new_raw,
        new_key_id=new_key.id,
        old_key_id=key_id,
        grace_period_end=old_key.grace_period_end.isoformat() if old_key and old_key.grace_period_end else "",
    )


@router.get("/{key_id}/usage", response_model=UsageAnalyticsResponse)
async def get_usage_endpoint(key_id: str) -> UsageAnalyticsResponse:
    """Get usage analytics for a specific API key."""
    analytics = get_usage_analytics(key_id)
    if analytics is None:
        raise HTTPException(status_code=404, detail="API key not found")
    return UsageAnalyticsResponse(**analytics)


@router.post("/cleanup", status_code=200)
async def cleanup_endpoint() -> dict[str, int]:
    """Remove expired and grace-period-ended keys."""
    count = cleanup_expired_keys()
    return {"cleaned_up": count}
