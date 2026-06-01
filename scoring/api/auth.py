"""API key authentication for the scoring service."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

import structlog
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = structlog.get_logger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory API key store. In production, use a database or secrets manager.
_api_keys: dict[str, APIKeyRecord] = {}


class APIKeyRecord:
    """Represents a registered API key with metadata."""

    __slots__ = ("key_hash", "name", "created_at", "rate_limit", "scopes")

    def __init__(
        self,
        key_hash: str,
        name: str,
        created_at: datetime | None = None,
        rate_limit: int = 1000,
        scopes: list[str] | None = None,
    ) -> None:
        self.key_hash = key_hash
        self.name = name
        self.created_at = created_at or datetime.now(UTC)
        self.rate_limit = rate_limit
        self.scopes = scopes or ["score", "read"]


def _hash_key(key: str) -> str:
    """Hash an API key using SHA-256."""
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key(name: str, rate_limit: int = 1000, scopes: list[str] | None = None) -> str:
    """Generate and register a new API key.

    Args:
        name: Human-readable name for the key.
        rate_limit: Requests per minute allowed.
        scopes: Permission scopes for this key.

    Returns:
        The raw API key string (only shown once).
    """
    raw_key = f"fdp_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)

    _api_keys[key_hash] = APIKeyRecord(
        key_hash=key_hash,
        name=name,
        rate_limit=rate_limit,
        scopes=scopes,
    )

    logger.info("api_key_generated", name=name, rate_limit=rate_limit)
    return raw_key


def register_api_key(raw_key: str, name: str, rate_limit: int = 1000) -> None:
    """Register an existing API key (e.g., from environment variable).

    Args:
        raw_key: The raw API key string.
        name: Human-readable name.
        rate_limit: Requests per minute allowed.
    """
    key_hash = _hash_key(raw_key)
    _api_keys[key_hash] = APIKeyRecord(
        key_hash=key_hash,
        name=name,
        rate_limit=rate_limit,
    )


def validate_api_key(raw_key: str) -> APIKeyRecord | None:
    """Validate an API key and return its record.

    Args:
        raw_key: The raw API key from the request.

    Returns:
        APIKeyRecord if valid, None otherwise.
    """
    key_hash = _hash_key(raw_key)
    return _api_keys.get(key_hash)


async def require_api_key(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
) -> APIKeyRecord:
    """FastAPI dependency that enforces API key authentication.

    Skips auth for health and metrics endpoints.

    Args:
        request: The incoming HTTP request.
        api_key: API key from X-API-Key header.

    Returns:
        Validated APIKeyRecord.

    Raises:
        HTTPException: If the key is missing or invalid.
    """
    # Skip auth for health/metrics/docs endpoints
    skip_paths = {"/health", "/health/ready", "/metrics", "/docs", "/openapi.json", "/redoc"}
    if request.url.path in skip_paths:
        return APIKeyRecord(key_hash="", name="internal", rate_limit=0)

    # Check if auth is disabled (development mode)
    auth_enabled = getattr(request.app.state, "auth_enabled", True)
    if not auth_enabled:
        return APIKeyRecord(key_hash="", name="dev-mode", rate_limit=0)

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    record = validate_api_key(api_key)
    if record is None:
        logger.warning("invalid_api_key_attempt", key_prefix=api_key[:8] if api_key else "empty")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    # Store key info in request state for downstream use
    request.state.api_key_name = record.name
    request.state.api_key_rate_limit = record.rate_limit

    return record
