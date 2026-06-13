"""API versioning and deprecation middleware for the fraud detection platform.

Provides:
  - APIVersion enum (V1, V2)
  - DeprecationPolicy model with sunset / migration metadata
  - VersioningMiddleware — extracts version from URL path or Accept header,
    injects deprecation response headers, and tracks Prometheus metrics
  - APIVersionRouter — wraps FastAPI APIRouter with per-version registration,
    version_gate decorator, and deprecated decorator
  - Response transformers — v1 compatibility shim and v2 pagination envelope
  - VersionNegotiator — central authority for version negotiation and stats
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime
from enum import Enum
from functools import wraps
from typing import Any

import structlog
from fastapi import APIRouter, Request, Response
from prometheus_client import Counter
from pydantic import BaseModel, ConfigDict, HttpUrl
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

API_REQUESTS_BY_VERSION = Counter(
    "api_requests_by_version_total",
    "Total API requests broken down by version",
    ["version", "method", "path"],
)

DEPRECATED_API_USAGE = Counter(
    "deprecated_api_usage_total",
    "Total requests made to deprecated API versions",
    ["version", "method", "path"],
)


# ── Version enum ──────────────────────────────────────────────────────────────


class APIVersion(str, Enum):
    """Supported API versions.

    String values match the path segment used in URLs (e.g. ``/api/v1/``).
    """

    V1 = "v1"
    V2 = "v2"

    @classmethod
    def from_path(cls, path: str) -> APIVersion | None:
        """Extract the version from a URL path like ``/api/v1/transactions``."""
        match = re.search(r"/api/(v\d+)/", path)
        if match:
            raw = match.group(1)
            try:
                return cls(raw)
            except ValueError:
                return None
        return None

    @classmethod
    def from_accept_header(cls, accept: str) -> APIVersion | None:
        """Parse version from an Accept header such as ``application/vnd.fraud.v2+json``."""
        match = re.search(r"application/vnd\.fraud\.(v\d+)\+json", accept)
        if match:
            raw = match.group(1)
            try:
                return cls(raw)
            except ValueError:
                return None
        return None


# ── Deprecation policy ────────────────────────────────────────────────────────


class DeprecationPolicy(BaseModel):
    """Deprecation metadata for an API version.

    Attributes:
        version: The deprecated :class:`APIVersion`.
        deprecated_at: The date on which the version was officially deprecated.
        sunset_at: The date after which the version will be removed.
        migration_guide_url: URL pointing to the migration documentation.
    """

    model_config = ConfigDict(frozen=True)

    version: APIVersion
    deprecated_at: date
    sunset_at: date
    migration_guide_url: HttpUrl

    @property
    def sunset_datetime(self) -> datetime:
        """Return sunset date as a UTC-aware datetime (used in HTTP headers)."""
        return datetime.combine(self.sunset_at, datetime.min.time(), tzinfo=UTC)

    @property
    def deprecation_header(self) -> str:
        """RFC 8594 ``Deprecation`` header value (HTTP-date)."""
        dt = datetime.combine(self.deprecated_at, datetime.min.time(), tzinfo=UTC)
        return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")

    @property
    def sunset_header(self) -> str:
        """RFC 8594 ``Sunset`` header value (HTTP-date)."""
        return self.sunset_datetime.strftime("%a, %d %b %Y %H:%M:%S GMT")

    @property
    def link_header(self) -> str:
        """``Link`` header pointing to the migration guide."""
        return f'<{self.migration_guide_url}>; rel="deprecation"'


# ── Version negotiator ────────────────────────────────────────────────────────


class VersionNegotiator:
    """Central authority for version negotiation, deprecation checks, and stats.

    The negotiator is configured once at startup and shared across the
    :class:`VersioningMiddleware` and :class:`APIVersionRouter`.

    Args:
        default_version: Fallback version when no version can be inferred.
        deprecation_policies: Mapping of :class:`APIVersion` to its
            :class:`DeprecationPolicy`, if any.
    """

    def __init__(
        self,
        default_version: APIVersion = APIVersion.V1,
        deprecation_policies: dict[APIVersion, DeprecationPolicy] | None = None,
    ) -> None:
        self._default = default_version
        self._policies: dict[APIVersion, DeprecationPolicy] = deprecation_policies or {}
        # usage_stats[version][method][path] = count (mirrors Prometheus but queryable in-process)
        self._usage_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

    # -- Public API ------------------------------------------------------------

    def negotiate(self, request: Request) -> APIVersion:
        """Determine the API version for *request*.

        Resolution order:
        1. URL path segment (``/api/v1/``, ``/api/v2/``).
        2. ``Accept`` header (``application/vnd.fraud.v2+json``).
        3. Default version.

        Args:
            request: The incoming FastAPI/Starlette request.

        Returns:
            Resolved :class:`APIVersion`.
        """
        version = APIVersion.from_path(request.url.path)
        if version is None:
            accept = request.headers.get("Accept", "")
            version = APIVersion.from_accept_header(accept)
        if version is None:
            version = self._default
        return version

    def is_deprecated(self, version: APIVersion) -> bool:
        """Return ``True`` if *version* has a registered deprecation policy."""
        return version in self._policies

    def get_deprecation_info(self, version: APIVersion) -> DeprecationPolicy | None:
        """Return the :class:`DeprecationPolicy` for *version*, or ``None``."""
        return self._policies.get(version)

    def get_supported_versions(self) -> list[APIVersion]:
        """Return all members of :class:`APIVersion` as the supported list."""
        return list(APIVersion)

    def get_version_usage_stats(self) -> dict[str, Any]:
        """Return an in-process snapshot of version usage statistics.

        Returns:
            Nested mapping ``{version: {method: {path: count}}}``.
        """
        # Convert defaultdicts to plain dicts for serialisation
        return {
            version: {method: dict(paths) for method, paths in methods.items()}
            for version, methods in self._usage_stats.items()
        }

    # -- Internal helpers ------------------------------------------------------

    def record_request(self, version: APIVersion, method: str, path: str) -> None:
        """Increment internal usage counters (called by middleware)."""
        self._usage_stats[version.value][method][path] += 1


# ── Versioning middleware ─────────────────────────────────────────────────────


class VersioningMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that handles API version extraction and deprecation.

    For each request the middleware:

    1. Resolves the API version via :class:`VersionNegotiator`.
    2. Stores the version on ``request.state.api_version``.
    3. Increments Prometheus counters.
    4. For deprecated versions, adds ``Deprecation``, ``Sunset``, and ``Link``
       response headers per RFC 8594.

    Args:
        app: The ASGI application to wrap.
        negotiator: Shared :class:`VersionNegotiator` instance.
    """

    def __init__(self, app: Any, negotiator: VersionNegotiator) -> None:
        super().__init__(app)
        self._negotiator = negotiator

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        version = self._negotiator.negotiate(request)
        request.state.api_version = version

        method = request.method
        path = request.url.path

        # Prometheus + in-process tracking
        API_REQUESTS_BY_VERSION.labels(
            version=version.value,
            method=method,
            path=path,
        ).inc()
        self._negotiator.record_request(version, method, path)

        is_deprecated = self._negotiator.is_deprecated(version)
        if is_deprecated:
            DEPRECATED_API_USAGE.labels(
                version=version.value,
                method=method,
                path=path,
            ).inc()
            await logger.awarning(
                "deprecated_api_version_used",
                version=version.value,
                method=method,
                path=path,
            )

        response = await call_next(request)

        # Attach deprecation headers AFTER the handler runs so they are always present
        if is_deprecated:
            policy = self._negotiator.get_deprecation_info(version)
            if policy is not None:
                response.headers["Deprecation"] = policy.deprecation_header
                response.headers["Sunset"] = policy.sunset_header
                response.headers["Link"] = policy.link_header

        response.headers["X-API-Version"] = version.value
        return response


# ── Response transformers ─────────────────────────────────────────────────────


def transform_v2_to_v1(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip v2-only fields and flatten the envelope for v1 clients.

    v2 responses wrap data under a ``data`` key.  v1 clients expect the
    payload at the top level.  Additionally, v2 ``snake_case`` aliases for
    renamed fields are mapped back to their v1 names.

    Args:
        payload: A raw v2 response dictionary.

    Returns:
        A v1-compatible dictionary.
    """
    # Unwrap data envelope if present
    result = payload.get("data", payload)

    # v2 -> v1 field renames
    field_map: dict[str, str] = {
        "risk_score": "score",
        "decision_label": "decision",
        "flagged_signals": "flags",
    }
    for v2_key, v1_key in field_map.items():
        if v2_key in result:
            result[v1_key] = result.pop(v2_key)

    # Drop v2-only metadata keys that v1 clients cannot consume
    for key in ("_version", "_pagination", "model_metadata"):
        result.pop(key, None)

    return result


def add_pagination_envelope(
    items: list[Any],
    *,
    page: int = 1,
    page_size: int = 50,
    total: int | None = None,
) -> dict[str, Any]:
    """Wrap a list of items in a v2 pagination envelope.

    Args:
        items: The list of result objects.
        page: Current (1-based) page number.
        page_size: Number of items per page.
        total: Total number of items across all pages (``None`` if unknown).

    Returns:
        A v2 response envelope with ``data``, ``_pagination``, and
        ``_version`` keys.
    """
    pagination: dict[str, Any] = {
        "page": page,
        "page_size": page_size,
        "count": len(items),
    }
    if total is not None:
        pagination["total"] = total
        pagination["total_pages"] = -(-total // page_size)  # ceiling division

    return {
        "data": items,
        "_pagination": pagination,
        "_version": APIVersion.V2.value,
    }


def version_transform(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the appropriate response transformation based on the negotiated version.

    Args:
        request: The current request (reads ``request.state.api_version``).
        payload: The raw response payload produced by the handler.

    Returns:
        A version-appropriate response dict.
    """
    version: APIVersion = getattr(request.state, "api_version", APIVersion.V1)
    if version == APIVersion.V1:
        return transform_v2_to_v1(payload)
    # V2: return as-is (handlers should already wrap with add_pagination_envelope)
    return payload


# ── API version router ────────────────────────────────────────────────────────


class APIVersionRouter:
    """Convenience wrapper around :class:`fastapi.APIRouter` with versioning helpers.

    Usage::

        vr = APIVersionRouter(negotiator=negotiator)
        vr.register_v1("/transactions", list_transactions_v1)
        vr.register_v2("/transactions", list_transactions_v2)

        @vr.version_gate(min_version=APIVersion.V2)
        async def v2_only_endpoint(request: Request): ...

        @vr.deprecated(
            sunset_date=date(2026, 12, 31),
            migration_url="https://docs.example.com/migrate-v1-v2",
        )
        async def legacy_endpoint(request: Request): ...

    Args:
        negotiator: Shared :class:`VersionNegotiator`.
        prefix: Optional URL prefix forwarded to the underlying router.
        tags: OpenAPI tags forwarded to the underlying router.
    """

    def __init__(
        self,
        negotiator: VersionNegotiator,
        prefix: str = "",
        tags: list[str] | None = None,
    ) -> None:
        self._negotiator = negotiator
        self.v1_router = APIRouter(prefix=f"{prefix}/api/v1", tags=tags or [])
        self.v2_router = APIRouter(prefix=f"{prefix}/api/v2", tags=tags or [])

    # -- Registration helpers --------------------------------------------------

    def register_v1(self, path: str, endpoint: Callable[..., Any], **kwargs: Any) -> None:
        """Register *endpoint* under the v1 router at *path*.

        Args:
            path: URL path relative to the v1 prefix (e.g. ``/transactions``).
            endpoint: An async FastAPI path function.
            **kwargs: Forwarded to :meth:`APIRouter.add_api_route`.
        """
        self.v1_router.add_api_route(path, endpoint, **kwargs)

    def register_v2(self, path: str, endpoint: Callable[..., Any], **kwargs: Any) -> None:
        """Register *endpoint* under the v2 router at *path*.

        Args:
            path: URL path relative to the v2 prefix.
            endpoint: An async FastAPI path function.
            **kwargs: Forwarded to :meth:`APIRouter.add_api_route`.
        """
        self.v2_router.add_api_route(path, endpoint, **kwargs)

    # -- Decorators ------------------------------------------------------------

    def version_gate(
        self,
        min_version: APIVersion | None = None,
        max_version: APIVersion | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that restricts an endpoint to a version range.

        Requests outside the allowed range receive a ``409 Conflict`` response
        detailing which version is required.

        Args:
            min_version: Minimum required :class:`APIVersion` (inclusive).
            max_version: Maximum allowed :class:`APIVersion` (inclusive).

        Returns:
            A decorator that wraps the endpoint with version-checking logic.
        """
        versions = list(APIVersion)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            async def wrapper(request: Request, *args: Any, **kwargs: Any) -> Any:
                current: APIVersion = getattr(
                    request.state, "api_version", self._negotiator.negotiate(request)
                )
                current_idx = versions.index(current)
                if min_version is not None and current_idx < versions.index(min_version):
                    return JSONResponse(
                        status_code=409,
                        content={
                            "error": "api_version_too_old",
                            "detail": f"This endpoint requires at least API {min_version.value}.",
                            "required_version": min_version.value,
                            "current_version": current.value,
                        },
                    )
                if max_version is not None and current_idx > versions.index(max_version):
                    return JSONResponse(
                        status_code=409,
                        content={
                            "error": "api_version_too_new",
                            "detail": f"This endpoint is only available up to API {max_version.value}.",
                            "max_version": max_version.value,
                            "current_version": current.value,
                        },
                    )
                return await func(request, *args, **kwargs)

            return wrapper

        return decorator

    def deprecated(
        self,
        sunset_date: date,
        migration_url: str,
        deprecated_since: date | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that marks a single endpoint as deprecated.

        Unlike version-level deprecation (managed by :class:`VersionNegotiator`),
        this marks individual endpoints that are being phased out regardless of
        the overall version.

        Args:
            sunset_date: The date after which the endpoint will be removed.
            migration_url: URL of the migration guide for this endpoint.
            deprecated_since: When the endpoint was deprecated (defaults to today).

        Returns:
            A decorator that injects deprecation headers into the response.
        """
        since = deprecated_since or date.today()

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await func(*args, **kwargs)

                since_dt = datetime.combine(since, datetime.min.time(), tzinfo=UTC)
                sunset_dt = datetime.combine(sunset_date, datetime.min.time(), tzinfo=UTC)
                fmt = "%a, %d %b %Y %H:%M:%S GMT"

                if isinstance(result, Response):
                    result.headers["Deprecation"] = since_dt.strftime(fmt)
                    result.headers["Sunset"] = sunset_dt.strftime(fmt)
                    result.headers["Link"] = f'<{migration_url}>; rel="deprecation"'

                DEPRECATED_API_USAGE.labels(
                    version="endpoint",
                    method="ANY",
                    path=func.__name__,
                ).inc()

                return result

            return wrapper

        return decorator


__all__ = ["APIVersion", "APIVersionRouter", "DeprecationPolicy", "VersionNegotiator", "VersioningMiddleware", "add_pagination_envelope", "transform_v2_to_v1", "version_transform"]
