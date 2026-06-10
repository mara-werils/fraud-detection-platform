"""Request context manager with trace ID propagation.

Provides request-scoped storage for metadata that must follow a request
through the entire call chain without being passed explicitly via
function arguments.  Built on Python's ``contextvars`` module so each
async task / thread gets its own isolated copy.

Tracked fields:
  - ``request_id``: unique correlation ID for distributed tracing.
  - ``tenant_id``: multi-tenant isolation key.
  - ``user_id``: authenticated caller identity.
  - ``extra``: arbitrary key-value pairs (feature flags, AB groups, etc.).

Includes a lightweight ASGI middleware that extracts headers, populates
the context, and ensures proper cleanup after the response.
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "RequestContext",
    "RequestContextMiddleware",
    "get_context",
    "get_request_id",
    "get_tenant_id",
    "set_context",
    "clear_context",
    "bind_to_structlog",
]


# ---------------------------------------------------------------------------
# Context storage
# ---------------------------------------------------------------------------

@dataclass
class RequestContext:
    """Immutable snapshot of the current request's metadata."""

    request_id: str
    tenant_id: str | None = None
    user_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


_current_context: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "request_context", default=None
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_context() -> RequestContext | None:
    """Return the current request context, or *None* outside a request."""
    return _current_context.get()


def get_request_id() -> str:
    """Return the current request ID, or generate a new one if unset."""
    ctx = _current_context.get()
    return ctx.request_id if ctx else str(uuid.uuid4())


def get_tenant_id() -> str | None:
    """Return the current tenant ID, or *None* outside a request."""
    ctx = _current_context.get()
    return ctx.tenant_id if ctx else None


def set_context(ctx: RequestContext) -> contextvars.Token[RequestContext | None]:
    """Explicitly set the request context (useful in background workers)."""
    return _current_context.set(ctx)


def clear_context() -> None:
    """Reset the context variable to *None*."""
    _current_context.set(None)


def bind_to_structlog() -> None:
    """Bind current request context fields to structlog's thread-local context."""
    ctx = _current_context.get()
    if ctx is None:
        return
    structlog.contextvars.bind_contextvars(
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id or "",
        user_id=ctx.user_id or "",
    )


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------

# Standard header names (case-insensitive in HTTP but stored lower in ASGI).
_HEADER_REQUEST_ID = b"x-request-id"
_HEADER_TENANT_ID = b"x-tenant-id"
_HEADER_USER_ID = b"x-user-id"


class RequestContextMiddleware:
    """ASGI middleware that populates ``RequestContext`` from headers.

    If no ``X-Request-Id`` header is present a new UUID is generated so
    every request is always traceable.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        request_id = (
            headers.get(_HEADER_REQUEST_ID, b"").decode() or str(uuid.uuid4())
        )
        tenant_id = headers.get(_HEADER_TENANT_ID, b"").decode() or None
        user_id = headers.get(_HEADER_USER_ID, b"").decode() or None

        ctx = RequestContext(
            request_id=request_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        token = set_context(ctx)
        bind_to_structlog()

        async def send_with_request_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers_list = list(message.get("headers", []))
                headers_list.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers_list
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            _current_context.reset(token)
            structlog.contextvars.unbind_contextvars(
                "request_id", "tenant_id", "user_id"
            )
