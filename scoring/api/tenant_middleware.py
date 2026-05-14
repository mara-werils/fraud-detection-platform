"""Tenant isolation middleware — injects org context into every request."""

from __future__ import annotations

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger(__name__)

SKIP_PATHS = {"/health", "/health/ready", "/metrics", "/docs", "/openapi.json", "/redoc"}


class TenantMiddleware(BaseHTTPMiddleware):
    """Extracts org_id from JWT/API key context and attaches it to request.state.

    Downstream handlers use request.state.org_id for data isolation.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in SKIP_PATHS:
            request.state.org_id = None
            return await call_next(request)

        # org_id is set by auth middleware after token/key validation
        org_id = getattr(request.state, "org_id", None)

        if org_id:
            # Bind org_id to structured log context for the duration of the request
            structlog.contextvars.bind_contextvars(org_id=str(org_id))

        response = await call_next(request)

        if org_id:
            structlog.contextvars.unbind_contextvars("org_id")

        response.headers["X-Org-ID"] = str(org_id) if org_id else "anonymous"
        return response
