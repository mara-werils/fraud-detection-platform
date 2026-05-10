"""FastAPI middleware for request tracking, latency, and CORS."""

from __future__ import annotations

import time
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger(__name__)

REQUEST_LATENCY = Histogram(
    "scoring_http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "path", "status_code"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that injects an X-Request-ID header into every request/response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LatencyMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks request latency via logs and Prometheus histogram."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        # Normalize path to avoid high-cardinality label explosion
        path = request.url.path
        REQUEST_LATENCY.labels(
            method=request.method,
            path=path,
            status_code=str(response.status_code),
        ).observe(elapsed)

        await logger.ainfo(
            "http_request",
            method=request.method,
            path=path,
            status_code=response.status_code,
            latency_ms=round(elapsed * 1000, 2),
            request_id=getattr(request.state, "request_id", None),
        )

        return response


def setup_middleware(app: FastAPI) -> None:
    """Attach all middleware to the FastAPI application.

    Order matters: outermost middleware executes first. We add RequestID
    first so it is available to LatencyMiddleware log entries.

    Args:
        app: The FastAPI application instance.
    """
    app.add_middleware(LatencyMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
