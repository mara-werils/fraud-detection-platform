"""Graceful shutdown manager.

Registers SIGTERM/SIGINT handlers that:
1. Stop accepting new connections (Kubernetes sends SIGTERM before pod deletion)
2. Wait for in-flight requests to complete (drain period)
3. Flush usage meter and audit log buffers
4. Close all connections cleanly

Kubernetes pod terminationGracePeriodSeconds should be set to drain_seconds + 10.
"""

from __future__ import annotations

import asyncio
import signal
import time

import structlog

logger = structlog.get_logger(__name__)

_DRAINING = False
_DRAIN_START: float = 0.0


def is_draining() -> bool:
    return _DRAINING


class GracefulShutdownManager:
    """Coordinates graceful shutdown across all services."""

    def __init__(self, drain_seconds: float = 30.0) -> None:
        self._drain_seconds = drain_seconds
        self._shutdown_event = asyncio.Event()
        self._services: list[tuple[str, object]] = []

    def register(self, name: str, service) -> None:
        """Register a service with a .stop() coroutine for shutdown."""
        self._services.append((name, service))

    def setup_signals(self) -> None:
        """Install SIGTERM and SIGINT handlers."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._handle_signal()))

    async def _handle_signal(self) -> None:
        global _DRAINING, _DRAIN_START
        if _DRAINING:
            return

        _DRAINING = True
        _DRAIN_START = time.monotonic()
        await logger.awarning(
            "graceful_shutdown_started",
            drain_seconds=self._drain_seconds,
        )

        # Allow in-flight requests to complete
        await asyncio.sleep(self._drain_seconds)

        await self.shutdown()
        self._shutdown_event.set()

    async def shutdown(self) -> None:
        """Stop all registered services in reverse registration order."""
        for name, service in reversed(self._services):
            try:
                stop = getattr(service, "stop", None)
                if stop and asyncio.iscoroutinefunction(stop):
                    await asyncio.wait_for(stop(), timeout=10.0)
                    await logger.ainfo("service_stopped", name=name)
            except asyncio.TimeoutError:
                await logger.awarning("service_stop_timeout", name=name)
            except Exception as exc:
                await logger.awarning("service_stop_error", name=name, error=str(exc))

    async def wait_for_shutdown(self) -> None:
        await self._shutdown_event.wait()


class DrainMiddleware:
    """Middleware that returns 503 during the drain period.

    This tells load balancers to stop routing new requests here
    while existing requests finish.
    """

    DRAIN_PATHS = frozenset({"/health", "/health/live", "/health/ready"})

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and _DRAINING:
            path = scope.get("path", "")
            if path not in self.DRAIN_PATHS:
                # Return 503 immediately
                await send({
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [(b"content-type", b"application/json")],
                })
                await send({
                    "type": "http.response.body",
                    "body": b'{"detail":"Service draining, please retry"}',
                })
                return
        await self._app(scope, receive, send)

    def __init__(self, app):
        self._app = app
