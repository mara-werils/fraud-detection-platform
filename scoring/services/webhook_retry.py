"""Webhook retry system with exponential backoff, circuit breaker, and dead-letter queue.

Provides a robust retry layer for outbound webhook delivery:
  - Exponential backoff with configurable jitter (full or decorrelated)
  - Per-endpoint circuit breaker to avoid hammering unhealthy receivers
  - Dead-letter queue for permanently failed webhooks
  - HMAC-SHA256 webhook signature generation and verification
  - Delivery status tracking with full attempt history
  - Configurable retry policies per webhook endpoint
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeliveryStatus(StrEnum):
    """Final delivery outcome."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"
    CIRCUIT_OPEN = "circuit_open"


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RetryPolicy(BaseModel):
    """Per-webhook retry configuration.

    Different endpoints may require different retry strategies depending on
    their expected latency, reliability characteristics, and SLAs.
    """

    max_retries: int = 5
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    timeout_seconds: float = 30.0


class WebhookEndpoint(BaseModel):
    """A registered webhook endpoint with its own retry policy."""

    endpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    url: str
    signing_secret: str | None = None
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WebhookEvent(BaseModel):
    """An event to be delivered via webhook."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    endpoint_id: str
    event_type: str
    payload: dict[str, Any]
    headers: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AttemptRecord(BaseModel):
    """Record of a single delivery attempt."""

    attempt_number: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status_code: int | None = None
    response_body: str | None = None
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status_code is not None and 200 <= self.status_code < 300


class DeliveryRecord(BaseModel):
    """Full delivery record including all retry attempts."""

    delivery_id: str = Field(default_factory=lambda: str(uuid4()))
    event_id: str
    endpoint_id: str
    event_type: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: list[AttemptRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreaker:
    """Per-endpoint circuit breaker to prevent overwhelming failed receivers.

    Transitions:
        CLOSED -> OPEN: after ``failure_threshold`` consecutive failures
        OPEN -> HALF_OPEN: after ``recovery_timeout`` seconds
        HALF_OPEN -> CLOSED: on a successful probe delivery
        HALF_OPEN -> OPEN: on another failure during probe
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_state_change: float = field(default_factory=time.monotonic)

    def record_success(self) -> None:
        """Record a successful delivery."""
        self.failure_count = 0
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.last_state_change = time.monotonic()
            logger.info("circuit_breaker_closed")

    def record_failure(self) -> None:
        """Record a failed delivery and possibly open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change = time.monotonic()
            logger.warning("circuit_breaker_reopened", failures=self.failure_count)
        elif (
            self.state == CircuitState.CLOSED
            and self.failure_count >= self.failure_threshold
        ):
            self.state = CircuitState.OPEN
            self.last_state_change = time.monotonic()
            logger.warning(
                "circuit_breaker_opened",
                failures=self.failure_count,
                threshold=self.failure_threshold,
            )

    def allow_request(self) -> bool:
        """Return True if a delivery attempt is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_state_change
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = time.monotonic()
                logger.info("circuit_breaker_half_open")
                return True
            return False

        # HALF_OPEN: allow a single probe request
        return True

    def reset(self) -> None:
        """Reset the breaker to initial closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.monotonic()


# ---------------------------------------------------------------------------
# Signature utilities
# ---------------------------------------------------------------------------


class WebhookSignature:
    """HMAC-SHA256 webhook signature generation and verification."""

    @staticmethod
    def sign(body: bytes, secret: str, timestamp: str | None = None) -> str:
        """Generate an HMAC-SHA256 signature for a webhook payload.

        When *timestamp* is provided the signed content is
        ``{timestamp}.{body}`` to defend against replay attacks.

        Args:
            body: Raw bytes of the request body.
            secret: Shared signing secret.
            timestamp: Optional Unix timestamp string for replay protection.

        Returns:
            Hex-encoded HMAC-SHA256 digest.
        """
        if timestamp:
            signed_content = f"{timestamp}.".encode() + body
        else:
            signed_content = body

        return hmac.new(
            secret.encode("utf-8"),
            signed_content,
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def verify(
        body: bytes,
        secret: str,
        signature_header: str,
        timestamp: str | None = None,
        tolerance_seconds: float = 300.0,
    ) -> bool:
        """Verify an incoming webhook signature.

        Supports optional timestamp tolerance to reject replayed payloads
        that are older than ``tolerance_seconds``.

        Args:
            body: Raw request body bytes.
            secret: Shared signing secret.
            signature_header: Value of the X-Webhook-Signature header
                (expected format: ``sha256=<hex>``).
            timestamp: The X-Webhook-Timestamp header value (Unix epoch).
            tolerance_seconds: Maximum acceptable payload age in seconds.

        Returns:
            True when the signature is valid (and fresh, if timestamp given).
        """
        if not signature_header.startswith("sha256="):
            return False

        received = signature_header.removeprefix("sha256=")

        # Timestamp freshness check
        if timestamp is not None:
            try:
                ts = float(timestamp)
            except (ValueError, TypeError):
                return False
            if abs(time.time() - ts) > tolerance_seconds:
                return False

        expected = WebhookSignature.sign(body, secret, timestamp)
        return hmac.compare_digest(expected, received)


# ---------------------------------------------------------------------------
# Retry service
# ---------------------------------------------------------------------------

# HTTP status codes worth retrying (transient / server-side errors)
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset(
    {408, 425, 429, 500, 502, 503, 504}
)


class WebhookRetryService:
    """Webhook retry service with exponential backoff, circuit breaker, and DLQ.

    Each registered endpoint gets its own circuit breaker and can carry a
    custom :class:`RetryPolicy`.  Permanently-failed events are moved to an
    in-memory dead-letter queue for later inspection or replay.

    Args:
        default_policy: Fallback retry policy when an endpoint has none.
        circuit_failure_threshold: Number of consecutive failures before the
            circuit breaker opens for an endpoint.
        circuit_recovery_timeout: Seconds to wait before probing an open circuit.
        dlq_enabled: Whether to enqueue permanently-failed events.

    Example::

        service = WebhookRetryService()
        ep = service.register_endpoint("https://example.com/hook", secret="s3cret")
        event = WebhookEvent(endpoint_id=ep.endpoint_id, event_type="tx.flagged", payload={...})
        record = await service.deliver(event)
    """

    def __init__(
        self,
        default_policy: RetryPolicy | None = None,
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: float = 60.0,
        dlq_enabled: bool = True,
    ) -> None:
        self._default_policy = default_policy or RetryPolicy()
        self._circuit_failure_threshold = circuit_failure_threshold
        self._circuit_recovery_timeout = circuit_recovery_timeout
        self._dlq_enabled = dlq_enabled

        self._endpoints: dict[str, WebhookEndpoint] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._delivery_history: list[DeliveryRecord] = []
        self._dlq: list[WebhookEvent] = []

        logger.info(
            "webhook_retry_service_initialized",
            default_max_retries=self._default_policy.max_retries,
            circuit_threshold=self._circuit_failure_threshold,
            dlq_enabled=self._dlq_enabled,
        )

    # ------------------------------------------------------------------
    # Endpoint management
    # ------------------------------------------------------------------

    def register_endpoint(
        self,
        url: str,
        signing_secret: str | None = None,
        retry_policy: RetryPolicy | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WebhookEndpoint:
        """Register a new webhook endpoint.

        Args:
            url: Target URL for webhook delivery.
            signing_secret: HMAC-SHA256 signing secret.
            retry_policy: Custom retry policy for this endpoint.
            metadata: Arbitrary metadata to attach to the endpoint.

        Returns:
            The created :class:`WebhookEndpoint`.
        """
        endpoint = WebhookEndpoint(
            url=url,
            signing_secret=signing_secret,
            retry_policy=retry_policy or self._default_policy.model_copy(),
            metadata=metadata or {},
        )
        self._endpoints[endpoint.endpoint_id] = endpoint
        self._circuit_breakers[endpoint.endpoint_id] = CircuitBreaker(
            failure_threshold=self._circuit_failure_threshold,
            recovery_timeout=self._circuit_recovery_timeout,
        )
        logger.info(
            "webhook_endpoint_registered",
            endpoint_id=endpoint.endpoint_id,
            url=url,
        )
        return endpoint

    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Remove a registered endpoint and its circuit breaker.

        Returns:
            True if the endpoint existed and was removed.
        """
        removed = self._endpoints.pop(endpoint_id, None) is not None
        self._circuit_breakers.pop(endpoint_id, None)
        return removed

    def get_endpoint(self, endpoint_id: str) -> WebhookEndpoint | None:
        """Retrieve an endpoint by its ID."""
        return self._endpoints.get(endpoint_id)

    def list_endpoints(self) -> list[WebhookEndpoint]:
        """Return all registered endpoints."""
        return list(self._endpoints.values())

    # ------------------------------------------------------------------
    # Circuit breaker access
    # ------------------------------------------------------------------

    def get_circuit_state(self, endpoint_id: str) -> CircuitState | None:
        """Return the current circuit breaker state for an endpoint."""
        cb = self._circuit_breakers.get(endpoint_id)
        return cb.state if cb else None

    def reset_circuit(self, endpoint_id: str) -> bool:
        """Manually reset the circuit breaker for an endpoint.

        Returns:
            True if the circuit breaker existed and was reset.
        """
        cb = self._circuit_breakers.get(endpoint_id)
        if cb is None:
            return False
        cb.reset()
        return True

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    async def deliver(self, event: WebhookEvent) -> DeliveryRecord:
        """Deliver a webhook event, retrying with exponential backoff.

        The circuit breaker for the target endpoint is consulted before each
        attempt.  If the circuit is open the event is immediately dead-lettered
        (when DLQ is enabled) or returned as ``circuit_open``.

        Args:
            event: The webhook event to deliver.

        Returns:
            A :class:`DeliveryRecord` containing every attempt and the final
            outcome.
        """
        endpoint = self._endpoints.get(event.endpoint_id)
        if endpoint is None:
            raise ValueError(f"Unknown endpoint: {event.endpoint_id}")

        policy = endpoint.retry_policy
        cb = self._circuit_breakers[endpoint.endpoint_id]

        log = logger.bind(
            event_id=event.event_id,
            endpoint_id=endpoint.endpoint_id,
            event_type=event.event_type,
            url=endpoint.url,
        )

        record = DeliveryRecord(
            event_id=event.event_id,
            endpoint_id=endpoint.endpoint_id,
            event_type=event.event_type,
        )

        # Check circuit before starting
        if not cb.allow_request():
            log.warning("webhook_circuit_open_skipping")
            record.status = DeliveryStatus.CIRCUIT_OPEN
            record.completed_at = datetime.now(UTC)
            if self._dlq_enabled:
                self._dlq.append(event)
                record.status = DeliveryStatus.DEAD_LETTERED
            self._delivery_history.append(record)
            return record

        log.info("webhook_retry_delivery_start")

        for attempt_num in range(1, policy.max_retries + 2):
            attempt = await self._attempt_delivery(endpoint, event, attempt_num)
            record.attempts.append(attempt)

            if attempt.succeeded:
                record.status = DeliveryStatus.DELIVERED
                record.completed_at = datetime.now(UTC)
                cb.record_success()
                log.info(
                    "webhook_retry_delivery_success",
                    attempt=attempt_num,
                    status_code=attempt.status_code,
                )
                break

            cb.record_failure()

            is_last_attempt = attempt_num > policy.max_retries
            if is_last_attempt:
                log.warning(
                    "webhook_retry_delivery_exhausted",
                    total_attempts=record.total_attempts,
                )
                break

            # Non-retryable status: stop early
            if attempt.status_code is not None and attempt.status_code not in _RETRYABLE_STATUS_CODES:
                log.warning(
                    "webhook_retry_non_retryable_status",
                    status_code=attempt.status_code,
                )
                break

            # Check circuit again before next attempt
            if not cb.allow_request():
                log.warning("webhook_circuit_opened_during_retries")
                break

            delay = _compute_delay(policy, attempt_num)
            log.info(
                "webhook_retry_scheduled",
                attempt=attempt_num,
                delay_seconds=round(delay, 3),
            )
            await asyncio.sleep(delay)

        # Finalise record
        if record.status == DeliveryStatus.PENDING:
            record.status = DeliveryStatus.FAILED
            record.completed_at = datetime.now(UTC)
            if self._dlq_enabled:
                self._dlq.append(event)
                record.status = DeliveryStatus.DEAD_LETTERED

        self._delivery_history.append(record)
        return record

    async def deliver_batch(
        self, events: list[WebhookEvent]
    ) -> list[DeliveryRecord]:
        """Deliver multiple events concurrently.

        Args:
            events: List of webhook events to deliver.

        Returns:
            Delivery records in the same order as input.
        """
        if not events:
            return []
        return list(
            await asyncio.gather(*[self.deliver(e) for e in events])
        )

    # ------------------------------------------------------------------
    # Dead-letter queue
    # ------------------------------------------------------------------

    def get_dead_letter_queue(self) -> list[WebhookEvent]:
        """Return a snapshot of the dead-letter queue."""
        return list(self._dlq)

    def dead_letter_queue_size(self) -> int:
        """Return the current DLQ size."""
        return len(self._dlq)

    async def retry_dead_letters(self) -> list[DeliveryRecord]:
        """Drain and re-attempt all events in the dead-letter queue.

        Events that fail again are re-enqueued.

        Returns:
            Delivery records for each retried event.
        """
        if not self._dlq:
            return []
        to_retry = list(self._dlq)
        self._dlq.clear()
        logger.info("webhook_dlq_retry_start", count=len(to_retry))
        results = await self.deliver_batch(to_retry)
        recovered = sum(1 for r in results if r.status == DeliveryStatus.DELIVERED)
        logger.info(
            "webhook_dlq_retry_complete",
            retried=len(to_retry),
            recovered=recovered,
        )
        return results

    def purge_dead_letter_queue(self) -> int:
        """Remove all events from the DLQ.

        Returns:
            Number of events purged.
        """
        count = len(self._dlq)
        self._dlq.clear()
        return count

    # ------------------------------------------------------------------
    # Delivery history
    # ------------------------------------------------------------------

    def get_delivery_history(
        self,
        endpoint_id: str | None = None,
        limit: int = 100,
    ) -> list[DeliveryRecord]:
        """Return recent delivery records.

        Args:
            endpoint_id: Filter by endpoint (all if ``None``).
            limit: Maximum number of records to return.

        Returns:
            Most recent records, newest last.
        """
        records = self._delivery_history
        if endpoint_id:
            records = [r for r in records if r.endpoint_id == endpoint_id]
        return records[-limit:]

    def get_delivery_stats(self) -> dict[str, Any]:
        """Return aggregate delivery statistics."""
        total = len(self._delivery_history)
        delivered = sum(
            1 for r in self._delivery_history if r.status == DeliveryStatus.DELIVERED
        )
        failed = sum(
            1 for r in self._delivery_history if r.status == DeliveryStatus.FAILED
        )
        dead_lettered = sum(
            1
            for r in self._delivery_history
            if r.status == DeliveryStatus.DEAD_LETTERED
        )
        circuit_open = sum(
            1
            for r in self._delivery_history
            if r.status == DeliveryStatus.CIRCUIT_OPEN
        )
        return {
            "total": total,
            "delivered": delivered,
            "failed": failed,
            "dead_lettered": dead_lettered,
            "circuit_open": circuit_open,
            "dlq_size": len(self._dlq),
            "endpoints": len(self._endpoints),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _attempt_delivery(
        self,
        endpoint: WebhookEndpoint,
        event: WebhookEvent,
        attempt_num: int,
    ) -> AttemptRecord:
        """Execute one HTTP POST and return an :class:`AttemptRecord`."""
        timestamp = str(int(time.time()))
        body = json.dumps(
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "timestamp": event.created_at.isoformat(),
                "data": event.payload,
            },
            default=str,
        ).encode()

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Webhook-ID": event.event_id,
            "X-Webhook-Event": event.event_type,
            "X-Webhook-Attempt": str(attempt_num),
            "X-Webhook-Timestamp": timestamp,
            **event.headers,
        }

        if endpoint.signing_secret:
            sig = WebhookSignature.sign(body, endpoint.signing_secret, timestamp)
            headers["X-Webhook-Signature"] = f"sha256={sig}"

        start = time.monotonic()
        attempt = AttemptRecord(attempt_number=attempt_num)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(endpoint.retry_policy.timeout_seconds)
            ) as client:
                response = await client.post(
                    endpoint.url, content=body, headers=headers
                )
            latency_ms = (time.monotonic() - start) * 1000
            attempt.status_code = response.status_code
            attempt.latency_ms = round(latency_ms, 2)
            try:
                raw = response.text
                attempt.response_body = raw[:512] if len(raw) > 512 else raw
            except Exception:
                attempt.response_body = None

        except httpx.TimeoutException as exc:
            attempt.error = f"timeout: {exc}"
            attempt.latency_ms = round((time.monotonic() - start) * 1000, 2)
        except httpx.RequestError as exc:
            attempt.error = f"request_error: {exc}"
            attempt.latency_ms = round((time.monotonic() - start) * 1000, 2)
        except Exception as exc:
            attempt.error = f"unexpected: {exc}"
            attempt.latency_ms = round((time.monotonic() - start) * 1000, 2)

        return attempt


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _compute_delay(policy: RetryPolicy, attempt: int) -> float:
    """Compute backoff delay with optional jitter.

    Uses full-jitter strategy: ``uniform(0, min(max_delay, base * mult^(n-1)))``.

    Args:
        policy: The retry policy to use.
        attempt: The 1-based attempt number that just failed.

    Returns:
        Seconds to sleep before the next attempt.
    """
    base = policy.initial_delay_seconds * (
        policy.backoff_multiplier ** (attempt - 1)
    )
    clamped = min(base, policy.max_delay_seconds)
    if policy.jitter:
        return random.uniform(0, clamped)
    return clamped


__all__ = ["AttemptRecord", "CircuitBreaker", "CircuitState", "DeliveryRecord", "DeliveryStatus", "RetryPolicy", "WebhookEndpoint", "WebhookEvent", "WebhookRetryService", "WebhookSignature"]
