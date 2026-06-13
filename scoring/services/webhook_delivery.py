"""Reliable webhook delivery service with exponential backoff and dead-letter queue.

Provides production-grade webhook delivery guarantees:
  - Exponential backoff with optional jitter across configurable retry attempts
  - HMAC-SHA256 payload signing for consumer verification
  - Dead-letter queue (DLQ) for payloads that exhaust all retries
  - Prometheus metrics for delivery health, latency, and DLQ size
  - Concurrent batch delivery via asyncio.gather
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import random
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
import structlog
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

webhook_deliveries_total = Counter(
    "webhook_deliveries_total",
    "Total webhook delivery outcomes",
    ["status", "event_type"],
)

webhook_delivery_latency_seconds = Histogram(
    "webhook_delivery_latency_seconds",
    "End-to-end latency for a single webhook delivery (all attempts)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

webhook_retry_count = Histogram(
    "webhook_retry_count",
    "Number of retry attempts made before a final outcome",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)

webhook_dead_letter_queue_size = Gauge(
    "webhook_dead_letter_queue_size",
    "Current number of payloads sitting in the dead-letter queue",
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class WebhookDeliveryConfig(BaseModel):
    """Runtime configuration for the delivery service."""

    max_retries: int = 5
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    timeout_seconds: float = 30.0
    dead_letter_queue_enabled: bool = True


class WebhookPayload(BaseModel):
    """A single webhook event ready for delivery."""

    webhook_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    payload: dict[str, Any]
    target_url: str
    headers: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DeliveryAttempt(BaseModel):
    """Record of one HTTP delivery attempt."""

    attempt_number: int
    status_code: int | None = None
    response_body: str | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status_code is not None and 200 <= self.status_code < 300


class WebhookDeliveryResult(BaseModel):
    """Aggregated outcome after all delivery attempts for one payload."""

    webhook_id: str
    delivered: bool
    attempts: list[DeliveryAttempt] = Field(default_factory=list)
    total_attempts: int = 0
    final_status: str = "pending"  # "delivered" | "failed" | "dead_lettered"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class WebhookDeliveryService:
    """Reliable webhook delivery with exponential backoff and dead-letter queue.

    Args:
        config: Delivery tuning parameters (retries, timeouts, jitter, …).
        signing_secret: Shared secret used to produce HMAC-SHA256 signatures.
            When provided, every request carries an ``X-Webhook-Signature``
            header that receivers can verify independently of TLS.

    Example::

        service = WebhookDeliveryService(
            config=WebhookDeliveryConfig(max_retries=3),
            signing_secret="whsec_abc123",
        )
        result = await service.deliver(payload)
    """

    # HTTP status codes that are worth retrying (server-side / transient errors).
    _RETRYABLE_STATUS_CODES: frozenset[int] = frozenset(
        {408, 425, 429, 500, 502, 503, 504}
    )

    def __init__(
        self,
        config: WebhookDeliveryConfig | None = None,
        signing_secret: str | None = None,
    ) -> None:
        self._config = config or WebhookDeliveryConfig()
        self._signing_secret = signing_secret

        # In-memory dead-letter queue (replace with Redis/DB in multi-process deployments)
        self._dlq: list[WebhookPayload] = []

        # Delivery stats counters (in addition to Prometheus)
        self._stats: dict[str, int] = {
            "total_delivered": 0,
            "total_failed": 0,
            "total_dead_lettered": 0,
            "total_attempts": 0,
        }

        logger.info(
            "webhook_delivery_service_initialized",
            max_retries=self._config.max_retries,
            initial_delay=self._config.initial_delay_seconds,
            jitter=self._config.jitter,
            dlq_enabled=self._config.dead_letter_queue_enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def deliver(self, payload: WebhookPayload) -> WebhookDeliveryResult:
        """Deliver one webhook payload, retrying with exponential backoff.

        Args:
            payload: The webhook event to deliver.

        Returns:
            A :class:`WebhookDeliveryResult` describing every attempt made
            and the final outcome.
        """
        log = logger.bind(
            webhook_id=payload.webhook_id,
            event_type=payload.event_type,
            target_url=payload.target_url,
        )
        log.info("webhook_delivery_start")

        start_wall = time.monotonic()
        result = WebhookDeliveryResult(webhook_id=payload.webhook_id, delivered=False)

        for attempt_num in range(1, self._config.max_retries + 2):
            attempt = await self._attempt_delivery(payload, attempt_num)
            result.attempts.append(attempt)
            result.total_attempts += 1
            self._stats["total_attempts"] += 1

            if attempt.succeeded:
                result.delivered = True
                result.final_status = "delivered"
                self._stats["total_delivered"] += 1
                webhook_deliveries_total.labels(
                    status="delivered", event_type=payload.event_type
                ).inc()
                log.info(
                    "webhook_delivery_success",
                    attempt=attempt_num,
                    status_code=attempt.status_code,
                    latency_ms=attempt.latency_ms,
                )
                break

            is_last_attempt = attempt_num > self._config.max_retries
            if is_last_attempt:
                log.warning(
                    "webhook_delivery_exhausted",
                    total_attempts=result.total_attempts,
                    last_status_code=attempt.status_code,
                    last_error=attempt.error,
                )
                break

            # Decide whether to retry at all for this status code
            if attempt.status_code is not None and not self._should_retry(
                attempt.status_code
            ):
                log.warning(
                    "webhook_delivery_non_retryable",
                    status_code=attempt.status_code,
                    attempt=attempt_num,
                )
                break

            delay = self._compute_delay(attempt_num)
            log.info(
                "webhook_delivery_retry_scheduled",
                attempt=attempt_num,
                delay_seconds=round(delay, 3),
                status_code=attempt.status_code,
                error=attempt.error,
            )
            await asyncio.sleep(delay)

        # Record Prometheus retry histogram
        webhook_retry_count.observe(result.total_attempts - 1)

        # Record total wall-clock latency
        total_elapsed = time.monotonic() - start_wall
        webhook_delivery_latency_seconds.observe(total_elapsed)

        if not result.delivered:
            result.final_status = "failed"
            self._stats["total_failed"] += 1
            webhook_deliveries_total.labels(
                status="failed", event_type=payload.event_type
            ).inc()

            if self._config.dead_letter_queue_enabled:
                self._enqueue_dead_letter(payload)
                result.final_status = "dead_lettered"

        return result

    async def deliver_batch(
        self, payloads: list[WebhookPayload]
    ) -> list[WebhookDeliveryResult]:
        """Deliver multiple webhook payloads concurrently.

        Each payload is delivered independently; a failure in one does not
        affect the others.

        Args:
            payloads: Webhook events to deliver.

        Returns:
            Results in the same order as the input list.
        """
        if not payloads:
            return []

        logger.info("webhook_batch_delivery_start", count=len(payloads))
        results: list[WebhookDeliveryResult] = await asyncio.gather(
            *[self.deliver(p) for p in payloads], return_exceptions=False
        )
        delivered = sum(1 for r in results if r.delivered)
        logger.info(
            "webhook_batch_delivery_complete",
            total=len(payloads),
            delivered=delivered,
            failed=len(payloads) - delivered,
        )
        return list(results)

    # ------------------------------------------------------------------
    # Dead-letter queue
    # ------------------------------------------------------------------

    def get_dead_letter_queue(self) -> list[WebhookPayload]:
        """Return all payloads currently in the dead-letter queue (snapshot)."""
        return list(self._dlq)

    async def retry_dead_letters(self) -> list[WebhookDeliveryResult]:
        """Drain and re-attempt every payload in the dead-letter queue.

        Successfully re-delivered payloads are removed from the DLQ.
        Payloads that fail again are re-added.

        Returns:
            Delivery results for every payload that was retried.
        """
        if not self._dlq:
            logger.info("webhook_dlq_empty_nothing_to_retry")
            return []

        to_retry = list(self._dlq)
        self._dlq.clear()
        webhook_dead_letter_queue_size.set(0)

        logger.info("webhook_dlq_retry_start", count=len(to_retry))
        results = await self.deliver_batch(to_retry)

        still_failed = [r for r in results if not r.delivered]
        logger.info(
            "webhook_dlq_retry_complete",
            retried=len(to_retry),
            recovered=len(to_retry) - len(still_failed),
            still_failed=len(still_failed),
        )
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_delivery_stats(self) -> dict[str, Any]:
        """Return a snapshot of delivery statistics.

        Returns:
            Dict containing aggregate counters and DLQ size.
        """
        dlq_size = len(self._dlq)
        return {
            **self._stats,
            "dead_letter_queue_size": dlq_size,
            "config": self._config.model_dump(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _attempt_delivery(
        self, payload: WebhookPayload, attempt_num: int
    ) -> DeliveryAttempt:
        """Execute one HTTP POST and return a :class:`DeliveryAttempt`.

        Args:
            payload: The webhook payload to send.
            attempt_num: 1-based attempt counter (used for logging).

        Returns:
            A populated :class:`DeliveryAttempt` (never raises).
        """
        body = json.dumps(
            {
                "webhook_id": payload.webhook_id,
                "event_type": payload.event_type,
                "timestamp": payload.created_at.isoformat(),
                "data": payload.payload,
            },
            default=str,
        ).encode()

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Webhook-ID": payload.webhook_id,
            "X-Webhook-Event": payload.event_type,
            "X-Webhook-Attempt": str(attempt_num),
            **payload.headers,
        }

        if self._signing_secret:
            signature = self._sign_payload(body, self._signing_secret)
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        start = time.monotonic()
        attempt = DeliveryAttempt(attempt_number=attempt_num)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout_seconds)
            ) as client:
                response = await client.post(
                    payload.target_url, content=body, headers=headers
                )

            latency_ms = (time.monotonic() - start) * 1000
            attempt.status_code = response.status_code
            attempt.latency_ms = round(latency_ms, 2)

            # Capture a truncated response body for debugging
            try:
                raw = response.text
                attempt.response_body = raw[:512] if len(raw) > 512 else raw
            except Exception:
                attempt.response_body = None

        except httpx.TimeoutException as exc:
            attempt.error = f"timeout: {exc}"
            attempt.latency_ms = round((time.monotonic() - start) * 1000, 2)
            logger.warning(
                "webhook_attempt_timeout",
                webhook_id=payload.webhook_id,
                attempt=attempt_num,
                error=attempt.error,
            )
        except httpx.RequestError as exc:
            attempt.error = f"request_error: {exc}"
            attempt.latency_ms = round((time.monotonic() - start) * 1000, 2)
            logger.warning(
                "webhook_attempt_request_error",
                webhook_id=payload.webhook_id,
                attempt=attempt_num,
                error=attempt.error,
            )
        except Exception as exc:
            attempt.error = f"unexpected: {exc}"
            attempt.latency_ms = round((time.monotonic() - start) * 1000, 2)
            logger.exception(
                "webhook_attempt_unexpected_error",
                webhook_id=payload.webhook_id,
                attempt=attempt_num,
            )

        return attempt

    def _compute_delay(self, attempt: int) -> float:
        """Compute the sleep duration before the next retry.

        Uses exponential backoff: ``initial_delay * multiplier^(attempt-1)``,
        clamped to ``max_delay``.  When jitter is enabled the result is
        randomly sampled from ``[0, computed_delay]`` to spread out traffic
        spikes (full-jitter strategy).

        Args:
            attempt: The attempt number that just *failed* (1-based).

        Returns:
            Seconds to sleep before the next attempt.
        """
        base = self._config.initial_delay_seconds * (
            self._config.backoff_multiplier ** (attempt - 1)
        )
        clamped = min(base, self._config.max_delay_seconds)
        if self._config.jitter:
            return random.uniform(0, clamped)
        return clamped

    def _should_retry(self, status_code: int) -> bool:
        """Return True if the HTTP status code warrants another attempt.

        Client errors (4xx) other than 408/429/425 are considered permanent
        failures and will not be retried so as to avoid spamming the receiver.

        Args:
            status_code: The HTTP status code from the last attempt.

        Returns:
            ``True`` when the delivery should be retried.
        """
        return status_code in self._RETRYABLE_STATUS_CODES

    @staticmethod
    def _sign_payload(body: bytes, secret: str) -> str:
        """Compute an HMAC-SHA256 hex digest of *body* using *secret*.

        The resulting hex string is placed in the ``X-Webhook-Signature``
        header as ``sha256=<hex>``, matching the convention used by GitHub,
        Stripe, and other major webhook providers.

        Args:
            body: The raw bytes of the JSON request body.
            secret: The shared signing secret.

        Returns:
            Lowercase hex HMAC-SHA256 digest (without the ``sha256=`` prefix).
        """
        return hmac.new(
            secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def verify_signature(body: bytes, secret: str, signature_header: str) -> bool:
        """Verify an incoming webhook signature against the expected HMAC.

        Use this on the receiving end to validate that a webhook payload
        was signed by a party that knows the shared secret.

        Args:
            body: Raw request body bytes.
            secret: Shared signing secret.
            signature_header: Value of the X-Webhook-Signature header
                (expected format: ``sha256=<hex>``).

        Returns:
            True if the signature is valid.
        """
        if not signature_header.startswith("sha256="):
            return False
        expected = hmac.new(
            secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        received = signature_header.removeprefix("sha256=")
        return hmac.compare_digest(expected, received)

    def _enqueue_dead_letter(self, payload: WebhookPayload) -> None:
        """Add a payload to the dead-letter queue and update metrics."""
        self._dlq.append(payload)
        self._stats["total_dead_lettered"] += 1
        dlq_size = len(self._dlq)
        webhook_dead_letter_queue_size.set(dlq_size)
        logger.warning(
            "webhook_dead_lettered",
            webhook_id=payload.webhook_id,
            event_type=payload.event_type,
            target_url=payload.target_url,
            dlq_size=dlq_size,
        )


__all__ = ["DeliveryAttempt", "WebhookDeliveryConfig", "WebhookDeliveryResult", "WebhookDeliveryService", "WebhookPayload"]
