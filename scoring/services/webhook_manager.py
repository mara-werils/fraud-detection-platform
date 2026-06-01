"""Webhook management for configurable alert destinations."""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from typing import Any
from uuid import uuid4

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class WebhookEvent(StrEnum):
    TRANSACTION_FLAGGED = "transaction.flagged"
    TRANSACTION_BLOCKED = "transaction.blocked"
    CASE_CREATED = "case.created"
    CASE_RESOLVED = "case.resolved"
    DRIFT_DETECTED = "drift.detected"
    MODEL_UPDATED = "model.updated"


class WebhookStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"


class WebhookConfig(BaseModel):
    """Configuration for a registered webhook."""

    webhook_id: str = Field(default_factory=lambda: str(uuid4()))
    url: str
    secret: str = Field(default_factory=lambda: f"whsec_{uuid4().hex[:32]}")
    events: list[str] = Field(default_factory=lambda: ["transaction.flagged"])
    status: WebhookStatus = WebhookStatus.ACTIVE
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_triggered: datetime | None = None
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0


class WebhookDelivery(BaseModel):
    """Record of a webhook delivery attempt."""

    delivery_id: str = Field(default_factory=lambda: str(uuid4()))
    webhook_id: str
    event: str
    status_code: int | None = None
    success: bool = False
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WebhookManager:
    """Manages webhook registrations and deliveries.

    Supports event filtering, HMAC signature verification,
    automatic disabling after consecutive failures, and delivery logging.

    Args:
        max_consecutive_failures: Failures before auto-pausing a webhook.
        delivery_timeout: HTTP timeout for webhook delivery in seconds.
    """

    def __init__(
        self,
        max_consecutive_failures: int = 5,
        delivery_timeout: float = 10.0,
    ) -> None:
        self._lock = Lock()
        self._webhooks: dict[str, WebhookConfig] = {}
        self._deliveries: list[WebhookDelivery] = []
        self._max_failures = max_consecutive_failures
        self._timeout = delivery_timeout

    def register(
        self,
        url: str,
        events: list[str] | None = None,
        description: str = "",
        secret: str | None = None,
    ) -> WebhookConfig:
        """Register a new webhook.

        Args:
            url: The URL to deliver events to.
            events: List of event types to subscribe to.
            description: Human-readable description.
            secret: Optional custom signing secret.

        Returns:
            The created WebhookConfig with the signing secret.
        """
        with self._lock:
            webhook = WebhookConfig(
                url=url,
                events=events or ["transaction.flagged"],
                description=description,
            )
            if secret:
                webhook.secret = secret

            self._webhooks[webhook.webhook_id] = webhook

            logger.info(
                "webhook_registered",
                webhook_id=webhook.webhook_id,
                url=url,
                events=webhook.events,
            )

            return webhook

    def unregister(self, webhook_id: str) -> bool:
        """Remove a webhook registration."""
        with self._lock:
            if webhook_id in self._webhooks:
                del self._webhooks[webhook_id]
                return True
            return False

    def update(
        self,
        webhook_id: str,
        url: str | None = None,
        events: list[str] | None = None,
        status: str | None = None,
    ) -> WebhookConfig | None:
        """Update a webhook configuration."""
        with self._lock:
            webhook = self._webhooks.get(webhook_id)
            if not webhook:
                return None

            if url:
                webhook.url = url
            if events:
                webhook.events = events
            if status:
                webhook.status = WebhookStatus(status)
                if status == "active":
                    webhook.consecutive_failures = 0

            return webhook

    def list_webhooks(self) -> list[WebhookConfig]:
        """List all registered webhooks."""
        with self._lock:
            return list(self._webhooks.values())

    def get(self, webhook_id: str) -> WebhookConfig | None:
        """Get a webhook by ID."""
        return self._webhooks.get(webhook_id)

    async def deliver(self, event: str, payload: dict[str, Any]) -> list[WebhookDelivery]:
        """Deliver an event to all subscribed webhooks.

        Args:
            event: The event type (e.g., "transaction.flagged").
            payload: The event payload to deliver.

        Returns:
            List of delivery attempt records.
        """
        deliveries: list[WebhookDelivery] = []

        with self._lock:
            targets = [
                wh for wh in self._webhooks.values()
                if wh.status == WebhookStatus.ACTIVE and event in wh.events
            ]

        for webhook in targets:
            delivery = await self._deliver_single(webhook, event, payload)
            deliveries.append(delivery)

            with self._lock:
                self._deliveries.append(delivery)
                # Keep only last 1000 deliveries
                if len(self._deliveries) > 1000:
                    self._deliveries = self._deliveries[-1000:]

        return deliveries

    async def _deliver_single(
        self,
        webhook: WebhookConfig,
        event: str,
        payload: dict[str, Any],
    ) -> WebhookDelivery:
        """Deliver to a single webhook."""
        import orjson

        body = orjson.dumps({
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": payload,
        })

        # Compute HMAC signature
        signature = hmac.new(
            webhook.secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": f"sha256={signature}",
            "X-Webhook-Event": event,
            "X-Webhook-ID": webhook.webhook_id,
        }

        delivery = WebhookDelivery(
            webhook_id=webhook.webhook_id,
            event=event,
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(webhook.url, content=body, headers=headers)

            delivery.status_code = resp.status_code
            delivery.success = 200 <= resp.status_code < 300

            with self._lock:
                if delivery.success:
                    webhook.success_count += 1
                    webhook.consecutive_failures = 0
                else:
                    webhook.failure_count += 1
                    webhook.consecutive_failures += 1

                webhook.last_triggered = datetime.now(UTC)

                # Auto-pause after too many failures
                if webhook.consecutive_failures >= self._max_failures:
                    webhook.status = WebhookStatus.FAILED
                    logger.warning(
                        "webhook_auto_paused",
                        webhook_id=webhook.webhook_id,
                        consecutive_failures=webhook.consecutive_failures,
                    )

        except Exception as exc:
            delivery.success = False
            delivery.error = str(exc)

            with self._lock:
                webhook.failure_count += 1
                webhook.consecutive_failures += 1
                if webhook.consecutive_failures >= self._max_failures:
                    webhook.status = WebhookStatus.FAILED

        return delivery

    def get_deliveries(self, webhook_id: str | None = None, limit: int = 50) -> list[WebhookDelivery]:
        """Get recent delivery records."""
        with self._lock:
            deliveries = list(self._deliveries)

        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        return deliveries[-limit:]
