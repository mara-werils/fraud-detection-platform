"""Generic webhook notifier with HMAC signature verification."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from typing import Any

import httpx
import orjson
import structlog

from shared.schemas import AlertEvent, AlertSeverity

__all__ = [
    "WebhookNotifier",
]

logger = structlog.get_logger(__name__)


class WebhookNotifier:
    """Send alert notifications via HTTP webhook.

    Supports generic JSON, PagerDuty, and Slack webhook formats.
    Signs payloads with HMAC-SHA256 for verification.
    """

    def __init__(
        self,
        url: str,
        secret: str,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        webhook_format: str = "generic",
    ) -> None:
        self._url = url
        self._secret = secret
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._format = webhook_format
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
        )
        await logger.ainfo(
            "webhook_notifier_started",
            url=self._url,
            format=self._format,
        )

    async def stop(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await logger.ainfo("webhook_notifier_stopped")

    def _compute_signature(self, payload_bytes: bytes) -> str:
        """Compute HMAC-SHA256 signature for the payload.

        Args:
            payload_bytes: The raw JSON payload bytes.

        Returns:
            Hex-encoded HMAC-SHA256 signature.
        """
        return hmac.new(
            self._secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()

    def _build_generic_payload(self, alert: AlertEvent) -> dict[str, Any]:
        """Build a generic JSON alert payload."""
        return {
            "alert_id": str(alert.alert_id),
            "transaction_id": str(alert.transaction_id),
            "user_id": str(alert.user_id),
            "fraud_score": alert.fraud_score,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "reason": alert.reason,
            "amount": str(alert.amount),
            "currency": alert.currency,
            "created_at": alert.created_at.isoformat(),
        }

    def _build_pagerduty_payload(self, alert: AlertEvent) -> dict[str, Any]:
        """Build a PagerDuty Events API v2 payload."""
        severity_map = {
            AlertSeverity.LOW: "info",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.HIGH: "error",
            AlertSeverity.CRITICAL: "critical",
        }
        return {
            "routing_key": self._secret,
            "event_action": "trigger",
            "dedup_key": str(alert.alert_id),
            "payload": {
                "summary": (
                    f"Fraud alert: {alert.severity.value.upper()} - "
                    f"Score {alert.fraud_score:.4f} - "
                    f"Amount {alert.amount} {alert.currency}"
                ),
                "source": "fraud-detection-platform",
                "severity": severity_map.get(alert.severity, "warning"),
                "component": "alert-service",
                "group": "fraud",
                "class": "fraud_alert",
                "custom_details": {
                    "transaction_id": str(alert.transaction_id),
                    "user_id": str(alert.user_id),
                    "fraud_score": alert.fraud_score,
                    "amount": str(alert.amount),
                    "currency": alert.currency,
                    "reason": alert.reason,
                },
            },
        }

    def _build_slack_payload(self, alert: AlertEvent) -> dict[str, Any]:
        """Build a Slack incoming webhook payload."""
        if alert.severity in (AlertSeverity.CRITICAL, AlertSeverity.HIGH):
            color = "#ff0000"
            icon = "\U0001f6a8"
        elif alert.severity == AlertSeverity.MEDIUM:
            color = "#ff9900"
            icon = "\u26a0\ufe0f"
        else:
            color = "#ffcc00"
            icon = "\u2139\ufe0f"

        return {
            "attachments": [
                {
                    "color": color,
                    "title": f"{icon} Fraud Alert - {alert.severity.value.upper()}",
                    "fields": [
                        {
                            "title": "Transaction ID",
                            "value": str(alert.transaction_id),
                            "short": False,
                        },
                        {
                            "title": "Fraud Score",
                            "value": f"{alert.fraud_score:.4f}",
                            "short": True,
                        },
                        {
                            "title": "Amount",
                            "value": f"{alert.amount} {alert.currency}",
                            "short": True,
                        },
                        {
                            "title": "User ID",
                            "value": str(alert.user_id),
                            "short": False,
                        },
                        {
                            "title": "Reason",
                            "value": alert.reason,
                            "short": False,
                        },
                    ],
                    "footer": "Fraud Detection Platform",
                    "ts": int(alert.created_at.timestamp()),
                }
            ]
        }

    def _build_payload(self, alert: AlertEvent) -> dict[str, Any]:
        """Build the payload based on the configured format.

        Args:
            alert: The alert event to serialize.

        Returns:
            The formatted payload dictionary.
        """
        if self._format == "pagerduty":
            return self._build_pagerduty_payload(alert)
        elif self._format == "slack":
            return self._build_slack_payload(alert)
        return self._build_generic_payload(alert)

    async def send(self, alert: AlertEvent) -> bool:
        """Send an alert via webhook.

        Args:
            alert: The alert event to send.

        Returns:
            True if the webhook was delivered successfully.
        """
        if self._client is None:
            await logger.aerror("webhook_client_not_initialized")
            return False

        payload = self._build_payload(alert)
        payload_bytes = orjson.dumps(payload)
        signature = self._compute_signature(payload_bytes)

        headers = {
            "Content-Type": "application/json",
            "X-Alert-Signature": f"sha256={signature}",
            "User-Agent": "FraudDetectionPlatform/1.0",
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.post(
                    self._url,
                    content=payload_bytes,
                    headers=headers,
                )
                if 200 <= response.status_code < 300:
                    await logger.ainfo(
                        "webhook_alert_sent",
                        alert_id=str(alert.alert_id),
                        status_code=response.status_code,
                        format=self._format,
                    )
                    return True

                await logger.awarning(
                    "webhook_response_error",
                    status_code=response.status_code,
                    response_body=response.text[:500],
                    attempt=attempt,
                )
            except httpx.HTTPError as exc:
                await logger.awarning(
                    "webhook_request_error",
                    error=str(exc),
                    attempt=attempt,
                )

            if attempt < self._max_retries:
                backoff = 2 ** (attempt - 1)
                await asyncio.sleep(backoff)

        await logger.aerror(
            "webhook_alert_failed",
            alert_id=str(alert.alert_id),
            url=self._url,
            format=self._format,
        )
        return False
