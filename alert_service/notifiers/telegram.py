"""Telegram notifier using the Bot API."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from shared.schemas import AlertEvent, AlertSeverity

__all__ = [
    "TelegramNotifier",
]

logger = structlog.get_logger(__name__)


class TelegramNotifier:
    """Send alert notifications via Telegram Bot API.

    Formats alerts as rich Markdown messages with emoji indicators
    and retries with exponential backoff on failure.
    """

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        max_retries: int = 3,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        await logger.ainfo("telegram_notifier_started")

    async def stop(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await logger.ainfo("telegram_notifier_stopped")

    def _format_message(self, alert: AlertEvent, extra: dict[str, Any] | None = None) -> str:
        """Format an alert event into a rich Telegram Markdown message.

        Args:
            alert: The alert event to format.
            extra: Optional extra data from the scored transaction.

        Returns:
            Formatted Markdown string.
        """
        extra = extra or {}

        if alert.severity in (AlertSeverity.CRITICAL, AlertSeverity.HIGH):
            header = "\U0001f6a8 *BLOCKED*"
        else:
            header = "\u26a0\ufe0f *REVIEW*"

        lines = [
            header,
            "",
            f"\U0001f4cb *Transaction ID:* `{alert.transaction_id}`",
            f"\U0001f4b0 *Amount:* {alert.amount} {alert.currency}",
            f"\U0001f464 *User ID:* `{alert.user_id}`",
            f"\U0001f3af *Fraud Score:* {alert.fraud_score:.4f}",
            f"\U0001f534 *Severity:* {alert.severity.value.upper()}",
            f"\U0001f4c5 *Time:* {alert.created_at.isoformat()}",
        ]

        # Score breakdown if available
        xgboost_score = extra.get("xgboost_score")
        gnn_score = extra.get("gnn_score")
        ensemble_score = extra.get("ensemble_score")
        if any(s is not None for s in (xgboost_score, gnn_score, ensemble_score)):
            lines.append("")
            lines.append("\U0001f4ca *Score Breakdown:*")
            if xgboost_score is not None:
                lines.append(f"  \u2022 XGBoost: {xgboost_score:.4f}")
            if gnn_score is not None:
                lines.append(f"  \u2022 GNN: {gnn_score:.4f}")
            if ensemble_score is not None:
                lines.append(f"  \u2022 Ensemble: {ensemble_score:.4f}")

        # Top triggered features
        top_features: list[str] = extra.get("top_features", [])
        if top_features:
            lines.append("")
            lines.append("\U0001f50d *Top Triggered Features:*")
            for feature in top_features[:5]:
                lines.append(f"  \u2022 {feature}")

        # LLM explanation
        explanation = extra.get("explanation")
        if explanation:
            lines.append("")
            lines.append(f"\U0001f4ac *Explanation:* {explanation}")

        lines.append("")
        lines.append(f"\U0001f4dd *Reason:* {alert.reason}")

        return "\n".join(lines)

    async def send(self, alert: AlertEvent, extra: dict[str, Any] | None = None) -> bool:
        """Send an alert notification to Telegram.

        Args:
            alert: The alert event to send.
            extra: Optional extra context (score breakdown, features, explanation).

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        if self._client is None:
            await logger.aerror("telegram_client_not_initialized")
            return False

        message = self._format_message(alert, extra)
        url = self.BASE_URL.format(token=self._bot_token)
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.post(url, json=payload)
                if response.status_code == 200:
                    await logger.ainfo(
                        "telegram_alert_sent",
                        alert_id=str(alert.alert_id),
                        severity=alert.severity.value,
                    )
                    return True

                await logger.awarning(
                    "telegram_api_error",
                    status_code=response.status_code,
                    response_body=response.text,
                    attempt=attempt,
                )
            except httpx.HTTPError as exc:
                await logger.awarning(
                    "telegram_request_error",
                    error=str(exc),
                    attempt=attempt,
                )

            if attempt < self._max_retries:
                backoff = 2 ** (attempt - 1)
                await asyncio.sleep(backoff)

        # Fallback: log the alert if Telegram is unavailable
        await logger.aerror(
            "telegram_alert_fallback",
            alert_id=str(alert.alert_id),
            user_id=str(alert.user_id),
            transaction_id=str(alert.transaction_id),
            fraud_score=alert.fraud_score,
            severity=alert.severity.value,
            amount=str(alert.amount),
            reason=alert.reason,
        )
        return False
