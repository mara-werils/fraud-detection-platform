"""Alert service entry point.

Consumes scored transactions from the 'alerts' Kafka topic,
deduplicates alerts, and routes notifications to Telegram,
webhook, and email notifiers based on severity.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any

import structlog

from alert_service.config import AlertConfig
from alert_service.dedup import AlertDeduplicator
from alert_service.notifiers.email import EmailNotifier
from alert_service.notifiers.telegram import TelegramNotifier
from alert_service.notifiers.webhook import WebhookNotifier
from shared.kafka_utils import KafkaConsumerWrapper
from shared.logging import setup_logging
from shared.schemas import AlertEvent, AlertSeverity, ScoredTransaction

logger = structlog.get_logger(__name__)


class AlertService:
    """Core alert service that orchestrates consumption and notification."""

    def __init__(self, config: AlertConfig) -> None:
        self._config = config
        self._consumer = KafkaConsumerWrapper(
            bootstrap_servers=config.kafka_bootstrap_servers,
            topics=[config.kafka_topic_alerts],
            group_id=config.kafka_consumer_group,
            max_retries=config.kafka_max_retries,
            retry_backoff_ms=config.kafka_retry_backoff_ms,
        )
        self._dedup = AlertDeduplicator(ttl_seconds=config.dedup_ttl_seconds)
        self._telegram: TelegramNotifier | None = None
        self._webhook: WebhookNotifier | None = None
        self._email: EmailNotifier | None = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Initialize all components and start consuming."""
        await self._dedup.start()

        if self._config.telegram_enabled:
            self._telegram = TelegramNotifier(
                bot_token=self._config.telegram_bot_token,
                chat_id=self._config.telegram_chat_id,
                max_retries=self._config.telegram_max_retries,
            )
            await self._telegram.start()

        if self._config.webhook_enabled:
            self._webhook = WebhookNotifier(
                url=self._config.webhook_url,
                secret=self._config.webhook_secret,
                timeout_seconds=self._config.webhook_timeout_seconds,
                max_retries=self._config.webhook_max_retries,
                webhook_format=self._config.webhook_format,
            )
            await self._webhook.start()

        if self._config.email_enabled:
            self._email = EmailNotifier(
                smtp_host=self._config.email_smtp_host,
                smtp_port=self._config.email_smtp_port,
                smtp_user=self._config.email_smtp_user,
                smtp_password=self._config.email_smtp_password,
                use_tls=self._config.email_smtp_use_tls,
                from_address=self._config.email_from_address,
                recipients=self._config.email_recipient_list,
                batch_interval_seconds=self._config.email_batch_interval_seconds,
            )
            await self._email.start()

        await self._consumer.start()
        await logger.ainfo("alert_service_started")

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        await logger.ainfo("alert_service_stopping")
        await self._consumer.stop()
        await self._dedup.stop()

        if self._telegram is not None:
            await self._telegram.stop()
        if self._webhook is not None:
            await self._webhook.stop()
        if self._email is not None:
            await self._email.stop()

        await logger.ainfo("alert_service_stopped")

    async def run(self) -> None:
        """Run the alert service until shutdown is requested."""
        await self.start()

        # Set up graceful shutdown via signals
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal, sig)

        try:
            await self._consumer.consume(self._handle_message)
        except Exception:
            await logger.aexception("alert_service_consumer_error")
        finally:
            await self.stop()

    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle OS shutdown signals."""
        asyncio.create_task(self._signal_shutdown(sig))

    async def _signal_shutdown(self, sig: signal.Signals) -> None:
        """Process shutdown signal."""
        await logger.ainfo("shutdown_signal_received", signal=sig.name)
        self._shutdown_event.set()
        await self._consumer.stop()

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Process a single alert message from Kafka.

        Args:
            message: Deserialized JSON message from the alerts topic.
        """
        try:
            scored = ScoredTransaction.model_validate(message)
        except Exception:
            await logger.aexception("invalid_alert_message", raw_message=str(message)[:500])
            return

        # Only process BLOCK/REVIEW decisions (flagged transactions)
        if not scored.is_flagged:
            return

        # Determine alert type based on fraud score
        alert_type = "BLOCK" if scored.fraud_score >= 0.9 else "REVIEW"

        # Deduplicate
        if await self._dedup.is_duplicate(scored.user_id, alert_type):
            return

        # Determine severity
        severity = self._compute_severity(scored.fraud_score)

        # Build alert event
        alert = AlertEvent(
            transaction_id=scored.transaction_id,
            user_id=scored.user_id,
            fraud_score=scored.fraud_score,
            severity=severity,
            reason=scored.flag_reason or f"Fraud score {scored.fraud_score:.4f} exceeded threshold",
            amount=scored.amount,
            currency=scored.currency,
        )

        # Build extra context for rich notifications
        extra = self._build_extra_context(scored)

        await logger.ainfo(
            "processing_alert",
            alert_id=str(alert.alert_id),
            alert_type=alert_type,
            severity=severity.value,
            fraud_score=scored.fraud_score,
            user_id=str(scored.user_id),
        )

        # Route to notifiers based on severity
        tasks: list[asyncio.Task[Any]] = []

        if self._telegram is not None:
            if severity in (AlertSeverity.CRITICAL, AlertSeverity.HIGH):
                tasks.append(asyncio.create_task(self._telegram.send(alert, extra)))
            elif severity == AlertSeverity.MEDIUM:
                tasks.append(asyncio.create_task(self._telegram.send(alert, extra)))

        if self._webhook is not None:
            tasks.append(asyncio.create_task(self._webhook.send(alert)))

        if self._email is not None:
            tasks.append(asyncio.create_task(self._email.enqueue(alert)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _compute_severity(fraud_score: float) -> AlertSeverity:
        """Map fraud score to alert severity.

        Args:
            fraud_score: The fraud probability score (0.0 - 1.0).

        Returns:
            Appropriate AlertSeverity level.
        """
        if fraud_score >= 0.95:
            return AlertSeverity.CRITICAL
        elif fraud_score >= 0.85:
            return AlertSeverity.HIGH
        elif fraud_score >= 0.75:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW

    @staticmethod
    def _build_extra_context(scored: ScoredTransaction) -> dict[str, Any]:
        """Extract extra context from a scored transaction for rich notifications.

        Args:
            scored: The scored transaction.

        Returns:
            Dictionary with score breakdown, features, and explanation.
        """
        extra: dict[str, Any] = {}

        if scored.features is not None:
            # Identify top triggered features
            top_features: list[str] = []
            f = scored.features

            if f.amount_zscore > 2.0:
                top_features.append(f"Amount Z-score: {f.amount_zscore:.2f}")
            if f.amount_to_avg_ratio > 3.0:
                top_features.append(f"Amount/avg ratio: {f.amount_to_avg_ratio:.2f}")
            if f.txn_count_1h > 5:
                top_features.append(f"Txn count (1h): {f.txn_count_1h}")
            if f.unique_countries_24h > 3:
                top_features.append(f"Unique countries (24h): {f.unique_countries_24h}")
            if f.distance_from_last_txn_km > 500:
                top_features.append(
                    f"Distance from last txn: {f.distance_from_last_txn_km:.0f} km"
                )
            if f.is_new_device:
                top_features.append("New device detected")
            if f.is_new_merchant:
                top_features.append("New merchant")
            if f.time_since_last_txn_seconds < 60:
                top_features.append(
                    f"Time since last txn: {f.time_since_last_txn_seconds:.0f}s"
                )

            extra["top_features"] = top_features

        # Include model metadata
        extra["ensemble_score"] = scored.fraud_score
        extra["model_version"] = scored.model_version

        # Explanation from flag_reason if present
        if scored.flag_reason:
            extra["explanation"] = scored.flag_reason

        return extra


async def main() -> None:
    """Entry point for the alert service."""
    config = AlertConfig()
    setup_logging(
        service_name="alert-service",
        log_level=config.log_level,
        log_format=config.log_format,
    )

    service = AlertService(config)
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
