"""Alert service configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AlertConfig(BaseSettings):
    """Configuration for the alert service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "alert-service"
    kafka_topic_alerts: str = "alerts"
    kafka_max_retries: int = 5
    kafka_retry_backoff_ms: int = 1000

    # Deduplication
    dedup_ttl_seconds: int = 600  # 10 minutes

    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_max_retries: int = 3

    # Webhook
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_secret: str = ""
    webhook_timeout_seconds: float = 10.0
    webhook_max_retries: int = 3
    webhook_format: str = "generic"  # generic, pagerduty, slack

    # Email
    email_enabled: bool = False
    email_smtp_host: str = "localhost"
    email_smtp_port: int = 587
    email_smtp_user: str = ""
    email_smtp_password: str = ""
    email_smtp_use_tls: bool = True
    email_from_address: str = "alerts@fraud-detection.local"
    email_recipients: str = ""  # comma-separated
    email_batch_interval_seconds: int = 300  # 5 minutes

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Prometheus
    prometheus_port: int = 9094

    @property
    def email_recipient_list(self) -> list[str]:
        """Parse comma-separated email recipients into a list."""
        if not self.email_recipients:
            return []
        return [r.strip() for r in self.email_recipients.split(",") if r.strip()]
