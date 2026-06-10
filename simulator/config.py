"""Simulator-specific configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "SimulatorConfig",
]


class SimulatorConfig(BaseSettings):
    """Configuration for the transaction simulator service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SIMULATOR_",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_raw_txn: str = "raw_txn"
    kafka_max_retries: int = 5
    kafka_retry_backoff_ms: int = 1000

    # Simulator behaviour
    rate: int = 10
    fraud_ratio: float = 0.02
    duration: int = 0

    # User pool
    user_pool_size: int = 500

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
