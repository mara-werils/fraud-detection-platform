"""Feature Store service configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class FeatureStoreConfig(BaseSettings):
    """Configuration for the Feature Store service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "feature-store"
    kafka_topic_raw_txn: str = "raw_txn"
    kafka_max_retries: int = 5
    kafka_retry_backoff_ms: int = 1000

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20
    redis_socket_timeout: float = 5.0
    redis_key_ttl_days: int = 30

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = "clickhouse"
    clickhouse_database: str = "fraud"

    # Batching
    batch_size: int = 500
    batch_flush_interval_seconds: float = 5.0

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Service
    service_name: str = "feature-store"
