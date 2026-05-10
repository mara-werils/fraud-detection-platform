"""Configuration for the data lake archiver service."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ArchiverConfig(BaseSettings):
    """Archiver configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="ARCHIVER_",
    )

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "data-lake-archiver"
    kafka_topic_scored: str = "scored"
    kafka_max_retries: int = 5
    kafka_retry_backoff_ms: int = 1000

    # MinIO / S3
    minio_endpoint: str = "localhost:9010"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket: str = "fraud-data"
    minio_use_ssl: bool = False

    # Buffering
    buffer_max_messages: int = 1000
    buffer_flush_interval_seconds: int = 60

    # Paths
    base_prefix: str = "transactions"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
