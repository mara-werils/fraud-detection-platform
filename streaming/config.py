"""Configuration for the streaming pipeline service."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class StreamingConfig(BaseSettings):
    """Streaming pipeline configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="STREAMING_",
    )

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "streaming-pipeline"
    kafka_topic_raw_txn: str = "raw_txn"
    kafka_topic_enriched: str = "enriched"
    kafka_max_retries: int = 5
    kafka_retry_backoff_ms: int = 1000

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20
    redis_socket_timeout: float = 5.0

    # GeoIP
    geoip_db_path: str = "/data/GeoLite2-City.mmdb"

    # Processing
    batch_size: int = 100
    max_concurrent_tasks: int = 50

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Prometheus
    prometheus_port: int = 9091
