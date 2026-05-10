"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for all fraud detection services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "fraud-detection"
    kafka_topic_raw_txn: str = "raw_txn"
    kafka_topic_scored: str = "scored"
    kafka_topic_alerts: str = "alerts"
    kafka_max_retries: int = 5
    kafka_retry_backoff_ms: int = 1000

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20
    redis_socket_timeout: float = 5.0

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = "clickhouse"
    clickhouse_database: str = "fraud"

    # MinIO / S3
    minio_endpoint: str = "localhost:9010"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket: str = "fraud-data"
    minio_use_ssl: bool = False

    # Scoring
    scoring_host: str = "0.0.0.0"
    scoring_port: int = 8000
    score_threshold: float = 0.75

    # Simulator
    simulator_tps: int = 100
    simulator_fraud_ratio: float = 0.02

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # OpenTelemetry
    otel_service_name: str = "fraud-detection-platform"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    # Prometheus
    prometheus_port: int = 9090
