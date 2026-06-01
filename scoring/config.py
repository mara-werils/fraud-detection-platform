"""Scoring service configuration."""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScoringConfig(BaseSettings):
    """Configuration specific to the scoring service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "scoring-service"
    kafka_topic_raw_txn: str = "raw_txn"
    kafka_topic_scored: str = "scored"
    kafka_topic_alerts: str = "alerts"
    kafka_max_retries: int = 5
    kafka_retry_backoff_ms: int = 1000

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20
    redis_socket_timeout: float = 5.0

    # Scoring
    scoring_host: str = "0.0.0.0"
    scoring_port: int = 8000
    score_threshold_block: float = 0.8
    score_threshold_review: float = 0.5

    # Model
    model_version: str = "rule-engine-v1"
    model_type: str = "rule-based"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Authentication
    api_auth_enabled: bool = False
    api_key: str = "fdp_dev_key_change_me_in_production"
    jwt_secret_key: str = "change-me-in-production-use-strong-secret"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_days: int = 30

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://fraud:fraud@localhost:5432/fraud"
    db_pool_size: int = 20
    db_max_overflow: int = 40

    # Feature cache
    feature_cache_ttl_seconds: int = 300
    feature_cache_l1_max_size: int = 10_000

    # Batch inference
    batch_inference_max_size: int = 64
    batch_inference_max_wait_ms: int = 10

    # SLO
    slo_p99_latency_ms: float = 200.0
    slo_error_rate_threshold: float = 0.001

    # CORS
    cors_allowed_origins: str = "*"
    cors_allow_credentials: bool = True

    # Graceful shutdown
    shutdown_timeout_seconds: int = 30

    # Service
    service_name: str = "scoring-service"
    service_version: str = "0.2.0"

    @field_validator("score_threshold_block")
    @classmethod
    def validate_block_threshold(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("score_threshold_block must be in (0, 1]")
        return v

    @field_validator("score_threshold_review")
    @classmethod
    def validate_review_threshold(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("score_threshold_review must be in (0, 1]")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    @field_validator("redis_max_connections")
    @classmethod
    def validate_redis_pool(cls, v: int) -> int:
        if v < 1:
            raise ValueError("redis_max_connections must be >= 1")
        return v
