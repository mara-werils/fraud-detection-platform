"""Pydantic v2 models for the fraud detection platform."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

import orjson
from pydantic import BaseModel, Field


def _orjson_dumps(v: Any, *, default: Any) -> str:
    return orjson.dumps(v, default=default).decode()


class TransactionType(StrEnum):
    PURCHASE = "purchase"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    REFUND = "refund"


class AlertSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class BaseSchema(BaseModel):
    """Base schema with orjson serialization config."""

    model_config = {
        "populate_by_name": True,
        "json_encoders": {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
            UUID: str,
        },
        "ser_json_bytes": "base64",
    }

    def to_json_bytes(self) -> bytes:
        return orjson.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_json_bytes(cls, data: bytes) -> BaseSchema:
        return cls.model_validate(orjson.loads(data))


class Transaction(BaseSchema):
    """Raw incoming transaction."""

    transaction_id: UUID = Field(default_factory=uuid4, description="Unique transaction ID")
    user_id: UUID = Field(..., description="ID of the user initiating the transaction")
    amount: Decimal = Field(..., ge=0, description="Transaction amount")
    currency: str = Field(default="USD", max_length=3, description="ISO 4217 currency code")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    merchant_id: str | None = Field(default=None, description="Merchant identifier")
    merchant_category: str | None = Field(default=None, description="Merchant category code")
    card_number_hash: str | None = Field(default=None, description="Hashed card number")
    ip_address: str | None = Field(default=None, description="Source IP address")
    device_id: str | None = Field(default=None, description="Device fingerprint")
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Transaction time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FeatureVector(BaseSchema):
    """Feature vector computed for a transaction."""

    transaction_id: UUID = Field(..., description="Associated transaction ID")
    user_id: UUID = Field(..., description="User ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Velocity features
    txn_count_1h: int = Field(default=0, description="Transaction count in last 1 hour")
    txn_count_24h: int = Field(default=0, description="Transaction count in last 24 hours")
    txn_amount_sum_1h: Decimal = Field(default=Decimal("0"), description="Sum of amounts in 1h")
    txn_amount_sum_24h: Decimal = Field(default=Decimal("0"), description="Sum of amounts in 24h")
    txn_amount_avg_24h: Decimal = Field(default=Decimal("0"), description="Average amount in 24h")

    # Behavioral features
    unique_merchants_24h: int = Field(default=0, description="Unique merchants in 24h")
    unique_countries_24h: int = Field(default=0, description="Unique countries in 24h")
    max_amount_24h: Decimal = Field(default=Decimal("0"), description="Max transaction in 24h")
    time_since_last_txn_seconds: float = Field(default=0.0, description="Seconds since last txn")

    # Location features
    distance_from_last_txn_km: float = Field(default=0.0, description="Distance from last txn")
    is_new_device: bool = Field(default=False, description="Whether this is a new device")
    is_new_merchant: bool = Field(default=False, description="Whether this is a new merchant")

    # Amount features
    amount_zscore: float = Field(default=0.0, description="Z-score of transaction amount")
    amount_to_avg_ratio: float = Field(default=0.0, description="Ratio of amount to user average")


class ScoredTransaction(BaseSchema):
    """Transaction with fraud score attached."""

    transaction_id: UUID = Field(..., description="Transaction ID")
    user_id: UUID = Field(..., description="User ID")
    amount: Decimal = Field(..., description="Transaction amount")
    currency: str = Field(default="USD", max_length=3)
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    timestamp: datetime = Field(..., description="Original transaction time")

    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Fraud probability score")
    model_version: str = Field(..., description="Model version used for scoring")
    scoring_latency_ms: float = Field(..., ge=0, description="Scoring latency in milliseconds")
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Scoring timestamp")

    features: FeatureVector | None = Field(default=None, description="Features used for scoring")
    is_flagged: bool = Field(default=False, description="Whether the transaction is flagged")
    flag_reason: str | None = Field(default=None, description="Reason for flagging")


class AlertEvent(BaseSchema):
    """Alert generated for a suspicious transaction."""

    alert_id: UUID = Field(default_factory=uuid4, description="Unique alert ID")
    transaction_id: UUID = Field(..., description="Associated transaction ID")
    user_id: UUID = Field(..., description="User ID")
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Fraud score that triggered alert")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    status: AlertStatus = Field(default=AlertStatus.OPEN, description="Alert status")
    reason: str = Field(..., description="Reason for the alert")
    amount: Decimal = Field(..., description="Transaction amount")
    currency: str = Field(default="USD", max_length=3)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    assigned_to: str | None = Field(default=None, description="Assigned analyst")
    notes: list[str] = Field(default_factory=list, description="Investigation notes")
