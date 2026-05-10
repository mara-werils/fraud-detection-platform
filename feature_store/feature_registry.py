"""Feature definitions, registry, and versioning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence


class FeatureDtype(StrEnum):
    INT = "int"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOL = "bool"
    STRING = "string"


class FeatureSource(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Immutable definition of a single feature."""

    name: str
    dtype: FeatureDtype
    source: FeatureSource
    description: str
    version: int = 1


# ---------------------------------------------------------------------------
# Canonical registry of all features
# ---------------------------------------------------------------------------
FEATURE_REGISTRY: tuple[FeatureDefinition, ...] = (
    # Velocity features — counts
    FeatureDefinition("txn_count_1m", FeatureDtype.INT, FeatureSource.ONLINE, "Transaction count in last 1 minute"),
    FeatureDefinition("txn_count_5m", FeatureDtype.INT, FeatureSource.ONLINE, "Transaction count in last 5 minutes"),
    FeatureDefinition("txn_count_1h", FeatureDtype.INT, FeatureSource.ONLINE, "Transaction count in last 1 hour"),
    FeatureDefinition("txn_count_24h", FeatureDtype.INT, FeatureSource.ONLINE, "Transaction count in last 24 hours"),
    # Velocity features — amounts
    FeatureDefinition("txn_amount_sum_1m", FeatureDtype.DECIMAL, FeatureSource.ONLINE, "Sum of amounts in last 1 minute"),
    FeatureDefinition("txn_amount_sum_5m", FeatureDtype.DECIMAL, FeatureSource.ONLINE, "Sum of amounts in last 5 minutes"),
    FeatureDefinition("txn_amount_sum_1h", FeatureDtype.DECIMAL, FeatureSource.ONLINE, "Sum of amounts in last 1 hour"),
    FeatureDefinition("txn_amount_sum_24h", FeatureDtype.DECIMAL, FeatureSource.ONLINE, "Sum of amounts in last 24 hours"),
    FeatureDefinition("txn_amount_avg_1h", FeatureDtype.DECIMAL, FeatureSource.ONLINE, "Average amount in last 1 hour"),
    FeatureDefinition("txn_amount_avg_24h", FeatureDtype.DECIMAL, FeatureSource.ONLINE, "Average amount in last 24 hours"),
    FeatureDefinition("txn_amount_max_1h", FeatureDtype.DECIMAL, FeatureSource.ONLINE, "Max amount in last 1 hour"),
    FeatureDefinition("txn_amount_max_24h", FeatureDtype.DECIMAL, FeatureSource.ONLINE, "Max amount in last 24 hours"),
    # Amount statistics
    FeatureDefinition("amount_running_avg", FeatureDtype.FLOAT, FeatureSource.ONLINE, "Running average transaction amount"),
    FeatureDefinition("amount_running_max", FeatureDtype.FLOAT, FeatureSource.ONLINE, "Running max transaction amount"),
    FeatureDefinition("amount_running_sum", FeatureDtype.FLOAT, FeatureSource.ONLINE, "Running sum of all transaction amounts"),
    FeatureDefinition("amount_running_count", FeatureDtype.INT, FeatureSource.ONLINE, "Running count of all transactions"),
    FeatureDefinition("amount_zscore", FeatureDtype.FLOAT, FeatureSource.ONLINE, "Z-score of current amount vs history"),
    FeatureDefinition("amount_to_avg_ratio", FeatureDtype.FLOAT, FeatureSource.ONLINE, "Ratio of current amount to running average"),
    # Behavioral features
    FeatureDefinition("unique_merchants_24h", FeatureDtype.INT, FeatureSource.ONLINE, "Unique merchants in last 24 hours"),
    FeatureDefinition("unique_merchants_7d", FeatureDtype.INT, FeatureSource.ONLINE, "Unique merchants in last 7 days"),
    FeatureDefinition("is_new_merchant", FeatureDtype.BOOL, FeatureSource.ONLINE, "Whether merchant was never seen before"),
    FeatureDefinition("time_since_last_txn_seconds", FeatureDtype.FLOAT, FeatureSource.ONLINE, "Seconds since last transaction"),
    # Location features
    FeatureDefinition("distance_from_last_txn_km", FeatureDtype.FLOAT, FeatureSource.ONLINE, "Distance from last transaction in km"),
    FeatureDefinition("distance_from_home_km", FeatureDtype.FLOAT, FeatureSource.ONLINE, "Distance from usual location in km"),
    FeatureDefinition("last_latitude", FeatureDtype.FLOAT, FeatureSource.ONLINE, "Latitude of last transaction"),
    FeatureDefinition("last_longitude", FeatureDtype.FLOAT, FeatureSource.ONLINE, "Longitude of last transaction"),
    # Device features
    FeatureDefinition("is_new_device", FeatureDtype.BOOL, FeatureSource.ONLINE, "Whether device was never seen before"),
    FeatureDefinition("unique_devices_count", FeatureDtype.INT, FeatureSource.ONLINE, "Total unique devices used"),
    # Offline / aggregate features
    FeatureDefinition("merchant_fraud_rate", FeatureDtype.FLOAT, FeatureSource.OFFLINE, "Historical fraud rate for merchant"),
    FeatureDefinition("user_historical_fraud_count", FeatureDtype.INT, FeatureSource.OFFLINE, "Number of past fraudulent txns for user"),
    FeatureDefinition("user_account_age_days", FeatureDtype.INT, FeatureSource.OFFLINE, "Days since user's first transaction"),
    FeatureDefinition("user_avg_daily_txn_count", FeatureDtype.FLOAT, FeatureSource.OFFLINE, "Average daily transactions historically"),
    FeatureDefinition("user_avg_daily_amount", FeatureDtype.FLOAT, FeatureSource.OFFLINE, "Average daily spend historically"),
)


def get_feature_names(source: FeatureSource | None = None) -> list[str]:
    """Return ordered list of feature names, optionally filtered by source.

    Args:
        source: If provided, only return features matching this source.
                Features with source=BOTH match any filter.
    """
    names: list[str] = []
    for feat in FEATURE_REGISTRY:
        if source is None:
            names.append(feat.name)
        elif feat.source == source or feat.source == FeatureSource.BOTH:
            names.append(feat.name)
    return names


def get_feature_version() -> str:
    """Compute a deterministic hash of the current feature registry schema.

    Returns:
        SHA-256 hex digest (first 12 chars) representing the registry state.
    """
    parts: list[str] = []
    for feat in FEATURE_REGISTRY:
        parts.append(f"{feat.name}:{feat.dtype}:{feat.source}:{feat.version}")
    payload = "|".join(parts).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def get_feature_definitions(source: FeatureSource | None = None) -> Sequence[FeatureDefinition]:
    """Return feature definitions, optionally filtered by source."""
    if source is None:
        return FEATURE_REGISTRY
    return tuple(
        f for f in FEATURE_REGISTRY
        if f.source == source or f.source == FeatureSource.BOTH
    )
