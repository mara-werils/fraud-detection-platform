"""Feature engineering: 30+ features organised by group.

All transformations use pandas vectorised operations for speed.
This module is used both during offline training and for feature
backfilling.  Online (real-time) features are computed in the
feature store; this module focuses on *batch* engineering.

Usage::

    from ml_pipeline.feature_engineering import engineer_features
    df = engineer_features(raw_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature importance tracker
# ---------------------------------------------------------------------------

@dataclass
class FeatureImportanceTracker:
    """Accumulates feature importance scores from multiple sources."""

    scores: dict[str, float] = field(default_factory=dict)

    def update(self, name: str, importance: float) -> None:
        self.scores[name] = self.scores.get(name, 0.0) + importance

    def update_many(self, mapping: dict[str, float]) -> None:
        for name, imp in mapping.items():
            self.update(name, imp)

    def top_k(self, k: int = 20) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:k]


_tracker = FeatureImportanceTracker()


def get_importance_tracker() -> FeatureImportanceTracker:
    """Return the module-level importance tracker."""
    return _tracker


# ---------------------------------------------------------------------------
# Feature group helpers
# ---------------------------------------------------------------------------

def _transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction-level features: amount transforms, category/channel encoding."""
    df = df.copy()

    # Amount transforms
    df["f_amount_log"] = np.log1p(df["amount"].astype(np.float64))

    user_stats = df.groupby("user_id")["amount"].transform
    user_mean = user_stats("mean")
    user_std = user_stats("std").fillna(1.0).replace(0.0, 1.0)
    df["f_amount_zscore"] = (df["amount"].astype(np.float64) - user_mean) / user_std

    df["f_is_round_amount"] = (df["amount"].astype(np.float64) % 10 == 0).astype(np.int32)

    # Category encoding (frequency)
    if "category" in df.columns:
        cat_freq = df["category"].value_counts(normalize=True)
        df["f_category_encoded"] = df["category"].map(cat_freq).astype(np.float64)
    else:
        df["f_category_encoded"] = 0.0

    # Channel encoding (frequency)
    if "channel" in df.columns:
        ch_freq = df["channel"].value_counts(normalize=True)
        df["f_channel_encoded"] = df["channel"].map(ch_freq).astype(np.float64)
    else:
        df["f_channel_encoded"] = 0.0

    return df


def _velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Velocity features: txn counts and sums over rolling windows.

    These are pre-computed in the dataset generator; this function
    ensures they exist and fills defaults.
    """
    velocity_cols = {
        "f_txn_count_1m": 0,
        "f_txn_count_5m": 0,
        "f_txn_count_1h": 0,
        "f_txn_count_24h": 0,
        "f_txn_sum_1h": 0.0,
        "f_txn_sum_24h": 0.0,
    }
    for col, default in velocity_cols.items():
        if col not in df.columns:
            df[col] = default
    return df


def _amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Amount-related features: ratios, deviations, percentiles."""
    amount = df["amount"].astype(np.float64)

    # Per-user statistics
    user_mean = df.groupby("user_id")["amount"].transform("mean").astype(np.float64)
    user_mean_safe = user_mean.replace(0.0, 1.0)

    if "f_amount_to_avg_ratio" not in df.columns:
        df["f_amount_to_avg_ratio"] = amount / user_mean_safe

    if "f_amount_deviation" not in df.columns:
        user_std = df.groupby("user_id")["amount"].transform("std").fillna(1.0).replace(0.0, 1.0)
        df["f_amount_deviation"] = (amount - user_mean) / user_std

    if "f_max_amount_7d" not in df.columns:
        df["f_max_amount_7d"] = 0.0
    max_7d_safe = df["f_max_amount_7d"].replace(0.0, 1.0)
    df["f_max_amount_7d_ratio"] = amount / max_7d_safe

    df["f_amount_percentile"] = amount.rank(pct=True)

    return df


def _geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Geo features: distances, new-city flag, geo velocity."""
    geo_cols = {
        "f_distance_from_home_km": 0.0,
        "f_distance_from_last_km": 0.0,
        "f_is_new_city": 0,
    }
    for col, default in geo_cols.items():
        if col not in df.columns:
            df[col] = default

    # Geo velocity: distance / time  (km/h)
    time_since = df.get("f_time_since_last_txn_s", pd.Series(3600.0, index=df.index))
    time_since_h = time_since.replace(0.0, 1.0) / 3600.0
    df["f_geo_velocity_kmh"] = df["f_distance_from_last_km"].astype(np.float64) / time_since_h

    return df


def _device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Device features: new-device flag, age, txn count."""
    device_cols = {
        "f_is_new_device": 0,
        "f_device_age_days": 0,
        "f_device_txn_count": 0,
    }
    for col, default in device_cols.items():
        if col not in df.columns:
            df[col] = default
    return df


def _merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merchant features: fraud rate, category risk, new-merchant flag."""
    merchant_cols = {
        "f_merchant_category_risk": 0.01,
        "f_is_new_merchant": 0,
    }
    for col, default in merchant_cols.items():
        if col not in df.columns:
            df[col] = default

    # Merchant fraud rate (computed from labelled data if available)
    if "merchant_id" in df.columns and "is_fraud" in df.columns:
        merchant_fr = df.groupby("merchant_id")["is_fraud"].mean()
        df["f_merchant_fraud_rate"] = (
            df["merchant_id"].map(merchant_fr).fillna(0.0).astype(np.float64)
        )
    elif "f_merchant_fraud_rate" not in df.columns:
        df["f_merchant_fraud_rate"] = 0.0

    return df


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time features: cyclical hour/day, weekend, night flags."""
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        hour = ts.dt.hour.astype(np.float64)
        dow = ts.dt.dayofweek.astype(np.float64)
    elif "f_hour_of_day" in df.columns:
        hour = df["f_hour_of_day"].astype(np.float64)
        dow = df.get("f_day_of_week", pd.Series(0.0, index=df.index)).astype(np.float64)
    else:
        hour = pd.Series(12.0, index=df.index)
        dow = pd.Series(0.0, index=df.index)

    # Cyclical encoding
    df["f_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["f_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["f_day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["f_day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)

    if "f_is_weekend" not in df.columns:
        df["f_is_weekend"] = (dow >= 5).astype(np.int32)
    if "f_is_night" not in df.columns:
        df["f_is_night"] = ((hour < 6) | (hour >= 23)).astype(np.int32)

    return df


def _behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioral features: timing patterns, session counts."""
    behav_cols = {
        "f_time_since_last_txn_s": 3600.0,
        "f_avg_time_between_txn_s": 3600.0,
    }
    for col, default in behav_cols.items():
        if col not in df.columns:
            df[col] = default

    # Session transaction count (txns within 30 min window per user)
    if "timestamp" in df.columns and "user_id" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df_sorted = df.assign(_ts=ts).sort_values(["user_id", "_ts"])
        # Count txns within a rolling 30-minute window per user
        df_sorted["f_session_txn_count"] = (
            df_sorted.groupby("user_id")["_ts"]
            .transform(lambda s: s.rolling("30min").count())
            .fillna(1)
            .astype(np.int32)
        )
        df["f_session_txn_count"] = df_sorted["f_session_txn_count"].reindex(df.index).fillna(1).astype(np.int32)
        df.drop(columns=["_ts"], errors="ignore", inplace=True)
    elif "f_session_txn_count" not in df.columns:
        df["f_session_txn_count"] = 1

    return df


# ---------------------------------------------------------------------------
# Feature list definition
# ---------------------------------------------------------------------------

ALL_FEATURE_NAMES: list[str] = [
    # Transaction
    "f_amount_log",
    "f_amount_zscore",
    "f_is_round_amount",
    "f_category_encoded",
    "f_channel_encoded",
    # Velocity
    "f_txn_count_1m",
    "f_txn_count_5m",
    "f_txn_count_1h",
    "f_txn_count_24h",
    "f_txn_sum_1h",
    "f_txn_sum_24h",
    # Amount
    "f_amount_to_avg_ratio",
    "f_amount_deviation",
    "f_max_amount_7d_ratio",
    "f_amount_percentile",
    # Geo
    "f_distance_from_home_km",
    "f_distance_from_last_km",
    "f_is_new_city",
    "f_geo_velocity_kmh",
    # Device
    "f_is_new_device",
    "f_device_age_days",
    "f_device_txn_count",
    # Merchant
    "f_merchant_fraud_rate",
    "f_merchant_category_risk",
    "f_is_new_merchant",
    # Time
    "f_hour_sin",
    "f_hour_cos",
    "f_day_of_week_sin",
    "f_day_of_week_cos",
    "f_is_weekend",
    "f_is_night",
    # Behavioral
    "f_time_since_last_txn_s",
    "f_avg_time_between_txn_s",
    "f_session_txn_count",
]


def get_feature_names() -> list[str]:
    """Return the canonical ordered list of feature names."""
    return list(ALL_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    *,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply all feature engineering groups to *df* and return it.

    Columns that already exist (e.g. pre-computed by the dataset generator)
    are preserved; missing ones are computed.

    Args:
        df: Raw transaction DataFrame.
        feature_names: If given, only keep these feature columns in the
            output (plus non-feature columns).  Defaults to
            ``ALL_FEATURE_NAMES``.

    Returns:
        DataFrame with all engineered feature columns present.
    """
    log.info("feature_engineering_start", rows=len(df), existing_features=len([c for c in df.columns if c.startswith("f_")]))

    df = _transaction_features(df)
    df = _velocity_features(df)
    df = _amount_features(df)
    df = _geo_features(df)
    df = _device_features(df)
    df = _merchant_features(df)
    df = _time_features(df)
    df = _behavioral_features(df)

    final_names = list(feature_names) if feature_names is not None else ALL_FEATURE_NAMES

    # Ensure every expected feature exists
    for fname in final_names:
        if fname not in df.columns:
            log.warning("missing_feature_defaulting_zero", feature=fname)
            df[fname] = 0.0

    all_feature_cols = [c for c in df.columns if c.startswith("f_")]
    log.info("feature_engineering_done", total_features=len(all_feature_cols), requested=len(final_names))

    return df
