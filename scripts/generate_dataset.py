#!/usr/bin/env python3
"""Generate a labeled training dataset for fraud detection models.

Produces 100K+ transactions with ~2% fraud rate covering all 5 fraud patterns,
and computes 30+ engineered features. Output is saved as a Parquet file.

Usage:
    python scripts/generate_dataset.py --output data/training_data.parquet --size 100000
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)

try:
    import pyarrow  # noqa: F401
except ImportError:
    print("ERROR: pyarrow is required. Install with: pip install pyarrow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = [
    "groceries",
    "electronics",
    "restaurants",
    "gas",
    "travel",
    "entertainment",
    "healthcare",
    "utilities",
]

CATEGORY_RISK = {
    "groceries": 0.005,
    "electronics": 0.04,
    "restaurants": 0.008,
    "gas": 0.01,
    "travel": 0.03,
    "entertainment": 0.015,
    "healthcare": 0.005,
    "utilities": 0.003,
}

CHANNELS = ["mobile_app", "web", "pos_terminal", "atm"]
CURRENCIES = ["USD", "EUR", "KZT", "GBP"]

FRAUD_PATTERNS = [
    "card_testing",
    "account_takeover",
    "geo_anomaly",
    "velocity_abuse",
    "merchant_collusion",
]

CITIES = [
    (40.7128, -74.0060),
    (34.0522, -118.2437),
    (51.5074, -0.1278),
    (48.8566, 2.3522),
    (43.2220, 76.8512),
    (51.1694, 71.4491),
    (35.6762, 139.6503),
    (55.7558, 37.6173),
    (25.2048, 55.2708),
    (37.5665, 126.9780),
    (41.8781, -87.6298),
    (29.7604, -95.3698),
    (-33.8688, 151.2093),
    (19.4326, -99.1332),
    (59.3293, 18.0686),
]


# ---------------------------------------------------------------------------
# User simulation context
# ---------------------------------------------------------------------------


class UserContext:
    """Maintains running statistics for a simulated user."""

    __slots__ = (
        "user_id",
        "home_lat",
        "home_lon",
        "avg_amount",
        "device_id",
        "merchant_ids",
        "txn_history",
        "last_lat",
        "last_lon",
        "last_txn_time",
        "total_amount",
        "txn_count",
    )

    def __init__(self) -> None:
        self.user_id = str(uuid4())
        city = random.choice(CITIES)
        self.home_lat = city[0] + random.uniform(-0.05, 0.05)
        self.home_lon = city[1] + random.uniform(-0.05, 0.05)
        self.avg_amount = random.lognormvariate(4.0, 0.8)
        self.device_id = uuid4().hex[:12]
        self.merchant_ids = [str(uuid4())[:8] for _ in range(random.randint(3, 10))]
        self.txn_history: list[dict] = []
        self.last_lat = self.home_lat
        self.last_lon = self.home_lon
        self.last_txn_time: datetime | None = None
        self.total_amount = 0.0
        self.txn_count = 0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points in km."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Legitimate transaction generator
# ---------------------------------------------------------------------------


def _generate_legit_txn(user: UserContext, ts: datetime) -> dict:
    """Generate a single legitimate transaction with features."""
    category = random.choice(CATEGORIES)
    amount = abs(random.gauss(user.avg_amount, user.avg_amount * 0.3))
    amount = round(min(amount, 15000.0), 2)

    lat = user.home_lat + random.uniform(-0.15, 0.15)
    lon = user.home_lon + random.uniform(-0.15, 0.15)
    device = user.device_id
    merchant = random.choice(user.merchant_ids)
    channel = random.choices(CHANNELS, weights=[0.35, 0.25, 0.30, 0.10], k=1)[0]
    is_international = random.random() < 0.05

    return _build_row(
        user=user,
        ts=ts,
        amount=amount,
        category=category,
        channel=channel,
        lat=lat,
        lon=lon,
        device=device,
        merchant=merchant,
        is_international=is_international,
        is_fraud=0,
        fraud_pattern="none",
    )


# ---------------------------------------------------------------------------
# Fraud transaction generators (5 patterns)
# ---------------------------------------------------------------------------


def _generate_card_testing(user: UserContext, ts: datetime) -> list[dict]:
    """Rapid micro-transactions: 10-15 txns, $1-$5 each, < 2 minutes."""
    rows = []
    count = random.randint(10, 15)
    for _ in range(count):
        t = ts + timedelta(seconds=random.uniform(0, 120))
        amt = round(random.uniform(1.0, 5.0), 2)
        rows.append(
            _build_row(
                user=user,
                ts=t,
                amount=amt,
                category=random.choice(CATEGORIES),
                channel="web",
                lat=user.home_lat + random.uniform(-0.5, 0.5),
                lon=user.home_lon + random.uniform(-0.5, 0.5),
                device=uuid4().hex[:12],
                merchant=str(uuid4())[:8],
                is_international=False,
                is_fraud=1,
                fraud_pattern="card_testing",
            )
        )
    return rows


def _generate_account_takeover(user: UserContext, ts: datetime) -> list[dict]:
    """Single high-value txn from a new device at an unusual hour."""
    t = ts.replace(hour=random.randint(2, 5))
    amt = round(user.avg_amount * random.uniform(10, 50), 2)
    return [
        _build_row(
            user=user,
            ts=t,
            amount=min(amt, 50000.0),
            category=random.choice(["electronics", "travel"]),
            channel=random.choice(["web", "mobile_app"]),
            lat=user.home_lat + random.uniform(-0.5, 0.5),
            lon=user.home_lon + random.uniform(-0.5, 0.5),
            device=uuid4().hex[:12],
            merchant=str(uuid4())[:8],
            is_international=random.random() < 0.5,
            is_fraud=1,
            fraud_pattern="account_takeover",
        )
    ]


def _generate_geo_anomaly(user: UserContext, ts: datetime) -> list[dict]:
    """Two txns from very distant locations within 30 minutes."""
    distant = random.choice(CITIES)
    return [
        _build_row(
            user=user,
            ts=ts,
            amount=round(random.uniform(10, 200), 2),
            category=random.choice(CATEGORIES),
            channel="pos_terminal",
            lat=user.home_lat,
            lon=user.home_lon,
            device=user.device_id,
            merchant=random.choice(user.merchant_ids),
            is_international=False,
            is_fraud=1,
            fraud_pattern="geo_anomaly",
        ),
        _build_row(
            user=user,
            ts=ts + timedelta(minutes=random.randint(15, 30)),
            amount=round(random.uniform(100, 3000), 2),
            category=random.choice(["electronics", "travel"]),
            channel="pos_terminal",
            lat=distant[0] + random.uniform(-0.2, 0.2),
            lon=distant[1] + random.uniform(-0.2, 0.2),
            device=uuid4().hex[:12],
            merchant=str(uuid4())[:8],
            is_international=True,
            is_fraud=1,
            fraud_pattern="geo_anomaly",
        ),
    ]


def _generate_velocity_abuse(user: UserContext, ts: datetime) -> list[dict]:
    """15-20 txns within 5 minutes with escalating amounts."""
    rows = []
    count = random.randint(15, 20)
    base_amt = random.uniform(10, 50)
    for i in range(count):
        t = ts + timedelta(seconds=random.uniform(0, 300))
        amt = round(base_amt + i * random.uniform(5, 30), 2)
        rows.append(
            _build_row(
                user=user,
                ts=t,
                amount=min(amt, 20000.0),
                category=random.choice(CATEGORIES),
                channel="mobile_app",
                lat=user.home_lat + random.uniform(-0.05, 0.05),
                lon=user.home_lon + random.uniform(-0.05, 0.05),
                device=user.device_id,
                merchant=random.choice(user.merchant_ids),
                is_international=False,
                is_fraud=1,
                fraud_pattern="velocity_abuse",
            )
        )
    return rows


def _generate_merchant_collusion(user: UserContext, ts: datetime) -> list[dict]:
    """3-5 circular transfers through a single merchant."""
    rows = []
    ring_size = random.randint(3, 5)
    colluding_merchant = str(uuid4())[:8]
    base_amt = round(random.uniform(500, 5000), 2)
    for i in range(ring_size):
        t = ts + timedelta(minutes=random.randint(2, 15) * (i + 1))
        amt = round(base_amt + random.uniform(-20, 20), 2)
        rows.append(
            _build_row(
                user=user,
                ts=t,
                amount=amt,
                category=random.choice(["electronics", "entertainment", "travel"]),
                channel="web",
                lat=user.home_lat + random.uniform(-0.05, 0.05),
                lon=user.home_lon + random.uniform(-0.05, 0.05),
                device=uuid4().hex[:12],
                merchant=colluding_merchant,
                is_international=False,
                is_fraud=1,
                fraud_pattern="merchant_collusion",
            )
        )
    return rows


FRAUD_GENERATORS = [
    _generate_card_testing,
    _generate_account_takeover,
    _generate_geo_anomaly,
    _generate_velocity_abuse,
    _generate_merchant_collusion,
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _build_row(
    *,
    user: UserContext,
    ts: datetime,
    amount: float,
    category: str,
    channel: str,
    lat: float,
    lon: float,
    device: str,
    merchant: str,
    is_international: bool,
    is_fraud: int,
    fraud_pattern: str,
) -> dict:
    """Build a single row with all raw fields and engineered features."""
    hour = ts.hour
    dow = ts.weekday()

    # Distance from last transaction
    dist_from_last = _haversine_km(user.last_lat, user.last_lon, lat, lon)
    dist_from_home = _haversine_km(user.home_lat, user.home_lon, lat, lon)

    # Time since last transaction
    if user.last_txn_time is not None:
        time_since_last = (ts - user.last_txn_time).total_seconds()
    else:
        time_since_last = 86400.0  # default 1 day

    # Running stats from history
    now_minus_1h = ts - timedelta(hours=1)
    now_minus_5m = ts - timedelta(minutes=5)
    now_minus_24h = ts - timedelta(hours=24)
    now_minus_7d = ts - timedelta(days=7)
    now_minus_30d = ts - timedelta(days=30)

    recent_1h = [t for t in user.txn_history if t["ts"] >= now_minus_1h]
    recent_5m = [t for t in user.txn_history if t["ts"] >= now_minus_5m]
    recent_24h = [t for t in user.txn_history if t["ts"] >= now_minus_24h]
    recent_7d = [t for t in user.txn_history if t["ts"] >= now_minus_7d]
    recent_30d = [t for t in user.txn_history if t["ts"] >= now_minus_30d]

    txn_count_1m = len([t for t in user.txn_history if t["ts"] >= ts - timedelta(minutes=1)])
    txn_count_5m = len(recent_5m)
    txn_count_1h = len(recent_1h)
    txn_count_24h = len(recent_24h)
    txn_sum_1h = sum(t["amount"] for t in recent_1h)
    txn_sum_24h = sum(t["amount"] for t in recent_24h)

    amounts_30d = [t["amount"] for t in recent_30d]
    avg_amount_30d = np.mean(amounts_30d) if amounts_30d else user.avg_amount
    std_amount_30d = np.std(amounts_30d) if len(amounts_30d) > 1 else user.avg_amount * 0.3
    max_amount_7d = max((t["amount"] for t in recent_7d), default=0.0)

    amount_deviation = (amount - avg_amount_30d) / max(std_amount_30d, 0.01)
    amount_to_avg_ratio = amount / max(avg_amount_30d, 0.01)

    unique_merchants_24h = len({t["merchant"] for t in recent_24h})
    unique_merchants_7d = len({t["merchant"] for t in recent_7d})
    unique_devices_24h = len({t["device"] for t in recent_24h})

    is_new_device = device != user.device_id
    is_new_merchant = merchant not in user.merchant_ids
    is_new_city = dist_from_home > 100.0

    # Device age in days (simplified)
    device_txn_count = len([t for t in user.txn_history if t["device"] == device])
    device_first = min(
        (t["ts"] for t in user.txn_history if t["device"] == device),
        default=ts,
    )
    device_age_days = (ts - device_first).days

    # Merchant fraud rate (simplified)
    merchant_risk = CATEGORY_RISK.get(category, 0.01)

    # Time features
    is_weekend = dow >= 5
    is_night = hour < 6 or hour >= 23

    # Average time between transactions
    if len(recent_24h) >= 2:
        times_sorted = sorted(t["ts"] for t in recent_24h)
        diffs = [
            (times_sorted[i + 1] - times_sorted[i]).total_seconds()
            for i in range(len(times_sorted) - 1)
        ]
        avg_time_between_txn = np.mean(diffs)
    else:
        avg_time_between_txn = 3600.0

    # Update user context
    user.last_lat = lat
    user.last_lon = lon
    user.last_txn_time = ts
    user.total_amount += amount
    user.txn_count += 1
    user.txn_history.append(
        {
            "ts": ts,
            "amount": amount,
            "merchant": merchant,
            "device": device,
            "lat": lat,
            "lon": lon,
        }
    )

    # Keep history bounded (last 200 txns)
    if len(user.txn_history) > 200:
        user.txn_history = user.txn_history[-200:]

    return {
        # Raw fields
        "txn_id": str(uuid4()),
        "user_id": user.user_id,
        "merchant_id": merchant,
        "amount": amount,
        "currency": random.choice(CURRENCIES),
        "category": category,
        "channel": channel,
        "ip_address": f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
        "device_fingerprint": device,
        "geo_lat": round(lat, 6),
        "geo_lon": round(lon, 6),
        "is_international": int(is_international),
        "timestamp": ts,
        "is_fraud": is_fraud,
        "fraud_pattern": fraud_pattern,
        # === Engineered features (30+) ===
        # Transaction features
        "f_amount": amount,
        "f_amount_log": round(math.log1p(amount), 6),
        "f_currency_usd": int(random.choice(CURRENCIES) == "USD"),
        # Velocity features
        "f_txn_count_1m": txn_count_1m,
        "f_txn_count_5m": txn_count_5m,
        "f_txn_count_1h": txn_count_1h,
        "f_txn_count_24h": txn_count_24h,
        "f_txn_sum_1h": round(txn_sum_1h, 2),
        "f_txn_sum_24h": round(txn_sum_24h, 2),
        # Amount features
        "f_amount_deviation": round(float(amount_deviation), 4),
        "f_amount_to_avg_ratio": round(float(amount_to_avg_ratio), 4),
        "f_avg_amount_30d": round(float(avg_amount_30d), 2),
        "f_max_amount_7d": round(float(max_amount_7d), 2),
        # Geo features
        "f_distance_from_last_km": round(dist_from_last, 2),
        "f_distance_from_home_km": round(dist_from_home, 2),
        "f_is_new_city": int(is_new_city),
        # Device features
        "f_is_new_device": int(is_new_device),
        "f_device_age_days": device_age_days,
        "f_device_txn_count": device_txn_count,
        # Merchant features
        "f_is_new_merchant": int(is_new_merchant),
        "f_merchant_category_risk": merchant_risk,
        "f_unique_merchants_24h": unique_merchants_24h,
        "f_unique_merchants_7d": unique_merchants_7d,
        # Time features
        "f_hour_of_day": hour,
        "f_day_of_week": dow,
        "f_is_weekend": int(is_weekend),
        "f_is_night": int(is_night),
        # Behavioral features
        "f_time_since_last_txn_s": round(time_since_last, 2),
        "f_avg_time_between_txn_s": round(float(avg_time_between_txn), 2),
        "f_unique_devices_24h": unique_devices_24h,
        # Interaction features
        "f_amount_x_velocity": round(amount * txn_count_1h, 2),
        "f_distance_x_time": round(dist_from_last / max(time_since_last, 1.0), 6),
    }


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def generate_dataset(size: int, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """Generate a full labeled dataset with engineered features.

    Args:
        size: Total number of transactions to generate.
        fraud_rate: Target fraud rate (default 2%).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with raw fields and 30+ engineered features.
    """
    random.seed(seed)
    np.random.seed(seed)

    n_fraud = int(size * fraud_rate)
    n_legit = size - n_fraud

    # Create user pool
    n_users = max(50, size // 100)
    users = [UserContext() for _ in range(n_users)]

    print(f"Generating {n_legit} legitimate transactions...")
    rows: list[dict] = []

    # Time span: last 90 days
    base_time = datetime(2026, 2, 1)
    end_time = datetime(2026, 5, 1)
    span_seconds = int((end_time - base_time).total_seconds())

    # Generate legitimate transactions (sorted by time for realistic features)
    legit_times = sorted(
        [base_time + timedelta(seconds=random.randint(0, span_seconds)) for _ in range(n_legit)]
    )

    for ts in legit_times:
        user = random.choice(users)
        rows.append(_generate_legit_txn(user, ts))

    # Generate fraud transactions distributed across patterns
    print(f"Generating {n_fraud} fraud transactions across {len(FRAUD_PATTERNS)} patterns...")
    fraud_generated = 0
    pattern_idx = 0

    while fraud_generated < n_fraud:
        user = random.choice(users)
        ts = base_time + timedelta(seconds=random.randint(0, span_seconds))
        generator = FRAUD_GENERATORS[pattern_idx % len(FRAUD_GENERATORS)]
        pattern_idx += 1

        fraud_rows = generator(user, ts)

        # Don't overshoot target
        remaining = n_fraud - fraud_generated
        fraud_rows = fraud_rows[:remaining]

        rows.extend(fraud_rows)
        fraud_generated += len(fraud_rows)

    # Shuffle and create DataFrame
    random.shuffle(rows)
    df = pd.DataFrame(rows)

    actual_fraud_rate = df["is_fraud"].mean()
    print(f"Dataset size: {len(df)} transactions")
    print(f"Fraud rate: {actual_fraud_rate:.4f} ({df['is_fraud'].sum()} fraud txns)")
    print(f"Patterns: {df[df['is_fraud'] == 1]['fraud_pattern'].value_counts().to_dict()}")
    print(f"Features: {len([c for c in df.columns if c.startswith('f_')])} engineered features")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a labeled training dataset for fraud detection",
    )
    parser.add_argument(
        "--output",
        default="data/training_data.parquet",
        help="Output Parquet file path (default: data/training_data.parquet)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100_000,
        help="Number of transactions to generate (default: 100000)",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.02,
        help="Target fraud rate (default: 0.02)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df = generate_dataset(size=args.size, fraud_rate=args.fraud_rate, seed=args.seed)

    print(f"\nSaving to {args.output}...")
    df.to_parquet(args.output, index=False, engine="pyarrow")

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved {args.output} ({file_size_mb:.1f} MB)")

    # Print feature summary
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    print(f"\n--- Feature Summary ({len(feature_cols)} features) ---")
    for col in feature_cols:
        print(
            f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}"
        )


if __name__ == "__main__":
    main()
