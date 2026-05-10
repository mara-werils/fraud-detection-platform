#!/usr/bin/env python3
"""Seed ClickHouse with sample transactions, user profiles, and merchant profiles.

Usage:
    python scripts/seed_data.py --clickhouse-url http://localhost:8123
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

try:
    import clickhouse_connect
except ImportError:
    print("ERROR: clickhouse-connect is required. Install with: pip install clickhouse-connect")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = [
    "groceries", "electronics", "restaurants", "gas",
    "travel", "entertainment", "healthcare", "utilities",
]

CHANNELS = ["mobile_app", "web", "pos_terminal", "atm"]
CURRENCIES = ["USD", "EUR", "KZT", "GBP"]
DECISIONS = ["ALLOW", "REVIEW", "BLOCK"]
RISK_TIERS = ["LOW", "MEDIUM", "HIGH"]

CITIES = [
    (40.7128, -74.0060, "New York"),
    (34.0522, -118.2437, "Los Angeles"),
    (51.5074, -0.1278, "London"),
    (48.8566, 2.3522, "Paris"),
    (43.2220, 76.8512, "Almaty"),
    (51.1694, 71.4491, "Astana"),
    (35.6762, 139.6503, "Tokyo"),
    (55.7558, 37.6173, "Moscow"),
    (25.2048, 55.2708, "Dubai"),
    (37.5665, 126.9780, "Seoul"),
]

MERCHANT_NAMES = [
    "FreshMart", "TechZone", "TheGoldenFork", "QuickFuel", "SkyWings",
    "CinePlex", "MedCare", "PowerGrid", "GadgetWorld", "BluePlate",
    "SpeedGas", "OceanCruise", "GameVault", "HealthFirst", "AquaSupply",
    "ByteShop", "UrbanBites", "NomadTravel", "StreamNow", "PharmaCo",
]


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def generate_transactions(
    n: int,
    user_ids: list[UUID],
    merchant_ids: list[UUID],
) -> list[dict]:
    """Generate n sample transactions with realistic distributions."""
    rows = []
    base_time = datetime.utcnow() - timedelta(days=90)

    for _ in range(n):
        user_id = random.choice(user_ids)
        merchant_id = random.choice(merchant_ids)
        created_at = base_time + timedelta(
            seconds=random.randint(0, 90 * 24 * 3600),
        )
        amount = round(random.lognormvariate(mu=4.0, sigma=1.2), 2)
        amount = min(amount, 50000.0)  # cap

        is_fraud = 1 if random.random() < 0.02 else 0
        fraud_score = (
            round(random.uniform(0.7, 0.99), 4) if is_fraud
            else round(random.uniform(0.01, 0.3), 4)
        )
        xgb_score = round(fraud_score + random.uniform(-0.05, 0.05), 4)
        xgb_score = max(0.0, min(1.0, xgb_score))
        gnn_score = round(fraud_score + random.uniform(-0.08, 0.08), 4)
        gnn_score = max(0.0, min(1.0, gnn_score))

        if fraud_score > 0.8:
            decision = "BLOCK"
        elif fraud_score > 0.5:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        city = random.choice(CITIES)
        geo_lat = city[0] + random.uniform(-0.1, 0.1)
        geo_lon = city[1] + random.uniform(-0.1, 0.1)

        ip_parts = [str(random.randint(1, 254)) for _ in range(4)]

        features = {
            "txn_count_1h": float(random.randint(0, 20)),
            "txn_count_24h": float(random.randint(0, 50)),
            "amount_zscore": round(random.gauss(0, 2), 4),
            "distance_from_last_km": round(random.expovariate(0.01), 2),
            "time_since_last_txn_s": float(random.randint(60, 86400)),
            "unique_merchants_24h": float(random.randint(1, 15)),
        }

        rows.append({
            "txn_id": uuid4(),
            "user_id": user_id,
            "merchant_id": merchant_id,
            "amount": Decimal(str(amount)),
            "currency": random.choice(CURRENCIES),
            "category": random.choice(CATEGORIES),
            "channel": random.choice(CHANNELS),
            "ip_address": ".".join(ip_parts),
            "device_fingerprint": uuid4().hex[:16],
            "geo_lat": geo_lat,
            "geo_lon": geo_lon,
            "fraud_score": fraud_score,
            "xgboost_score": xgb_score,
            "gnn_score": gnn_score,
            "decision": decision,
            "explanation": f"Score={fraud_score:.2f}" if is_fraud else "",
            "is_fraud": is_fraud,
            "features": features,
            "created_at": created_at,
            "scored_at": created_at + timedelta(milliseconds=random.randint(20, 200)),
        })

    return rows


def generate_user_profiles(user_ids: list[UUID]) -> list[dict]:
    """Generate user profile rows."""
    rows = []
    now = datetime.utcnow()

    for uid in user_ids:
        total_txn = random.randint(10, 5000)
        fraud_count = random.randint(0, max(1, total_txn // 50))
        avg_amount = round(random.uniform(20.0, 500.0), 2)
        city = random.choice(CITIES)

        rows.append({
            "user_id": uid,
            "total_txn_count": total_txn,
            "total_txn_amount": Decimal(str(round(avg_amount * total_txn, 2))),
            "avg_txn_amount": Decimal(str(avg_amount)),
            "fraud_count": fraud_count,
            "fraud_rate": round(fraud_count / max(total_txn, 1), 6),
            "first_txn_date": (now - timedelta(days=random.randint(180, 730))).date(),
            "last_txn_date": (now - timedelta(days=random.randint(0, 30))).date(),
            "unique_merchants": random.randint(3, 100),
            "unique_devices": random.randint(1, 5),
            "primary_geo": f"{city[2]}",
            "risk_tier": random.choices(RISK_TIERS, weights=[0.7, 0.2, 0.1], k=1)[0],
            "updated_at": now,
        })

    return rows


def generate_merchant_profiles(merchant_ids: list[UUID]) -> list[dict]:
    """Generate merchant profile rows."""
    rows = []
    now = datetime.utcnow()

    for i, mid in enumerate(merchant_ids):
        name = MERCHANT_NAMES[i % len(MERCHANT_NAMES)] + f" #{random.randint(1000, 9999)}"
        total_txn = random.randint(100, 50000)
        fraud_count = random.randint(0, max(1, total_txn // 100))

        rows.append({
            "merchant_id": mid,
            "name": name,
            "category": random.choice(CATEGORIES),
            "total_txn": total_txn,
            "fraud_count": fraud_count,
            "fraud_rate": round(fraud_count / max(total_txn, 1), 6),
            "avg_txn_amount": Decimal(str(round(random.uniform(15.0, 800.0), 2))),
            "risk_tier": random.choices(RISK_TIERS, weights=[0.6, 0.25, 0.15], k=1)[0],
            "updated_at": now,
        })

    return rows


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def insert_transactions(client: clickhouse_connect.driver.Client, rows: list[dict]) -> None:
    """Insert transactions into ClickHouse."""
    columns = [
        "txn_id", "user_id", "merchant_id", "amount", "currency", "category",
        "channel", "ip_address", "device_fingerprint", "geo_lat", "geo_lon",
        "fraud_score", "xgboost_score", "gnn_score", "decision", "explanation",
        "is_fraud", "features", "created_at", "scored_at",
    ]
    data = [[row[c] for c in columns] for row in rows]
    client.insert("transactions", data, column_names=columns)


def insert_user_profiles(client: clickhouse_connect.driver.Client, rows: list[dict]) -> None:
    """Insert user profiles into ClickHouse."""
    columns = [
        "user_id", "total_txn_count", "total_txn_amount", "avg_txn_amount",
        "fraud_count", "fraud_rate", "first_txn_date", "last_txn_date",
        "unique_merchants", "unique_devices", "primary_geo", "risk_tier",
        "updated_at",
    ]
    data = [[row[c] for c in columns] for row in rows]
    client.insert("user_profiles", data, column_names=columns)


def insert_merchant_profiles(client: clickhouse_connect.driver.Client, rows: list[dict]) -> None:
    """Insert merchant profiles into ClickHouse."""
    columns = [
        "merchant_id", "name", "category", "total_txn", "fraud_count",
        "fraud_rate", "avg_txn_amount", "risk_tier", "updated_at",
    ]
    data = [[row[c] for c in columns] for row in rows]
    client.insert("merchant_profiles", data, column_names=columns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ClickHouse with sample data")
    parser.add_argument(
        "--clickhouse-url",
        default="http://localhost:8123",
        help="ClickHouse HTTP URL (default: http://localhost:8123)",
    )
    parser.add_argument("--transactions", type=int, default=1000, help="Number of transactions")
    parser.add_argument("--users", type=int, default=50, help="Number of user profiles")
    parser.add_argument("--merchants", type=int, default=100, help="Number of merchant profiles")
    args = parser.parse_args()

    # Parse host/port from URL
    url = args.clickhouse_url.replace("http://", "").replace("https://", "")
    host, _, port = url.partition(":")
    port_int = int(port) if port else 8123

    print(f"Connecting to ClickHouse at {host}:{port_int}...")
    client = clickhouse_connect.get_client(host=host, port=port_int)

    # Generate IDs
    user_ids = [uuid4() for _ in range(args.users)]
    merchant_ids = [uuid4() for _ in range(args.merchants)]

    # Generate and insert transactions
    print(f"Generating {args.transactions} transactions...")
    txn_rows = generate_transactions(args.transactions, user_ids, merchant_ids)
    print("Inserting transactions...")
    insert_transactions(client, txn_rows)
    print(f"  Inserted {len(txn_rows)} transactions.")

    # Generate and insert user profiles
    print(f"Generating {args.users} user profiles...")
    user_rows = generate_user_profiles(user_ids)
    print("Inserting user profiles...")
    insert_user_profiles(client, user_rows)
    print(f"  Inserted {len(user_rows)} user profiles.")

    # Generate and insert merchant profiles
    print(f"Generating {args.merchants} merchant profiles...")
    merchant_rows = generate_merchant_profiles(merchant_ids)
    print("Inserting merchant profiles...")
    insert_merchant_profiles(client, merchant_rows)
    print(f"  Inserted {len(merchant_rows)} merchant profiles.")

    # Summary
    txn_count = client.command("SELECT count() FROM transactions")
    fraud_count = client.command("SELECT countIf(is_fraud = 1) FROM transactions")
    user_count = client.command("SELECT count() FROM user_profiles")
    merch_count = client.command("SELECT count() FROM merchant_profiles")

    print("\n--- Seed Summary ---")
    print(f"  Transactions: {txn_count} (fraud: {fraud_count})")
    print(f"  User profiles: {user_count}")
    print(f"  Merchant profiles: {merch_count}")
    print("Done.")


if __name__ == "__main__":
    main()
