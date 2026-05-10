"""Five realistic fraud pattern generators.

Each generator produces a list of ``Transaction`` objects whose ``metadata``
includes ``is_fraud: True`` and a ``fraud_pattern`` label for downstream
analytics and model training.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final
from uuid import uuid4

from shared.schemas import Transaction, TransactionType

from simulator.generators import (
    AMOUNT_RANGES,
    CARD_TYPES,
    CATEGORIES,
    CITIES,
    MERCHANT_TEMPLATES,
    UserProfile,
    _generate_ip_prefix,
    _hash_card,
    _jitter_location,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DISTANT_CITIES: Final[list[tuple[float, float, str]]] = [
    (35.6762, 139.6503, "Tokyo"),
    (-33.8688, 151.2093, "Sydney"),
    (55.7558, 37.6173, "Moscow"),
    (19.4326, -99.1332, "Mexico City"),
    (-23.5505, -46.6333, "Sao Paulo"),
    (28.6139, 77.2090, "New Delhi"),
    (1.3521, 103.8198, "Singapore"),
    (-1.2921, 36.8219, "Nairobi"),
]


def _random_merchant(category: str) -> str:
    tpl = random.choice(MERCHANT_TEMPLATES[category])
    return tpl.replace("#{id}", str(random.randint(1000, 9999)))


def _fraud_meta(pattern: str) -> dict[str, object]:
    return {"is_fraud": True, "fraud_pattern": pattern}


# ---------------------------------------------------------------------------
# 1. Card Testing
# ---------------------------------------------------------------------------


def generate_card_testing(user: UserProfile) -> list[Transaction]:
    """Rapid sequence of small-value transactions from the same card.

    Fraudsters validate stolen card numbers by making many tiny purchases at
    different merchants within a very short window.

    Args:
        user: The victim user profile.

    Returns:
        10-20 transactions, each $1-$5, spread over ~1 minute.
    """
    count = random.randint(10, 20)
    base_time = datetime.utcnow()
    txns: list[Transaction] = []

    for i in range(count):
        offset_seconds = random.uniform(0, 60)
        ts = base_time + timedelta(seconds=offset_seconds)
        amount = round(random.uniform(1.0, 5.0), 2)

        category = random.choice(CATEGORIES)
        merchant_id = _random_merchant(category)
        lat, lon = _jitter_location(user.home_lat(), user.home_lon(), radius_km=50.0)

        txns.append(
            Transaction(
                user_id=user.user_id,
                amount=Decimal(str(amount)),
                currency=user.currency,
                transaction_type=TransactionType.PURCHASE,
                merchant_id=merchant_id,
                merchant_category=category,
                card_number_hash=user.card_hash,
                ip_address=f"{_generate_ip_prefix()}.{random.randint(1, 254)}",
                device_id=f"dev-{uuid4().hex[:12]}",
                latitude=lat,
                longitude=lon,
                timestamp=ts,
                metadata=_fraud_meta("card_testing"),
            )
        )

    return sorted(txns, key=lambda t: t.timestamp)


# ---------------------------------------------------------------------------
# 2. Account Takeover
# ---------------------------------------------------------------------------


def generate_account_takeover(user: UserProfile) -> list[Transaction]:
    """Single high-value transaction from a new device and IP at an unusual hour.

    Simulates an attacker who has compromised credentials and immediately
    attempts a large purchase or transfer.

    Args:
        user: The victim user profile.

    Returns:
        A list containing one fraudulent transaction.
    """
    unusual_hour = random.randint(2, 5)
    now = datetime.utcnow().replace(hour=unusual_hour, minute=random.randint(0, 59))

    multiplier = random.uniform(10.0, 50.0)
    amount = round(user.avg_spend * multiplier, 2)

    new_ip = f"{_generate_ip_prefix()}.{random.randint(1, 254)}"
    new_device = f"dev-{uuid4().hex[:12]}"

    # Pick a high-value category
    category = random.choice(["electronics", "travel"])
    merchant_id = _random_merchant(category)
    lat, lon = _jitter_location(user.home_lat(), user.home_lon(), radius_km=5.0)

    txn_type = random.choice([TransactionType.PURCHASE, TransactionType.TRANSFER])

    return [
        Transaction(
            user_id=user.user_id,
            amount=Decimal(str(amount)),
            currency=user.currency,
            transaction_type=txn_type,
            merchant_id=merchant_id,
            merchant_category=category,
            card_number_hash=user.card_hash,
            ip_address=new_ip,
            device_id=new_device,
            latitude=lat,
            longitude=lon,
            timestamp=now,
            metadata={
                **_fraud_meta("account_takeover"),
                "new_device": True,
                "new_ip": True,
                "amount_multiplier": round(multiplier, 1),
            },
        )
    ]


# ---------------------------------------------------------------------------
# 3. Geo Anomaly
# ---------------------------------------------------------------------------


def generate_geo_anomaly(user: UserProfile) -> list[Transaction]:
    """Two transactions from vastly different locations within a short time.

    The first transaction is from the user's home city; the second is from
    a distant city (>1 000 km away) less than one hour later -- physically
    impossible travel.

    Args:
        user: The victim user profile.

    Returns:
        Two transactions: one legitimate-looking, one from a distant location.
    """
    base_time = datetime.utcnow()

    # Legitimate-looking transaction at home
    home_lat, home_lon = _jitter_location(user.home_lat(), user.home_lon())
    category = random.choice(user.preferred_categories)
    merchant_id = user.preferred_merchants.get(category, _random_merchant(category))

    txn_home = Transaction(
        user_id=user.user_id,
        amount=Decimal(str(round(random.uniform(10.0, 100.0), 2))),
        currency=user.currency,
        transaction_type=TransactionType.PURCHASE,
        merchant_id=merchant_id,
        merchant_category=category,
        card_number_hash=user.card_hash,
        ip_address=f"{user.ip_prefix}.{random.randint(1, 254)}",
        device_id=user.device_id,
        latitude=home_lat,
        longitude=home_lon,
        timestamp=base_time,
        metadata={
            **_fraud_meta("geo_anomaly"),
            "leg": "home",
        },
    )

    # Distant transaction 15-50 minutes later
    offset = timedelta(minutes=random.randint(15, 50))
    distant = random.choice(_DISTANT_CITIES)
    far_lat, far_lon = _jitter_location(distant[0], distant[1], radius_km=20.0)
    far_category = random.choice(CATEGORIES)
    far_merchant = _random_merchant(far_category)

    txn_far = Transaction(
        user_id=user.user_id,
        amount=Decimal(str(round(random.uniform(50.0, 2000.0), 2))),
        currency=user.currency,
        transaction_type=TransactionType.PURCHASE,
        merchant_id=far_merchant,
        merchant_category=far_category,
        card_number_hash=user.card_hash,
        ip_address=f"{_generate_ip_prefix()}.{random.randint(1, 254)}",
        device_id=f"dev-{uuid4().hex[:12]}",
        latitude=far_lat,
        longitude=far_lon,
        timestamp=base_time + offset,
        metadata={
            **_fraud_meta("geo_anomaly"),
            "leg": "distant",
            "distant_city": distant[2],
        },
    )

    return [txn_home, txn_far]


# ---------------------------------------------------------------------------
# 4. Velocity Abuse
# ---------------------------------------------------------------------------


def generate_velocity_abuse(user: UserProfile) -> list[Transaction]:
    """More than 15 transactions within 5 minutes with increasing amounts.

    Simulates a fraudster rapidly draining an account or exploiting a
    promotion by submitting many transactions in quick succession.

    Args:
        user: The victim user profile.

    Returns:
        16-25 transactions with monotonically increasing amounts.
    """
    count = random.randint(16, 25)
    base_time = datetime.utcnow()
    base_amount = random.uniform(10.0, 50.0)
    increment = random.uniform(5.0, 30.0)

    category = random.choice(CATEGORIES)
    merchant_id = _random_merchant(category)
    lat, lon = _jitter_location(user.home_lat(), user.home_lon())

    txns: list[Transaction] = []
    for i in range(count):
        offset = timedelta(seconds=random.uniform(0, 300))  # within 5 min
        amount = round(base_amount + increment * i + random.uniform(-2.0, 2.0), 2)
        amount = max(1.0, amount)

        txns.append(
            Transaction(
                user_id=user.user_id,
                amount=Decimal(str(amount)),
                currency=user.currency,
                transaction_type=TransactionType.PURCHASE,
                merchant_id=merchant_id,
                merchant_category=category,
                card_number_hash=user.card_hash,
                ip_address=f"{user.ip_prefix}.{random.randint(1, 254)}",
                device_id=user.device_id,
                latitude=lat,
                longitude=lon,
                timestamp=base_time + offset,
                metadata={
                    **_fraud_meta("velocity_abuse"),
                    "sequence_index": i,
                },
            )
        )

    return sorted(txns, key=lambda t: t.timestamp)


# ---------------------------------------------------------------------------
# 5. Merchant Collusion
# ---------------------------------------------------------------------------


def generate_merchant_collusion(user: UserProfile) -> list[Transaction]:
    """Circular transfers between 3-5 accounts through the same merchant.

    Simulates a money-laundering ring where multiple accounts send similar
    amounts through a single colluding merchant, forming a cycle.

    Args:
        user: The seed user profile (first in the ring).

    Returns:
        One transaction per participant in the ring, with similar amounts
        routed through the same merchant.
    """
    ring_size = random.randint(3, 5)
    ring_users = [user.user_id] + [uuid4() for _ in range(ring_size - 1)]

    # Shared colluding merchant
    category = random.choice(["electronics", "entertainment", "travel"])
    colluding_merchant = _random_merchant(category)

    base_amount = round(random.uniform(500.0, 5000.0), 2)
    base_time = datetime.utcnow()
    lat, lon = _jitter_location(user.home_lat(), user.home_lon(), radius_km=5.0)

    txns: list[Transaction] = []
    for i in range(ring_size):
        sender = ring_users[i]
        receiver = ring_users[(i + 1) % ring_size]

        # Slightly vary the amount so it is not perfectly identical
        amount = round(base_amount + random.uniform(-20.0, 20.0), 2)
        offset = timedelta(minutes=random.randint(2, 15) * (i + 1))

        txns.append(
            Transaction(
                user_id=sender,
                amount=Decimal(str(amount)),
                currency=user.currency,
                transaction_type=TransactionType.TRANSFER,
                merchant_id=colluding_merchant,
                merchant_category=category,
                card_number_hash=user.card_hash if sender == user.user_id else _hash_card(uuid4().hex),
                ip_address=f"{_generate_ip_prefix()}.{random.randint(1, 254)}",
                device_id=f"dev-{uuid4().hex[:12]}",
                latitude=lat,
                longitude=lon,
                timestamp=base_time + offset,
                metadata={
                    **_fraud_meta("merchant_collusion"),
                    "ring_position": i,
                    "ring_size": ring_size,
                    "receiver_id": str(receiver),
                    "colluding_merchant": colluding_merchant,
                },
            )
        )

    return txns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FRAUD_GENERATORS: Final[list[callable]] = [
    generate_card_testing,
    generate_account_takeover,
    generate_geo_anomaly,
    generate_velocity_abuse,
    generate_merchant_collusion,
]


def generate_fraud_transactions(user: UserProfile) -> list[Transaction]:
    """Pick a random fraud pattern and generate transactions for it.

    Args:
        user: The user profile to use as the seed/victim.

    Returns:
        A list of fraudulent ``Transaction`` objects.
    """
    generator = random.choice(FRAUD_GENERATORS)
    return generator(user)
