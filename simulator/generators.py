"""Legitimate transaction generator with realistic user profiles and spending patterns."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final
from uuid import UUID, uuid4

from shared.schemas import Transaction, TransactionType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES: Final[list[str]] = [
    "groceries",
    "electronics",
    "restaurants",
    "gas",
    "travel",
    "entertainment",
    "healthcare",
    "utilities",
]

AMOUNT_RANGES: Final[dict[str, tuple[float, float]]] = {
    "groceries": (5.0, 200.0),
    "electronics": (50.0, 5000.0),
    "restaurants": (10.0, 300.0),
    "gas": (15.0, 120.0),
    "travel": (100.0, 8000.0),
    "entertainment": (5.0, 500.0),
    "healthcare": (20.0, 2000.0),
    "utilities": (30.0, 500.0),
}

CHANNELS: Final[list[str]] = ["mobile_app", "web", "pos_terminal", "atm"]

CARD_TYPES: Final[list[str]] = ["visa", "mastercard", "amex"]

CURRENCIES: Final[list[str]] = ["USD", "EUR", "KZT", "GBP"]

# Realistic city centres with (lat, lon, name, country, timezone_offset_hours)
CITIES: Final[list[tuple[float, float, str, str, int]]] = [
    (40.7128, -74.0060, "New York", "US", -5),
    (34.0522, -118.2437, "Los Angeles", "US", -8),
    (41.8781, -87.6298, "Chicago", "US", -6),
    (29.7604, -95.3698, "Houston", "US", -6),
    (33.4484, -112.0740, "Phoenix", "US", -7),
    (51.5074, -0.1278, "London", "GB", 0),
    (48.8566, 2.3522, "Paris", "FR", 1),
    (52.5200, 13.4050, "Berlin", "DE", 1),
    (55.7558, 37.6173, "Moscow", "RU", 3),
    (43.2220, 76.8512, "Almaty", "KZ", 6),
    (51.1694, 71.4491, "Astana", "KZ", 6),
    (35.6762, 139.6503, "Tokyo", "JP", 9),
    (37.5665, 126.9780, "Seoul", "KR", 9),
    (-33.8688, 151.2093, "Sydney", "AU", 10),
    (19.4326, -99.1332, "Mexico City", "MX", -6),
    (55.6761, 12.5683, "Copenhagen", "DK", 1),
    (59.3293, 18.0686, "Stockholm", "SE", 1),
    (40.4168, -3.7038, "Madrid", "ES", 1),
    (45.4642, 9.1900, "Milan", "IT", 1),
    (25.2048, 55.2708, "Dubai", "AE", 4),
]

# Merchant name templates per category
MERCHANT_TEMPLATES: Final[dict[str, list[str]]] = {
    "groceries": [
        "FreshMart #{id}", "GreenGrocer #{id}", "SuperSave #{id}",
        "DailyBasket #{id}", "OrganicHub #{id}",
    ],
    "electronics": [
        "TechZone #{id}", "GadgetWorld #{id}", "ByteShop #{id}",
        "CircuitCity #{id}", "PixelStore #{id}",
    ],
    "restaurants": [
        "TheGoldenFork #{id}", "BluePlate #{id}", "UrbanBites #{id}",
        "SavorKitchen #{id}", "SpiceRoute #{id}",
    ],
    "gas": [
        "QuickFuel #{id}", "SpeedGas #{id}", "GreenPump #{id}",
        "RoadRunner #{id}", "FillUp #{id}",
    ],
    "travel": [
        "SkyWings #{id}", "OceanCruise #{id}", "NomadTravel #{id}",
        "JetSet #{id}", "GlobalHotel #{id}",
    ],
    "entertainment": [
        "CinePlex #{id}", "GameVault #{id}", "StreamNow #{id}",
        "LiveNation #{id}", "FunPark #{id}",
    ],
    "healthcare": [
        "MedCare #{id}", "HealthFirst #{id}", "PharmaCo #{id}",
        "WellnessHub #{id}", "LifeClinic #{id}",
    ],
    "utilities": [
        "PowerGrid #{id}", "AquaSupply #{id}", "TelcoPlus #{id}",
        "NetConnect #{id}", "GasCorp #{id}",
    ],
}

# Hourly transaction probability weights (index 0 = midnight, 23 = 11 PM)
# Peaks during business hours and early evening.
HOUR_WEIGHTS: Final[list[float]] = [
    0.02, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5
    0.03, 0.05, 0.07, 0.08, 0.09, 0.09,  # 6-11
    0.10, 0.09, 0.08, 0.07, 0.06, 0.05,  # 12-17
    0.04, 0.04, 0.03, 0.03, 0.02, 0.02,  # 18-23
]


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    """A simulated user with stable behavioural attributes."""

    user_id: UUID
    home_city: tuple[float, float, str, str, int]
    preferred_categories: list[str]
    preferred_merchants: dict[str, str]  # category -> merchant_id
    avg_spend: float
    card_hash: str
    device_id: str
    card_type: str
    currency: str
    ip_prefix: str  # first 3 octets
    last_txn_time: datetime = field(default_factory=datetime.utcnow)

    def home_lat(self) -> float:
        return self.home_city[0]

    def home_lon(self) -> float:
        return self.home_city[1]


def _generate_ip_prefix() -> str:
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}"


def _hash_card(card_num: str) -> str:
    return hashlib.sha256(card_num.encode()).hexdigest()[:16]


def build_user_pool(size: int) -> list[UserProfile]:
    """Create a pool of simulated users with realistic, stable profiles.

    Args:
        size: Number of users to generate.

    Returns:
        List of ``UserProfile`` instances.
    """
    users: list[UserProfile] = []
    for i in range(size):
        city = random.choice(CITIES)
        # Each user prefers 2-4 categories
        n_cats = random.randint(2, 4)
        preferred_cats = random.sample(CATEGORIES, k=n_cats)

        # Assign a stable merchant per category
        merchants: dict[str, str] = {}
        for cat in preferred_cats:
            tpl = random.choice(MERCHANT_TEMPLATES[cat])
            merchants[cat] = tpl.replace("#{id}", str(random.randint(1000, 9999)))

        # Currency biased by country
        if city[3] in ("US",):
            currency = "USD"
        elif city[3] in ("GB",):
            currency = "GBP"
        elif city[3] in ("KZ",):
            currency = "KZT"
        else:
            currency = random.choice(["USD", "EUR"])

        avg_spend = round(random.uniform(30.0, 500.0), 2)
        card_number = f"4{random.randint(10**14, 10**15 - 1)}"
        profile = UserProfile(
            user_id=uuid4(),
            home_city=city,
            preferred_categories=preferred_cats,
            preferred_merchants=merchants,
            avg_spend=avg_spend,
            card_hash=_hash_card(card_number),
            device_id=f"dev-{uuid4().hex[:12]}",
            card_type=random.choice(CARD_TYPES),
            currency=currency,
            ip_prefix=_generate_ip_prefix(),
        )
        users.append(profile)
    return users


# ---------------------------------------------------------------------------
# Legitimate transaction generation
# ---------------------------------------------------------------------------


def _jitter_location(lat: float, lon: float, radius_km: float = 15.0) -> tuple[float, float]:
    """Add a small random offset to coordinates (uniform in a square)."""
    # ~0.009 degrees per km at equator; good enough for simulation
    offset_deg = radius_km * 0.009
    new_lat = lat + random.uniform(-offset_deg, offset_deg)
    new_lon = lon + random.uniform(-offset_deg, offset_deg)
    new_lat = max(-90.0, min(90.0, new_lat))
    new_lon = max(-180.0, min(180.0, new_lon))
    return round(new_lat, 6), round(new_lon, 6)


def _pick_hour_weighted() -> int:
    """Select an hour of the day based on realistic transaction distribution."""
    return random.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]


def generate_legitimate_transaction(user: UserProfile) -> Transaction:
    """Generate a single realistic legitimate transaction for a user.

    The generated transaction respects the user's home city, preferred
    categories, typical merchants, and spending patterns.

    Args:
        user: The user profile to generate the transaction for.

    Returns:
        A ``Transaction`` instance.
    """
    category = random.choice(user.preferred_categories)
    merchant_id = user.preferred_merchants[category]
    lo, hi = AMOUNT_RANGES[category]

    # Amount follows a log-normal-ish distribution centred on user avg
    raw_amount = random.gauss(mu=user.avg_spend, sigma=user.avg_spend * 0.3)
    amount = max(lo, min(hi, abs(raw_amount)))
    amount = round(amount, 2)

    lat, lon = _jitter_location(user.home_lat(), user.home_lon())

    # Channel probability
    channel = random.choices(
        CHANNELS,
        weights=[0.35, 0.25, 0.30, 0.10],
        k=1,
    )[0]

    # Transaction type — mostly purchases for legitimate traffic
    txn_type = random.choices(
        [TransactionType.PURCHASE, TransactionType.TRANSFER, TransactionType.WITHDRAWAL],
        weights=[0.80, 0.12, 0.08],
        k=1,
    )[0]

    ip_address = f"{user.ip_prefix}.{random.randint(1, 254)}"

    now = datetime.utcnow()
    user.last_txn_time = now

    return Transaction(
        user_id=user.user_id,
        amount=Decimal(str(amount)),
        currency=user.currency,
        transaction_type=txn_type,
        merchant_id=merchant_id,
        merchant_category=category,
        card_number_hash=user.card_hash,
        ip_address=ip_address,
        device_id=user.device_id,
        latitude=lat,
        longitude=lon,
        timestamp=now,
        metadata={
            "channel": channel,
            "card_type": user.card_type,
            "is_fraud": False,
        },
    )
