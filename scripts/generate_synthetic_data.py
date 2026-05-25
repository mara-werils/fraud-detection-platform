#!/usr/bin/env python3
"""Comprehensive synthetic fraud data generator for the fraud detection platform.

Produces realistic, correlated transactions covering 8 distinct fraud patterns
with proper temporal patterns (business hours, seasonality, user behaviour).

Usage:
    python scripts/generate_synthetic_data.py \\
        --num-users 5000 \\
        --num-merchants 500 \\
        --num-transactions 100000 \\
        --fraud-rate 0.03 \\
        --output-format parquet \\
        --output-path data/synthetic_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required.  pip install pandas")
    sys.exit(1)

try:
    import structlog
except ImportError:
    print("ERROR: structlog is required.  pip install structlog")
    sys.exit(1)

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    print("ERROR: pydantic v2 is required.  pip install pydantic>=2.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

COUNTRIES = ["US", "GB", "DE", "FR", "CA", "AU", "JP", "SG", "AE", "BR", "NG", "RU"]
HIGH_RISK_COUNTRIES = {"NG", "RU", "BR"}

MERCHANT_CATEGORIES = [
    "groceries",
    "electronics",
    "restaurants",
    "gas",
    "travel",
    "entertainment",
    "healthcare",
    "utilities",
    "jewelry",
    "gambling",
    "crypto_exchange",
    "money_transfer",
]

HIGH_RISK_CATEGORIES = {"gambling", "crypto_exchange", "money_transfer", "jewelry"}

CHANNELS = ["mobile_app", "web", "pos_terminal", "atm"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Wei", "Aisha", "Fatima", "Raj", "Priya",
    "Ahmed", "Yuki", "Carlos", "Ana", "Ivan",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Wilson", "Taylor", "Chen", "Patel", "Kim", "Nguyen", "Lopez", "Martinez",
    "Anderson", "Thompson", "White", "Harris", "Lee", "Walker", "Robinson", "Lewis",
]

MERCHANT_NAME_PARTS = [
    "Quick", "Global", "Prime", "Metro", "Elite", "Fresh", "Smart", "Easy",
    "First", "Star", "Royal", "Urban", "Digital", "Express", "Super",
]
MERCHANT_NAME_SUFFIXES = [
    "Market", "Shop", "Store", "Hub", "Mart", "Center", "Place", "Point",
    "Exchange", "Direct", "Online", "Pay", "Cash", "Trade",
]

DEVICE_TYPES = ["iPhone", "Android", "iPad", "Windows_PC", "Mac", "Unknown"]

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class GeneratorConfig(BaseModel):
    """Configuration for the synthetic data generator."""

    num_users: int = Field(default=5_000, ge=10)
    num_merchants: int = Field(default=500, ge=5)
    num_transactions: int = Field(default=100_000, ge=100)
    fraud_rate: float = Field(default=0.03, ge=0.0, le=1.0)
    start_date: datetime = Field(
        default_factory=lambda: datetime(2024, 1, 1, tzinfo=UTC)
    )
    end_date: datetime = Field(
        default_factory=lambda: datetime(2024, 12, 31, tzinfo=UTC)
    )
    output_format: Literal["csv", "json", "parquet"] = "parquet"
    output_path: str = "data/synthetic_dataset.parquet"
    seed: int | None = None

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: datetime, info: Any) -> datetime:
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date must be after start_date")
        return v


class UserProfile(BaseModel):
    """Behavioural profile for a synthetic user."""

    user_id: str
    name: str
    email: str
    country: str
    avg_amount: float          # typical transaction amount (USD)
    usual_merchants: list[str]  # preferred merchant IDs
    usual_devices: list[str]
    usual_hours: list[int]     # preferred transaction hours (0-23)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    currency: str
    is_synthetic_identity: bool = False  # flag for synthetic identity fraud seed


class MerchantProfile(BaseModel):
    """Profile for a synthetic merchant."""

    merchant_id: str
    name: str
    category: str
    country: str
    avg_transaction: float
    fraud_rate: float          # inherent fraud rate for this merchant
    is_colluding: bool = False  # flag for merchant collusion pattern


@dataclass
class DatasetSummary:
    """Summary statistics returned after generation."""

    total_transactions: int
    fraud_transactions: int
    legitimate_transactions: int
    fraud_rate_actual: float
    fraud_pattern_counts: dict[str, int]
    user_count: int
    merchant_count: int
    date_range: tuple[datetime, datetime]
    output_path: str
    output_format: str


# ---------------------------------------------------------------------------
# Fraud Pattern Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CardTestingPattern:
    """Many small transactions in rapid succession from a single stolen card."""

    user_id: str
    merchant_ids: list[str]
    base_time: datetime
    burst_count: int = field(default_factory=lambda: random.randint(8, 25))
    amount_range: tuple[float, float] = (0.50, 5.00)

    def generate_transactions(self) -> list[dict[str, Any]]:
        txns: list[dict[str, Any]] = []
        device = f"Unknown_{uuid4().hex[:6]}"
        for i in range(self.burst_count):
            offset_seconds = i * random.randint(15, 90)
            txns.append(
                _base_txn(
                    user_id=self.user_id,
                    merchant_id=random.choice(self.merchant_ids),
                    amount=round(random.uniform(*self.amount_range), 2),
                    timestamp=self.base_time + timedelta(seconds=offset_seconds),
                    channel="web",
                    device_id=device,
                    fraud_label=True,
                    fraud_pattern="card_testing",
                )
            )
        return txns


@dataclass
class AccountTakeoverPattern:
    """Sudden behavioural shift: new device, new location, new merchants."""

    user_profile: UserProfile
    all_merchant_ids: list[str]
    base_time: datetime
    txn_count: int = field(default_factory=lambda: random.randint(3, 8))

    def generate_transactions(self) -> list[dict[str, Any]]:
        txns: list[dict[str, Any]] = []
        foreign_country = random.choice([c for c in COUNTRIES if c != self.user_profile.country])
        new_device = f"{random.choice(DEVICE_TYPES)}_{uuid4().hex[:6]}"
        # Pick merchants outside usual set
        foreign_merchants = [
            m for m in self.all_merchant_ids if m not in self.user_profile.usual_merchants
        ]
        if not foreign_merchants:
            foreign_merchants = self.all_merchant_ids
        for i in range(self.txn_count):
            amount = self.user_profile.avg_amount * random.uniform(2.0, 8.0)
            txns.append(
                _base_txn(
                    user_id=self.user_profile.user_id,
                    merchant_id=random.choice(foreign_merchants),
                    amount=round(amount, 2),
                    timestamp=self.base_time + timedelta(minutes=i * random.randint(5, 30)),
                    channel=random.choice(["web", "mobile_app"]),
                    device_id=new_device,
                    fraud_label=True,
                    fraud_pattern="account_takeover",
                    country_override=foreign_country,
                )
            )
        return txns


@dataclass
class FriendlyFraudPattern:
    """Legitimate-looking transactions later disputed by the cardholder."""

    user_profile: UserProfile
    base_time: datetime
    txn_count: int = field(default_factory=lambda: random.randint(2, 5))

    def generate_transactions(self) -> list[dict[str, Any]]:
        txns: list[dict[str, Any]] = []
        for i in range(self.txn_count):
            # Looks completely normal — same device, same merchants, normal amount
            merchant = random.choice(self.user_profile.usual_merchants) if self.user_profile.usual_merchants else uuid4().hex[:8]
            device = random.choice(self.user_profile.usual_devices) if self.user_profile.usual_devices else "unknown"
            hour = random.choice(self.user_profile.usual_hours) if self.user_profile.usual_hours else 12
            ts = self.base_time + timedelta(days=i * random.randint(1, 7))
            ts = ts.replace(hour=hour, minute=random.randint(0, 59))
            txns.append(
                _base_txn(
                    user_id=self.user_profile.user_id,
                    merchant_id=merchant,
                    amount=round(self.user_profile.avg_amount * random.uniform(0.8, 2.5), 2),
                    timestamp=ts,
                    channel=random.choice(["web", "mobile_app"]),
                    device_id=device,
                    fraud_label=True,
                    fraud_pattern="friendly_fraud",
                )
            )
        return txns


@dataclass
class SyntheticIdentityPattern:
    """Mix of real and fabricated identity elements; slow trust-building."""

    synthetic_user: UserProfile
    merchant_ids: list[str]
    start_time: datetime
    warm_up_txns: int = field(default_factory=lambda: random.randint(5, 15))
    bust_txns: int = field(default_factory=lambda: random.randint(2, 4))

    def generate_transactions(self) -> list[dict[str, Any]]:
        txns: list[dict[str, Any]] = []
        device = random.choice(self.synthetic_user.usual_devices) if self.synthetic_user.usual_devices else "unknown"
        # Warm-up: small legitimate-looking transactions
        for i in range(self.warm_up_txns):
            ts = self.start_time + timedelta(days=i * random.randint(3, 10))
            txns.append(
                _base_txn(
                    user_id=self.synthetic_user.user_id,
                    merchant_id=random.choice(merchant_ids := self.merchant_ids),
                    amount=round(random.uniform(10, 80), 2),
                    timestamp=ts,
                    channel="web",
                    device_id=device,
                    fraud_label=False,
                    fraud_pattern=None,
                )
            )
        # Bust: large fraudulent transactions
        bust_start = self.start_time + timedelta(days=self.warm_up_txns * 7)
        for j in range(self.bust_txns):
            ts = bust_start + timedelta(hours=j * random.randint(1, 6))
            txns.append(
                _base_txn(
                    user_id=self.synthetic_user.user_id,
                    merchant_id=random.choice(merchant_ids),
                    amount=round(random.uniform(500, 3000), 2),
                    timestamp=ts,
                    channel="web",
                    device_id=device,
                    fraud_label=True,
                    fraud_pattern="synthetic_identity",
                )
            )
        return txns


@dataclass
class BustOutPattern:
    """Build trust with small transactions, then large fraudulent burst."""

    user_profile: UserProfile
    merchant_ids: list[str]
    start_time: datetime
    trust_period_days: int = field(default_factory=lambda: random.randint(30, 90))
    trust_txns: int = field(default_factory=lambda: random.randint(10, 30))
    bust_txns: int = field(default_factory=lambda: random.randint(3, 7))

    def generate_transactions(self) -> list[dict[str, Any]]:
        txns: list[dict[str, Any]] = []
        device = random.choice(self.user_profile.usual_devices) if self.user_profile.usual_devices else "unknown"
        daily_step = self.trust_period_days / max(self.trust_txns, 1)
        # Trust-building phase
        for i in range(self.trust_txns):
            ts = self.start_time + timedelta(days=i * daily_step)
            txns.append(
                _base_txn(
                    user_id=self.user_profile.user_id,
                    merchant_id=random.choice(self.merchant_ids),
                    amount=round(random.uniform(20, 150), 2),
                    timestamp=ts,
                    channel=random.choice(["pos_terminal", "web"]),
                    device_id=device,
                    fraud_label=False,
                    fraud_pattern=None,
                )
            )
        # Bust phase
        bust_base = self.start_time + timedelta(days=self.trust_period_days)
        for j in range(self.bust_txns):
            ts = bust_base + timedelta(hours=j * random.randint(1, 8))
            amount = random.uniform(1000, 10000)
            txns.append(
                _base_txn(
                    user_id=self.user_profile.user_id,
                    merchant_id=random.choice(self.merchant_ids),
                    amount=round(amount, 2),
                    timestamp=ts,
                    channel="web",
                    device_id=device,
                    fraud_label=True,
                    fraud_pattern="bust_out",
                )
            )
        return txns


@dataclass
class MoneyLaunderingPattern:
    """Structuring: many transactions just under $10,000 reporting threshold (smurfing)."""

    user_ids: list[str]  # ring of users
    merchant_ids: list[str]
    base_time: datetime
    num_transactions: int = field(default_factory=lambda: random.randint(10, 30))
    threshold: float = 10_000.0

    def generate_transactions(self) -> list[dict[str, Any]]:
        txns: list[dict[str, Any]] = []
        device = f"Unknown_{uuid4().hex[:6]}"
        for i in range(self.num_transactions):
            # Amount structured just under threshold
            amount = self.threshold - random.uniform(100, 999)
            user = random.choice(self.user_ids)
            ts = self.base_time + timedelta(hours=i * random.randint(2, 12))
            txns.append(
                _base_txn(
                    user_id=user,
                    merchant_id=random.choice(self.merchant_ids),
                    amount=round(amount, 2),
                    timestamp=ts,
                    channel=random.choice(["web", "atm"]),
                    device_id=device,
                    fraud_label=True,
                    fraud_pattern="money_laundering",
                )
            )
        return txns


@dataclass
class MerchantCollusionPattern:
    """High fraud rate from a specific colluding merchant."""

    user_ids: list[str]
    merchant_id: str
    base_time: datetime
    txn_count: int = field(default_factory=lambda: random.randint(20, 60))

    def generate_transactions(self) -> list[dict[str, Any]]:
        txns: list[dict[str, Any]] = []
        for i in range(self.txn_count):
            user = random.choice(self.user_ids)
            ts = self.base_time + timedelta(hours=i * random.uniform(0.5, 4.0))
            amount = round(random.uniform(50, 800), 2)
            txns.append(
                _base_txn(
                    user_id=user,
                    merchant_id=self.merchant_id,
                    amount=amount,
                    timestamp=ts,
                    channel="web",
                    device_id=f"device_{uuid4().hex[:8]}",
                    fraud_label=True,
                    fraud_pattern="merchant_collusion",
                )
            )
        return txns


@dataclass
class CardNotPresentPattern:
    """E-commerce fraud using stolen card details; mismatched billing/shipping."""

    user_id: str
    merchant_ids: list[str]
    base_time: datetime
    txn_count: int = field(default_factory=lambda: random.randint(3, 10))

    def generate_transactions(self) -> list[dict[str, Any]]:
        txns: list[dict[str, Any]] = []
        # Attacker's device
        device = f"Unknown_{uuid4().hex[:6]}"
        country = random.choice(HIGH_RISK_COUNTRIES)
        for i in range(self.txn_count):
            ts = self.base_time + timedelta(minutes=i * random.randint(30, 180))
            amount = round(random.uniform(100, 1500), 2)
            txn = _base_txn(
                user_id=self.user_id,
                merchant_id=random.choice(self.merchant_ids),
                amount=amount,
                timestamp=ts,
                channel="web",
                device_id=device,
                fraud_label=True,
                fraud_pattern="card_not_present",
                country_override=country,
            )
            txn["billing_shipping_mismatch"] = True
            txns.append(txn)
        return txns


# ---------------------------------------------------------------------------
# Helper: base transaction dict
# ---------------------------------------------------------------------------


def _base_txn(
    *,
    user_id: str,
    merchant_id: str,
    amount: float,
    timestamp: datetime,
    channel: str,
    device_id: str,
    fraud_label: bool,
    fraud_pattern: str | None,
    country_override: str | None = None,
) -> dict[str, Any]:
    return {
        "transaction_id": uuid4().hex,
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "currency": "USD",
        "timestamp": timestamp.isoformat(),
        "channel": channel,
        "device_id": device_id,
        "country": country_override,        # filled in later if None
        "is_fraud": fraud_label,
        "fraud_pattern": fraud_pattern,
        "billing_shipping_mismatch": False,
    }


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------


def _business_hour_weight(hour: int) -> float:
    """Probability weight for a transaction occurring at `hour` (0-23)."""
    if 9 <= hour <= 12:
        return 1.5
    if 12 < hour <= 18:
        return 1.3
    if 18 < hour <= 22:
        return 1.0
    if 6 <= hour < 9:
        return 0.6
    return 0.2  # overnight


def _seasonal_weight(month: int) -> float:
    """Boost for holiday/seasonal spending."""
    if month == 12:
        return 1.5   # Christmas
    if month in (11, 6, 7):
        return 1.2   # Black Friday / summer
    if month == 2:
        return 0.85  # quiet February
    return 1.0


def _random_timestamp(start: datetime, end: datetime, rng: random.Random) -> datetime:
    """Return a random datetime between start and end with realistic hour weighting."""
    hours_total = int((end - start).total_seconds() / 3600)
    hour_weights = [_business_hour_weight(h % 24) for h in range(hours_total)]
    chosen_hour_index = rng.choices(range(hours_total), weights=hour_weights, k=1)[0]
    ts = start + timedelta(hours=chosen_hour_index, minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
    # Apply seasonal scaling (probabilistic thinning not needed; we just add noise via weight)
    return ts


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------


class SyntheticDataGenerator:
    """Generates realistic synthetic fraud datasets."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        np.random.seed(config.seed)
        self._users: list[UserProfile] = []
        self._merchants: list[MerchantProfile] = []

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------

    def generate_users(self, n: int) -> list[UserProfile]:
        """Generate `n` user profiles with correlated behavioural attributes."""
        users: list[UserProfile] = []
        for _ in range(n):
            uid = uuid4().hex[:12]
            first = self._rng.choice(FIRST_NAMES)
            last = self._rng.choice(LAST_NAMES)
            country = self._rng.choices(
                COUNTRIES,
                weights=[20, 10, 8, 8, 8, 6, 5, 5, 5, 5, 3, 3],
                k=1,
            )[0]
            # Amount follows log-normal to model realistic spending
            avg_amount = float(np.random.lognormal(mean=3.5, sigma=1.0))
            avg_amount = max(5.0, min(avg_amount, 2000.0))

            # Higher income → more diverse merchants
            _merchant_count = self._rng.randint(2, 8)
            usual_hours_count = self._rng.randint(3, 8)
            preferred_hours = sorted(
                self._rng.sample(range(7, 23), k=min(usual_hours_count, 16))
            )

            risk = self._rng.choices(
                ["LOW", "MEDIUM", "HIGH"],
                weights=[65, 25, 10],
                k=1,
            )[0]
            if country in HIGH_RISK_COUNTRIES:
                risk = self._rng.choices(["MEDIUM", "HIGH"], weights=[50, 50], k=1)[0]

            users.append(
                UserProfile(
                    user_id=uid,
                    name=f"{first} {last}",
                    email=f"{first.lower()}.{last.lower()}{self._rng.randint(1,999)}@example.com",
                    country=country,
                    avg_amount=round(avg_amount, 2),
                    usual_merchants=[],        # filled after merchant generation
                    usual_devices=[f"{self._rng.choice(DEVICE_TYPES)}_{uuid4().hex[:6]}"
                                   for _ in range(self._rng.randint(1, 3))],
                    usual_hours=preferred_hours,
                    risk_level=risk,
                    currency=self._rng.choices(CURRENCIES, weights=[50, 15, 10, 8, 7, 10], k=1)[0],
                    is_synthetic_identity=False,
                )
            )
        return users

    def generate_merchants(self, n: int) -> list[MerchantProfile]:
        """Generate `n` merchant profiles."""
        merchants: list[MerchantProfile] = []
        # Designate a small fraction as colluding merchants
        colluding_count = max(1, int(n * 0.02))
        colluding_indices = set(self._rng.sample(range(n), colluding_count))

        for i in range(n):
            mid = uuid4().hex[:12]
            part = self._rng.choice(MERCHANT_NAME_PARTS)
            suffix = self._rng.choice(MERCHANT_NAME_SUFFIXES)
            category = self._rng.choice(MERCHANT_CATEGORIES)
            country = self._rng.choices(COUNTRIES, weights=[25, 10, 8, 8, 8, 5, 5, 5, 5, 5, 3, 3], k=1)[0]
            avg_txn = float(np.random.lognormal(mean=3.8, sigma=0.9))
            avg_txn = max(5.0, min(avg_txn, 5000.0))

            # Base fraud rate; high-risk categories get higher rates
            base_fr = self._rng.uniform(0.001, 0.015)
            if category in HIGH_RISK_CATEGORIES:
                base_fr *= self._rng.uniform(2.0, 5.0)
            is_colluding = i in colluding_indices
            if is_colluding:
                base_fr = self._rng.uniform(0.15, 0.40)

            merchants.append(
                MerchantProfile(
                    merchant_id=mid,
                    name=f"{part} {suffix}",
                    category=category,
                    country=country,
                    avg_transaction=round(avg_txn, 2),
                    fraud_rate=round(base_fr, 4),
                    is_colluding=is_colluding,
                )
            )
        return merchants

    # ------------------------------------------------------------------
    # Legitimate transaction generation
    # ------------------------------------------------------------------

    def _generate_legitimate_transaction(
        self,
        user: UserProfile,
        merchant: MerchantProfile,
    ) -> dict[str, Any]:
        """Produce a single realistic legitimate transaction for a user."""
        # Amount: blend of user avg and merchant avg with log-normal noise
        base_amount = (user.avg_amount + merchant.avg_transaction) / 2.0
        amount = float(np.random.lognormal(mean=math.log(max(base_amount, 1.0)), sigma=0.4))
        amount = max(0.50, min(amount, 15_000.0))

        # Timestamp respecting user's usual hours and seasonal weights
        month_weights = [_seasonal_weight(m) for m in range(1, 13)]
        chosen_month = self._rng.choices(range(1, 13), weights=month_weights, k=1)[0]
        days_in_month = 28 if chosen_month == 2 else (30 if chosen_month in (4, 6, 9, 11) else 31)
        year = self.config.start_date.year
        hour = self._rng.choices(range(24), weights=[_business_hour_weight(h) for h in range(24)], k=1)[0]
        ts = datetime(
            year, chosen_month, self._rng.randint(1, days_in_month),
            hour, self._rng.randint(0, 59), self._rng.randint(0, 59),
            tzinfo=UTC,
        )
        # Clamp to config date range
        if ts < self.config.start_date:
            ts = self.config.start_date
        if ts > self.config.end_date:
            ts = self.config.end_date

        device = self._rng.choice(user.usual_devices) if user.usual_devices else "unknown"
        channel = self._rng.choices(CHANNELS, weights=[35, 30, 25, 10], k=1)[0]

        return _base_txn(
            user_id=user.user_id,
            merchant_id=merchant.merchant_id,
            amount=round(amount, 2),
            timestamp=ts,
            channel=channel,
            device_id=device,
            fraud_label=False,
            fraud_pattern=None,
            country_override=user.country,
        )

    # ------------------------------------------------------------------
    # Fraud pattern application
    # ------------------------------------------------------------------

    def _pick_random_time(self) -> datetime:
        return _random_timestamp(self.config.start_date, self.config.end_date, self._rng)

    def apply_fraud_patterns(
        self,
        transactions: list[dict[str, Any]],
        fraud_rate: float,
    ) -> list[dict[str, Any]]:
        """Inject fraud pattern transactions to reach the target fraud_rate."""
        target_fraud = int(len(transactions) * fraud_rate / (1 - fraud_rate))
        log.info("injecting_fraud_patterns", target_fraud_count=target_fraud)

        merchant_ids = [m.merchant_id for m in self._merchants]
        user_ids = [u.user_id for u in self._users]

        # Colluding merchants
        colluding_merchant_ids = [m.merchant_id for m in self._merchants if m.is_colluding]
        # Synthetic-identity-flagged users
        synth_users = [u for u in self._users if u.is_synthetic_identity]

        fraud_txns: list[dict[str, Any]] = []
        pattern_counts: dict[str, int] = {}

        # Weights for pattern selection (tunable)
        patterns = [
            "card_testing", "account_takeover", "friendly_fraud",
            "synthetic_identity", "bust_out", "money_laundering",
            "merchant_collusion", "card_not_present",
        ]
        weights = [15, 20, 10, 10, 10, 10, 15, 10]

        while len(fraud_txns) < target_fraud:
            pattern = self._rng.choices(patterns, weights=weights, k=1)[0]
            new_txns: list[dict[str, Any]] = []

            if pattern == "card_testing":
                p = CardTestingPattern(
                    user_id=self._rng.choice(user_ids),
                    merchant_ids=self._rng.sample(merchant_ids, k=min(5, len(merchant_ids))),
                    base_time=self._pick_random_time(),
                )
                new_txns = p.generate_transactions()

            elif pattern == "account_takeover":
                user = self._rng.choice(self._users)
                p = AccountTakeoverPattern(
                    user_profile=user,
                    all_merchant_ids=merchant_ids,
                    base_time=self._pick_random_time(),
                )
                new_txns = p.generate_transactions()

            elif pattern == "friendly_fraud":
                user = self._rng.choice(self._users)
                if not user.usual_merchants:
                    user.usual_merchants.extend(self._rng.sample(merchant_ids, k=min(3, len(merchant_ids))))
                p = FriendlyFraudPattern(
                    user_profile=user,
                    base_time=self._pick_random_time(),
                )
                new_txns = p.generate_transactions()

            elif pattern == "synthetic_identity":
                if synth_users:
                    user = self._rng.choice(synth_users)
                else:
                    user = self._rng.choice(self._users)
                p = SyntheticIdentityPattern(
                    synthetic_user=user,
                    merchant_ids=self._rng.sample(merchant_ids, k=min(4, len(merchant_ids))),
                    start_time=self.config.start_date + timedelta(days=self._rng.randint(0, 60)),
                )
                new_txns = p.generate_transactions()

            elif pattern == "bust_out":
                user = self._rng.choice(self._users)
                if not user.usual_merchants:
                    user.usual_merchants.extend(self._rng.sample(merchant_ids, k=min(3, len(merchant_ids))))
                p = BustOutPattern(
                    user_profile=user,
                    merchant_ids=merchant_ids,
                    start_time=self.config.start_date + timedelta(days=self._rng.randint(0, 90)),
                )
                new_txns = p.generate_transactions()

            elif pattern == "money_laundering":
                ring_size = self._rng.randint(3, 8)
                ring = self._rng.sample(user_ids, k=min(ring_size, len(user_ids)))
                p = MoneyLaunderingPattern(
                    user_ids=ring,
                    merchant_ids=self._rng.sample(merchant_ids, k=min(3, len(merchant_ids))),
                    base_time=self._pick_random_time(),
                )
                new_txns = p.generate_transactions()

            elif pattern == "merchant_collusion":
                if colluding_merchant_ids:
                    mid = self._rng.choice(colluding_merchant_ids)
                else:
                    mid = self._rng.choice(merchant_ids)
                victims = self._rng.sample(user_ids, k=min(15, len(user_ids)))
                p = MerchantCollusionPattern(
                    user_ids=victims,
                    merchant_id=mid,
                    base_time=self._pick_random_time(),
                )
                new_txns = p.generate_transactions()

            elif pattern == "card_not_present":
                p = CardNotPresentPattern(
                    user_id=self._rng.choice(user_ids),
                    merchant_ids=self._rng.sample(merchant_ids, k=min(5, len(merchant_ids))),
                    base_time=self._pick_random_time(),
                )
                new_txns = p.generate_transactions()

            # Fill in country field where still None
            for txn in new_txns:
                if txn.get("country") is None:
                    txn["country"] = self._rng.choice(COUNTRIES)

            fraud_txns.extend(new_txns)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + len([t for t in new_txns if t["is_fraud"]])

        log.info("fraud_patterns_applied", pattern_counts=pattern_counts, total_fraud=len(fraud_txns))
        return transactions + fraud_txns, pattern_counts

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def generate_features(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compute ML-ready features over the transaction list."""
        log.info("computing_features", count=len(transactions))

        # Sort by user + timestamp for velocity calculations
        df = pd.DataFrame(transactions)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        df.sort_values(["user_id", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # --- Temporal features ---
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Mon
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["month"] = df["timestamp"].dt.month
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

        # --- Amount features ---
        user_avg = df.groupby("user_id")["amount"].transform("mean")
        user_std = df.groupby("user_id")["amount"].transform("std").fillna(1.0)
        df["amount_z_score"] = (df["amount"] - user_avg) / user_std.clip(lower=0.01)
        df["amount_log"] = np.log1p(df["amount"])

        # --- Velocity features (1h, 24h, 7d windows) ---
        # Sort by user+time, then use DatetimeIndex-based rolling per-user group.
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        # Rolling velocity per user using explicit DatetimeIndex on sub-DataFrames.
        # We work on a slim projection to avoid the pandas groupby include_groups warning.
        slim = df[["user_id", "timestamp", "amount"]].copy()

        def _rolling_count_grp(group: pd.DataFrame, window: str) -> pd.Series:
            s = group.set_index("timestamp")["amount"]
            rolled = s.rolling(window, closed="both").count()
            rolled.index = group.index
            return rolled

        def _rolling_sum_grp(group: pd.DataFrame, window: str) -> pd.Series:
            s = group.set_index("timestamp")["amount"]
            rolled = s.rolling(window, closed="both").sum()
            rolled.index = group.index
            return rolled

        for window_name, window_str in [("1h", "1h"), ("24h", "24h"), ("7d", "7d")]:
            col = f"txn_count_{window_name}"
            df[col] = (
                slim.groupby("user_id", group_keys=False)[["timestamp", "amount"]]
                .apply(lambda g, w=window_str: _rolling_count_grp(g, w))
                .fillna(1)
                .astype(int)
            )

        for window_name, window_str in [("1h", "1h"), ("24h", "24h")]:
            col = f"amount_sum_{window_name}"
            df[col] = (
                slim.groupby("user_id", group_keys=False)[["timestamp", "amount"]]
                .apply(lambda g, w=window_str: _rolling_sum_grp(g, w))
                .fillna(df["amount"])
            )

        # --- Merchant risk features ---
        merchant_fraud_rates = {m.merchant_id: m.fraud_rate for m in self._merchants}
        merchant_categories = {m.merchant_id: m.category for m in self._merchants}
        df["merchant_fraud_rate"] = df["merchant_id"].map(merchant_fraud_rates).fillna(0.01)
        df["merchant_category"] = df["merchant_id"].map(merchant_categories).fillna("unknown")
        df["merchant_is_high_risk"] = df["merchant_category"].isin(HIGH_RISK_CATEGORIES).astype(int)

        # --- Device / channel features ---
        df["is_new_device"] = 0  # will be set below
        user_devices: dict[str, set[str]] = {}
        for idx, row in df.iterrows():
            uid = row["user_id"]
            dev = row["device_id"]
            seen = user_devices.setdefault(uid, set())
            if dev not in seen:
                df.at[idx, "is_new_device"] = 1
                seen.add(dev)

        df["is_web_channel"] = (df["channel"] == "web").astype(int)
        df["is_atm"] = (df["channel"] == "atm").astype(int)

        # --- Country mismatch ---
        user_home_countries = {u.user_id: u.country for u in self._users}
        df["user_home_country"] = df["user_id"].map(user_home_countries)
        df["country_mismatch"] = (df["country"] != df["user_home_country"]).astype(int)
        df["is_high_risk_country"] = df["country"].isin(HIGH_RISK_COUNTRIES).astype(int)

        # --- Structuring detection ---
        df["near_threshold"] = ((df["amount"] >= 9_000) & (df["amount"] < 10_000)).astype(int)

        # Encode fraud_pattern as categorical integer (0 = legit)
        pattern_map = {
            None: 0,
            "card_testing": 1,
            "account_takeover": 2,
            "friendly_fraud": 3,
            "synthetic_identity": 4,
            "bust_out": 5,
            "money_laundering": 6,
            "merchant_collusion": 7,
            "card_not_present": 8,
        }
        df["fraud_pattern_id"] = df["fraud_pattern"].map(pattern_map).fillna(0).astype(int)

        # Drop helper columns not needed in output
        df.drop(columns=["user_home_country"], inplace=True, errors="ignore")

        return df.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        transactions: list[dict[str, Any]],
        path: str,
        fmt: Literal["csv", "json", "parquet"],
    ) -> None:
        """Persist transaction list to disk in the requested format."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        df = pd.DataFrame(transactions)
        # Ensure timestamp is string for JSON/CSV; keep as-is for parquet
        if fmt == "parquet":
            try:
                import pyarrow  # noqa: F401
                df.to_parquet(path, index=False)
            except ImportError:
                log.warning("pyarrow_not_found_falling_back_to_csv")
                csv_path = path.replace(".parquet", ".csv")
                df.to_csv(csv_path, index=False)
                log.info("saved_csv", path=csv_path)
                return
        elif fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="records", lines=True)
        log.info("dataset_saved", path=path, format=fmt, rows=len(df))

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def generate_full_dataset(self, config: GeneratorConfig) -> DatasetSummary:
        """End-to-end generation: users → merchants → transactions → fraud → features → save."""
        log.info(
            "starting_generation",
            users=config.num_users,
            merchants=config.num_merchants,
            transactions=config.num_transactions,
            fraud_rate=config.fraud_rate,
        )

        # 1. Profiles
        self._users = self.generate_users(config.num_users)
        self._merchants = self.generate_merchants(config.num_merchants)

        merchant_ids = [m.merchant_id for m in self._merchants]

        # 2. Assign usual_merchants to users now that merchants exist
        for user in self._users:
            k = self._rng.randint(2, min(8, len(merchant_ids)))
            user.usual_merchants = self._rng.sample(merchant_ids, k=k)

        # 3. Flag a small fraction as synthetic-identity seeds
        synth_count = max(1, int(config.num_users * 0.01))
        for u in self._rng.sample(self._users, k=synth_count):
            object.__setattr__(u, "is_synthetic_identity", True)  # bypass frozen check
        # Pydantic v2 models are not frozen by default so direct assignment works
        for u in self._rng.sample(self._users, k=synth_count):
            u.is_synthetic_identity = True

        # 4. Generate legitimate transactions
        log.info("generating_legitimate_transactions")
        legit_txns: list[dict[str, Any]] = []
        for _ in range(config.num_transactions):
            user = self._rng.choice(self._users)
            merchant_id = self._rng.choice(user.usual_merchants)
            merchant = next((m for m in self._merchants if m.merchant_id == merchant_id), self._merchants[0])
            txn = self._generate_legitimate_transaction(user, merchant)
            txn["country"] = user.country
            legit_txns.append(txn)

        # 5. Inject fraud patterns
        all_txns, pattern_counts = self.apply_fraud_patterns(legit_txns, config.fraud_rate)

        # 6. Compute features
        enriched = self.generate_features(all_txns)

        # 7. Shuffle and save
        self._rng.shuffle(enriched)
        self.save(enriched, config.output_path, config.output_format)

        # 8. Summary
        fraud_count = sum(1 for t in enriched if t.get("is_fraud"))
        legit_count = len(enriched) - fraud_count
        actual_rate = fraud_count / len(enriched) if enriched else 0.0

        summary = DatasetSummary(
            total_transactions=len(enriched),
            fraud_transactions=fraud_count,
            legitimate_transactions=legit_count,
            fraud_rate_actual=round(actual_rate, 4),
            fraud_pattern_counts=pattern_counts,
            user_count=len(self._users),
            merchant_count=len(self._merchants),
            date_range=(config.start_date, config.end_date),
            output_path=config.output_path,
            output_format=config.output_format,
        )

        log.info(
            "generation_complete",
            total=summary.total_transactions,
            fraud=summary.fraud_transactions,
            fraud_rate=summary.fraud_rate_actual,
            output=summary.output_path,
        )
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic fraud detection training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-users", type=int, default=5_000, help="Number of user profiles")
    parser.add_argument("--num-merchants", type=int, default=500, help="Number of merchant profiles")
    parser.add_argument("--num-transactions", type=int, default=100_000, help="Number of legitimate transactions")
    parser.add_argument("--fraud-rate", type=float, default=0.03, help="Target fraud rate (0.0–1.0)")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output-format",
        choices=["csv", "json", "parquet"],
        default="parquet",
        help="Output file format",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/synthetic_dataset.parquet",
        help="Output file path",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--summary-json", type=str, default=None, help="Write JSON summary to this path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    config = GeneratorConfig(
        num_users=args.num_users,
        num_merchants=args.num_merchants,
        num_transactions=args.num_transactions,
        fraud_rate=args.fraud_rate,
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=UTC),
        end_date=datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=UTC),
        output_format=args.output_format,
        output_path=args.output_path,
        seed=args.seed,
    )

    generator = SyntheticDataGenerator(config)
    summary = generator.generate_full_dataset(config)

    print("\n=== Dataset Summary ===")
    print(f"  Total transactions : {summary.total_transactions:,}")
    print(f"  Fraud transactions : {summary.fraud_transactions:,}")
    print(f"  Legit transactions : {summary.legitimate_transactions:,}")
    print(f"  Actual fraud rate  : {summary.fraud_rate_actual:.2%}")
    print(f"  Users              : {summary.user_count:,}")
    print(f"  Merchants          : {summary.merchant_count:,}")
    print(f"  Output             : {summary.output_path} ({summary.output_format})")
    print("\n  Fraud pattern breakdown:")
    for pattern, count in sorted(summary.fraud_pattern_counts.items(), key=lambda x: -x[1]):
        print(f"    {pattern:<25} {count:>6,}")

    if args.summary_json:
        import os
        os.makedirs(os.path.dirname(args.summary_json) if os.path.dirname(args.summary_json) else ".", exist_ok=True)
        summary_dict = {
            "total_transactions": summary.total_transactions,
            "fraud_transactions": summary.fraud_transactions,
            "legitimate_transactions": summary.legitimate_transactions,
            "fraud_rate_actual": summary.fraud_rate_actual,
            "fraud_pattern_counts": summary.fraud_pattern_counts,
            "user_count": summary.user_count,
            "merchant_count": summary.merchant_count,
            "date_range": [summary.date_range[0].isoformat(), summary.date_range[1].isoformat()],
            "output_path": summary.output_path,
            "output_format": summary.output_format,
        }
        with open(args.summary_json, "w") as f:
            json.dump(summary_dict, f, indent=2)
        print(f"\n  Summary JSON written to: {args.summary_json}")


if __name__ == "__main__":
    main()
