"""Realistic fraud simulation scenarios for testing and demos.

Each scenario is a class with a ``generate()`` method that returns a list of
``Transaction`` objects whose ``metadata`` includes ``is_fraud: True`` and a
``fraud_scenario`` label.  These are higher-level *multi-step* attack
simulations compared to the single-pattern generators in ``fraud_patterns.py``.
"""

from __future__ import annotations

import abc
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final
from uuid import uuid4

from shared.schemas import Transaction, TransactionType
from simulator.generators import (
    AMOUNT_RANGES,
    CATEGORIES,
    CHANNELS,
    MERCHANT_TEMPLATES,
    UserProfile,
    _generate_ip_prefix,
    _hash_card,
    _jitter_location,
)

__all__ = [
    "FraudScenario",
    "CardTestingAttack",
    "AccountTakeoverSimulation",
    "FriendlyFraudSimulation",
    "SyntheticIdentityFraud",
    "MoneyMuleChain",
    "RefundAbuseSimulation",
    "MerchantCollusionSimulation",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIRST_NAMES: Final[list[str]] = [
    "James", "Maria", "Alex", "Chen", "Olga",
    "Ahmed", "Yuki", "Carlos", "Emma", "Raj",
]

_LAST_NAMES: Final[list[str]] = [
    "Smith", "Garcia", "Wang", "Johnson", "Kim",
    "Patel", "Mueller", "Santos", "Andersson", "Okafor",
]


def _scenario_meta(scenario: str, **extra: object) -> dict[str, object]:
    return {"is_fraud": True, "fraud_scenario": scenario, **extra}


def _random_merchant(category: str) -> str:
    tpl = random.choice(MERCHANT_TEMPLATES[category])
    return tpl.replace("#{id}", str(random.randint(1000, 9999)))


def _random_ip() -> str:
    return f"{_generate_ip_prefix()}.{random.randint(1, 254)}"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class FraudScenario(abc.ABC):
    """Abstract base class for fraud simulation scenarios."""

    @abc.abstractmethod
    def generate(self) -> list[Transaction]:
        """Generate a list of transactions that represent this fraud scenario."""


# ---------------------------------------------------------------------------
# 1. Card Testing Attack
# ---------------------------------------------------------------------------


class CardTestingAttack(FraudScenario):
    """Many small authorization probes followed by one large purchase.

    Fraudsters validate stolen card numbers by making a burst of tiny charges
    at various merchants.  Once the card is confirmed active they make a
    single high-value purchase.

    Parameters
    ----------
    user : UserProfile
        The victim cardholder profile.
    probe_count : int | None
        Number of small probes (default: random 8-20).
    """

    def __init__(self, user: UserProfile, *, probe_count: int | None = None) -> None:
        self.user = user
        self.probe_count = probe_count or random.randint(8, 20)

    def generate(self) -> list[Transaction]:
        txns: list[Transaction] = []
        base_time = datetime.now(UTC)

        # Phase 1 -- rapid small probes over ~2 minutes
        for i in range(self.probe_count):
            ts = base_time + timedelta(seconds=random.uniform(0, 120))
            amount = round(random.uniform(0.50, 4.99), 2)
            category = random.choice(CATEGORIES)

            txns.append(
                Transaction(
                    user_id=self.user.user_id,
                    amount=Decimal(str(amount)),
                    currency=self.user.currency,
                    transaction_type=TransactionType.PURCHASE,
                    merchant_id=_random_merchant(category),
                    merchant_category=category,
                    card_number_hash=self.user.card_hash,
                    ip_address=_random_ip(),
                    device_id=f"dev-{uuid4().hex[:12]}",
                    latitude=_jitter_location(self.user.home_lat(), self.user.home_lon(), 80.0)[0],
                    longitude=_jitter_location(self.user.home_lat(), self.user.home_lon(), 80.0)[1],
                    timestamp=ts,
                    metadata=_scenario_meta("card_testing_attack", phase="probe", probe_index=i),
                )
            )

        # Phase 2 -- single high-value cashout
        cashout_delay = timedelta(minutes=random.randint(3, 15))
        cashout_amount = round(random.uniform(1500.0, 9999.99), 2)
        cashout_cat = random.choice(["electronics", "travel"])

        txns.append(
            Transaction(
                user_id=self.user.user_id,
                amount=Decimal(str(cashout_amount)),
                currency=self.user.currency,
                transaction_type=TransactionType.PURCHASE,
                merchant_id=_random_merchant(cashout_cat),
                merchant_category=cashout_cat,
                card_number_hash=self.user.card_hash,
                ip_address=_random_ip(),
                device_id=f"dev-{uuid4().hex[:12]}",
                latitude=_jitter_location(self.user.home_lat(), self.user.home_lon(), 50.0)[0],
                longitude=_jitter_location(self.user.home_lat(), self.user.home_lon(), 50.0)[1],
                timestamp=base_time + cashout_delay,
                metadata=_scenario_meta("card_testing_attack", phase="cashout"),
            )
        )

        return sorted(txns, key=lambda t: t.timestamp)


# ---------------------------------------------------------------------------
# 2. Account Takeover Simulation
# ---------------------------------------------------------------------------


class AccountTakeoverSimulation(FraudScenario):
    """Login from a new device, change settings, then transfer funds.

    Simulates a full account takeover lifecycle: the attacker logs in from
    an unfamiliar device/IP, changes security settings (modelled as a small
    metadata-only "settings" transaction), then initiates a high-value
    transfer to an external account.

    Parameters
    ----------
    user : UserProfile
        The victim cardholder profile.
    """

    def __init__(self, user: UserProfile) -> None:
        self.user = user

    def generate(self) -> list[Transaction]:
        txns: list[Transaction] = []
        base_time = datetime.now(UTC).replace(
            hour=random.randint(1, 4), minute=random.randint(0, 59)
        )
        attacker_ip = _random_ip()
        attacker_device = f"dev-{uuid4().hex[:12]}"

        # Step 1 -- login probe (small auth check)
        txns.append(
            Transaction(
                user_id=self.user.user_id,
                amount=Decimal("0.00"),
                currency=self.user.currency,
                transaction_type=TransactionType.PURCHASE,
                merchant_id="internal_auth_check",
                merchant_category="authentication",
                card_number_hash=self.user.card_hash,
                ip_address=attacker_ip,
                device_id=attacker_device,
                latitude=_jitter_location(self.user.home_lat(), self.user.home_lon(), 200.0)[0],
                longitude=_jitter_location(self.user.home_lat(), self.user.home_lon(), 200.0)[1],
                timestamp=base_time,
                metadata=_scenario_meta(
                    "account_takeover",
                    step="login",
                    new_device=True,
                    new_ip=True,
                ),
            )
        )

        # Step 2 -- change settings (password / email)
        txns.append(
            Transaction(
                user_id=self.user.user_id,
                amount=Decimal("0.00"),
                currency=self.user.currency,
                transaction_type=TransactionType.PURCHASE,
                merchant_id="internal_settings_change",
                merchant_category="account_management",
                card_number_hash=self.user.card_hash,
                ip_address=attacker_ip,
                device_id=attacker_device,
                latitude=_jitter_location(self.user.home_lat(), self.user.home_lon(), 200.0)[0],
                longitude=_jitter_location(self.user.home_lat(), self.user.home_lon(), 200.0)[1],
                timestamp=base_time + timedelta(minutes=random.randint(2, 8)),
                metadata=_scenario_meta(
                    "account_takeover",
                    step="settings_change",
                    changed_fields=["email", "password"],
                ),
            )
        )

        # Step 3 -- large transfer out
        transfer_amount = round(self.user.avg_spend * random.uniform(20.0, 80.0), 2)
        txns.append(
            Transaction(
                user_id=self.user.user_id,
                amount=Decimal(str(transfer_amount)),
                currency=self.user.currency,
                transaction_type=TransactionType.TRANSFER,
                merchant_id=f"external_acct_{uuid4().hex[:8]}",
                merchant_category="transfer",
                card_number_hash=self.user.card_hash,
                ip_address=attacker_ip,
                device_id=attacker_device,
                latitude=_jitter_location(self.user.home_lat(), self.user.home_lon(), 200.0)[0],
                longitude=_jitter_location(self.user.home_lat(), self.user.home_lon(), 200.0)[1],
                timestamp=base_time + timedelta(minutes=random.randint(10, 25)),
                metadata=_scenario_meta(
                    "account_takeover",
                    step="transfer",
                    amount_multiplier=round(transfer_amount / self.user.avg_spend, 1),
                ),
            )
        )

        return sorted(txns, key=lambda t: t.timestamp)


# ---------------------------------------------------------------------------
# 3. Friendly Fraud Simulation
# ---------------------------------------------------------------------------


class FriendlyFraudSimulation(FraudScenario):
    """Legitimate-looking charge followed by a chargeback / dispute.

    The cardholder makes a real purchase from their usual device and location,
    then files a fraudulent chargeback claiming the transaction was
    unauthorized.

    Parameters
    ----------
    user : UserProfile
        The cardholder profile.
    """

    def __init__(self, user: UserProfile) -> None:
        self.user = user

    def generate(self) -> list[Transaction]:
        txns: list[Transaction] = []
        purchase_time = datetime.now(UTC)

        category = random.choice(self.user.preferred_categories)
        merchant_id = self.user.preferred_merchants.get(category, _random_merchant(category))
        lo, hi = AMOUNT_RANGES[category]
        amount = round(random.uniform(max(lo, 80.0), min(hi, 2000.0)), 2)
        lat, lon = _jitter_location(self.user.home_lat(), self.user.home_lon())

        # Original purchase -- looks completely legitimate
        txns.append(
            Transaction(
                user_id=self.user.user_id,
                amount=Decimal(str(amount)),
                currency=self.user.currency,
                transaction_type=TransactionType.PURCHASE,
                merchant_id=merchant_id,
                merchant_category=category,
                card_number_hash=self.user.card_hash,
                ip_address=f"{self.user.ip_prefix}.{random.randint(1, 254)}",
                device_id=self.user.device_id,
                latitude=lat,
                longitude=lon,
                timestamp=purchase_time,
                metadata=_scenario_meta(
                    "friendly_fraud",
                    step="purchase",
                    channel=random.choice(CHANNELS),
                ),
            )
        )

        # Chargeback filed 5-30 days later
        chargeback_delay = timedelta(days=random.randint(5, 30))
        txns.append(
            Transaction(
                user_id=self.user.user_id,
                amount=Decimal(str(amount)),
                currency=self.user.currency,
                transaction_type=TransactionType.REFUND,
                merchant_id=merchant_id,
                merchant_category=category,
                card_number_hash=self.user.card_hash,
                ip_address=f"{self.user.ip_prefix}.{random.randint(1, 254)}",
                device_id=self.user.device_id,
                latitude=lat,
                longitude=lon,
                timestamp=purchase_time + chargeback_delay,
                metadata=_scenario_meta(
                    "friendly_fraud",
                    step="chargeback",
                    dispute_reason="unauthorized_transaction",
                    days_after_purchase=chargeback_delay.days,
                ),
            )
        )

        return txns


# ---------------------------------------------------------------------------
# 4. Synthetic Identity Fraud
# ---------------------------------------------------------------------------


class SyntheticIdentityFraud(FraudScenario):
    """New account with mixed real/fake data that builds credit then busts out.

    The fraudster creates an identity by mixing a real SSN fragment with fake
    personal details, opens an account, makes a few small "credit-building"
    purchases over weeks, and then makes a large bust-out purchase.

    Parameters
    ----------
    buildup_count : int | None
        Number of credit-building transactions (default: random 4-8).
    """

    def __init__(self, *, buildup_count: int | None = None) -> None:
        self.buildup_count = buildup_count or random.randint(4, 8)

    def generate(self) -> list[Transaction]:
        txns: list[Transaction] = []

        # Synthetic identity
        synth_user_id = uuid4()
        synth_card = _hash_card(uuid4().hex)
        synth_device = f"dev-{uuid4().hex[:12]}"
        synth_ip_prefix = _generate_ip_prefix()
        currency = random.choice(["USD", "EUR"])
        fake_name = f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"

        # Pick a home location
        from simulator.generators import CITIES
        city = random.choice(CITIES)
        lat_base, lon_base = city[0], city[1]

        base_time = datetime.now(UTC) - timedelta(days=random.randint(60, 120))

        # Phase 1 -- credit building: small legitimate-looking purchases
        for i in range(self.buildup_count):
            day_offset = timedelta(days=random.randint(3, 14) * (i + 1))
            amount = round(random.uniform(15.0, 80.0), 2)
            category = random.choice(["groceries", "gas", "restaurants", "utilities"])
            lat, lon = _jitter_location(lat_base, lon_base)

            txns.append(
                Transaction(
                    user_id=synth_user_id,
                    amount=Decimal(str(amount)),
                    currency=currency,
                    transaction_type=TransactionType.PURCHASE,
                    merchant_id=_random_merchant(category),
                    merchant_category=category,
                    card_number_hash=synth_card,
                    ip_address=f"{synth_ip_prefix}.{random.randint(1, 254)}",
                    device_id=synth_device,
                    latitude=lat,
                    longitude=lon,
                    timestamp=base_time + day_offset,
                    metadata=_scenario_meta(
                        "synthetic_identity",
                        step="credit_building",
                        buildup_index=i,
                        synthetic_name=fake_name,
                    ),
                )
            )

        # Phase 2 -- bust out: large high-value purchase
        bustout_delay = timedelta(days=random.randint(3, 7)) + timedelta(
            days=random.randint(3, 14) * self.buildup_count
        )
        bustout_amount = round(random.uniform(3000.0, 15000.0), 2)
        bustout_cat = random.choice(["electronics", "travel"])
        lat, lon = _jitter_location(lat_base, lon_base)

        txns.append(
            Transaction(
                user_id=synth_user_id,
                amount=Decimal(str(bustout_amount)),
                currency=currency,
                transaction_type=TransactionType.PURCHASE,
                merchant_id=_random_merchant(bustout_cat),
                merchant_category=bustout_cat,
                card_number_hash=synth_card,
                ip_address=f"{synth_ip_prefix}.{random.randint(1, 254)}",
                device_id=synth_device,
                latitude=lat,
                longitude=lon,
                timestamp=base_time + bustout_delay,
                metadata=_scenario_meta(
                    "synthetic_identity",
                    step="bustout",
                    synthetic_name=fake_name,
                ),
            )
        )

        return sorted(txns, key=lambda t: t.timestamp)


# ---------------------------------------------------------------------------
# 5. Money Mule Chain Simulation
# ---------------------------------------------------------------------------


class MoneyMuleChain(FraudScenario):
    """A -> B -> C -> D rapid transfers to obscure fund origin.

    Simulates a layered money laundering chain where stolen funds are rapidly
    bounced through 3-5 mule accounts, each skimming a small percentage,
    before being withdrawn or converted.

    Parameters
    ----------
    chain_length : int | None
        Number of hops (default: random 3-5).
    initial_amount : float | None
        Starting amount (default: random 2000-20000).
    """

    def __init__(
        self,
        *,
        chain_length: int | None = None,
        initial_amount: float | None = None,
    ) -> None:
        self.chain_length = chain_length or random.randint(3, 5)
        self.initial_amount = initial_amount or round(random.uniform(2000.0, 20000.0), 2)

    def generate(self) -> list[Transaction]:
        txns: list[Transaction] = []
        base_time = datetime.now(UTC)

        mule_ids = [uuid4() for _ in range(self.chain_length + 1)]
        amount = self.initial_amount

        from simulator.generators import CITIES
        city = random.choice(CITIES)

        cumulative_minutes = 0
        for hop in range(self.chain_length):
            sender = mule_ids[hop]
            receiver = mule_ids[hop + 1]

            # Each hop skims 2-8%
            skim = round(amount * random.uniform(0.02, 0.08), 2)
            transfer_amount = round(amount - skim, 2)

            cumulative_minutes += random.randint(3, 20)
            delay = timedelta(minutes=cumulative_minutes)
            lat, lon = _jitter_location(city[0], city[1], radius_km=30.0)

            txns.append(
                Transaction(
                    user_id=sender,
                    amount=Decimal(str(transfer_amount)),
                    currency="USD",
                    transaction_type=TransactionType.TRANSFER,
                    merchant_id=f"p2p_transfer_{uuid4().hex[:8]}",
                    merchant_category="transfer",
                    card_number_hash=_hash_card(uuid4().hex),
                    ip_address=_random_ip(),
                    device_id=f"dev-{uuid4().hex[:12]}",
                    latitude=lat,
                    longitude=lon,
                    timestamp=base_time + delay,
                    metadata=_scenario_meta(
                        "money_mule_chain",
                        hop=hop,
                        chain_length=self.chain_length,
                        sender_id=str(sender),
                        receiver_id=str(receiver),
                        skim_amount=skim,
                    ),
                )
            )

            amount = transfer_amount

        # Final withdrawal at end of chain
        cumulative_minutes += random.randint(3, 20)
        final_delay = timedelta(minutes=cumulative_minutes)
        lat, lon = _jitter_location(city[0], city[1], radius_km=30.0)
        txns.append(
            Transaction(
                user_id=mule_ids[-1],
                amount=Decimal(str(amount)),
                currency="USD",
                transaction_type=TransactionType.WITHDRAWAL,
                merchant_id=f"atm_{uuid4().hex[:8]}",
                merchant_category="atm_withdrawal",
                card_number_hash=_hash_card(uuid4().hex),
                ip_address=_random_ip(),
                device_id=f"dev-{uuid4().hex[:12]}",
                latitude=lat,
                longitude=lon,
                timestamp=base_time + final_delay,
                metadata=_scenario_meta(
                    "money_mule_chain",
                    step="final_withdrawal",
                    chain_length=self.chain_length,
                ),
            )
        )

        return sorted(txns, key=lambda t: t.timestamp)


# ---------------------------------------------------------------------------
# 6. Refund Abuse Simulation
# ---------------------------------------------------------------------------


class RefundAbuseSimulation(FraudScenario):
    """Repeated purchase-then-refund pattern to exploit refund policies.

    The fraudster buys items, receives them, claims they were never delivered
    or arrived damaged, and gets a refund while keeping the goods.  This is
    repeated across multiple merchants to stay below individual thresholds.

    Parameters
    ----------
    user : UserProfile
        The abuser's profile.
    cycle_count : int | None
        Number of buy-refund cycles (default: random 3-6).
    """

    def __init__(self, user: UserProfile, *, cycle_count: int | None = None) -> None:
        self.user = user
        self.cycle_count = cycle_count or random.randint(3, 6)

    def generate(self) -> list[Transaction]:
        txns: list[Transaction] = []
        base_time = datetime.now(UTC) - timedelta(days=self.cycle_count * 10)

        for cycle in range(self.cycle_count):
            category = random.choice(["electronics", "entertainment", "healthcare"])
            merchant_id = _random_merchant(category)
            lo, hi = AMOUNT_RANGES[category]
            amount = round(random.uniform(max(lo, 50.0), min(hi, 800.0)), 2)
            lat, lon = _jitter_location(self.user.home_lat(), self.user.home_lon())

            purchase_offset = timedelta(days=random.randint(5, 12) * (cycle + 1))

            # Purchase
            txns.append(
                Transaction(
                    user_id=self.user.user_id,
                    amount=Decimal(str(amount)),
                    currency=self.user.currency,
                    transaction_type=TransactionType.PURCHASE,
                    merchant_id=merchant_id,
                    merchant_category=category,
                    card_number_hash=self.user.card_hash,
                    ip_address=f"{self.user.ip_prefix}.{random.randint(1, 254)}",
                    device_id=self.user.device_id,
                    latitude=lat,
                    longitude=lon,
                    timestamp=base_time + purchase_offset,
                    metadata=_scenario_meta(
                        "refund_abuse",
                        step="purchase",
                        cycle=cycle,
                    ),
                )
            )

            # Refund 3-14 days later
            refund_delay = timedelta(days=random.randint(3, 14))
            refund_reason = random.choice([
                "item_not_received",
                "item_damaged",
                "wrong_item",
                "quality_issue",
            ])

            txns.append(
                Transaction(
                    user_id=self.user.user_id,
                    amount=Decimal(str(amount)),
                    currency=self.user.currency,
                    transaction_type=TransactionType.REFUND,
                    merchant_id=merchant_id,
                    merchant_category=category,
                    card_number_hash=self.user.card_hash,
                    ip_address=f"{self.user.ip_prefix}.{random.randint(1, 254)}",
                    device_id=self.user.device_id,
                    latitude=lat,
                    longitude=lon,
                    timestamp=base_time + purchase_offset + refund_delay,
                    metadata=_scenario_meta(
                        "refund_abuse",
                        step="refund",
                        cycle=cycle,
                        refund_reason=refund_reason,
                    ),
                )
            )

        return sorted(txns, key=lambda t: t.timestamp)


# ---------------------------------------------------------------------------
# 7. Merchant Collusion Simulation
# ---------------------------------------------------------------------------


class MerchantCollusionSimulation(FraudScenario):
    """Circular fund movement through a colluding merchant network.

    Multiple accounts route funds through a shared set of colluding merchants
    in a round-robin pattern.  Amounts are similar but not identical to evade
    exact-match detection, and the transactions complete a full circle.

    Parameters
    ----------
    user : UserProfile
        Seed user for the ring.
    ring_size : int | None
        Number of participants in the ring (default: random 3-6).
    """

    def __init__(self, user: UserProfile, *, ring_size: int | None = None) -> None:
        self.user = user
        self.ring_size = ring_size or random.randint(3, 6)

    def generate(self) -> list[Transaction]:
        txns: list[Transaction] = []
        base_time = datetime.now(UTC)

        ring_users = [self.user.user_id] + [uuid4() for _ in range(self.ring_size - 1)]

        # 2-3 colluding merchants
        colluding_categories = random.sample(
            ["electronics", "entertainment", "travel", "restaurants"], k=min(3, self.ring_size)
        )
        colluding_merchants = [_random_merchant(cat) for cat in colluding_categories]

        base_amount = round(random.uniform(800.0, 8000.0), 2)
        lat, lon = _jitter_location(self.user.home_lat(), self.user.home_lon(), radius_km=10.0)

        for i in range(self.ring_size):
            sender = ring_users[i]
            receiver = ring_users[(i + 1) % self.ring_size]
            merchant_idx = i % len(colluding_merchants)
            merchant = colluding_merchants[merchant_idx]
            category = colluding_categories[merchant_idx]

            amount = round(base_amount + random.uniform(-50.0, 50.0), 2)
            delay = timedelta(minutes=random.randint(5, 30) * (i + 1))

            txns.append(
                Transaction(
                    user_id=sender,
                    amount=Decimal(str(amount)),
                    currency=self.user.currency,
                    transaction_type=TransactionType.TRANSFER,
                    merchant_id=merchant,
                    merchant_category=category,
                    card_number_hash=(
                        self.user.card_hash
                        if sender == self.user.user_id
                        else _hash_card(uuid4().hex)
                    ),
                    ip_address=_random_ip(),
                    device_id=f"dev-{uuid4().hex[:12]}",
                    latitude=lat,
                    longitude=lon,
                    timestamp=base_time + delay,
                    metadata=_scenario_meta(
                        "merchant_collusion",
                        ring_position=i,
                        ring_size=self.ring_size,
                        sender_id=str(sender),
                        receiver_id=str(receiver),
                        colluding_merchant=merchant,
                    ),
                )
            )

        return sorted(txns, key=lambda t: t.timestamp)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIO_CLASSES: Final[list[type[FraudScenario]]] = [
    CardTestingAttack,
    AccountTakeoverSimulation,
    FriendlyFraudSimulation,
    SyntheticIdentityFraud,
    MoneyMuleChain,
    RefundAbuseSimulation,
    MerchantCollusionSimulation,
]
