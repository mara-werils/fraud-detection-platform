"""Merchant risk profiling with MCC tiers, chargeback tracking, and reputation scoring.

Complements the lower-level ``MerchantRiskService`` by adding:

* **MCC risk tiers** – ISO-18245 Merchant Category Codes mapped to risk tiers.
* **New-merchant detection** – elevated scrutiny during a configurable probation
  period after first transaction.
* **Chargeback ratio tracking** – per-merchant chargeback / transaction ratio
  with configurable alert thresholds.
* **Merchant velocity anomaly detection** – statistical deviation of recent
  transaction velocity from the merchant's own historical baseline.
* **High-risk category flags** – gambling, crypto, and adult merchants receive
  automatic risk elevation.
* **Merchant reputation scoring** – composite score combining fraud history,
  chargeback ratio, age, and peer comparison.

All state is held in-memory for development; production would use a persistent
store (PostgreSQL, Redis, etc.).
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Lock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ── MCC risk tier mapping ────────────────────────────────────────────────────


class MCCRiskTier(StrEnum):
    """Discrete risk tiers derived from ISO-18245 Merchant Category Codes."""

    CRITICAL = "critical"
    HIGH = "high"
    ELEVATED = "elevated"
    STANDARD = "standard"
    LOW = "low"


# Representative MCC ranges mapped to risk tiers.
# In production this table would be loaded from a configuration store.
MCC_TIER_MAP: dict[str, MCCRiskTier] = {
    # Gambling / betting
    "7995": MCCRiskTier.CRITICAL,
    "7800": MCCRiskTier.CRITICAL,
    "7801": MCCRiskTier.CRITICAL,
    "7802": MCCRiskTier.CRITICAL,
    # Crypto / virtual currency
    "6051": MCCRiskTier.CRITICAL,
    "6211": MCCRiskTier.HIGH,
    # Adult / dating
    "5967": MCCRiskTier.HIGH,
    "7273": MCCRiskTier.HIGH,
    # Money transfer / wire
    "4829": MCCRiskTier.HIGH,
    "6012": MCCRiskTier.HIGH,
    # Pharma / tobacco
    "5912": MCCRiskTier.ELEVATED,
    "5993": MCCRiskTier.ELEVATED,
    # Electronics / high-value goods
    "5732": MCCRiskTier.ELEVATED,
    "5045": MCCRiskTier.ELEVATED,
    # Travel
    "4511": MCCRiskTier.ELEVATED,
    "3000": MCCRiskTier.ELEVATED,
    # Grocery / restaurants / low-risk retail
    "5411": MCCRiskTier.LOW,
    "5812": MCCRiskTier.LOW,
    "5814": MCCRiskTier.LOW,
    "5311": MCCRiskTier.STANDARD,
    "5999": MCCRiskTier.STANDARD,
}

# Tier -> numeric weight used in composite scoring (0–1).
TIER_RISK_WEIGHT: dict[MCCRiskTier, float] = {
    MCCRiskTier.CRITICAL: 0.90,
    MCCRiskTier.HIGH: 0.70,
    MCCRiskTier.ELEVATED: 0.45,
    MCCRiskTier.STANDARD: 0.20,
    MCCRiskTier.LOW: 0.10,
}

# Categories considered inherently high-risk regardless of MCC.
HIGH_RISK_CATEGORIES: frozenset[str] = frozenset({"gambling", "crypto", "adult"})

# ── Configuration defaults ───────────────────────────────────────────────────

DEFAULT_NEW_MERCHANT_DAYS = 90
DEFAULT_CHARGEBACK_WARN_THRESHOLD = 0.01
DEFAULT_CHARGEBACK_CRITICAL_THRESHOLD = 0.02
DEFAULT_VELOCITY_LOOKBACK_HOURS = 24
DEFAULT_VELOCITY_STDDEV_MULTIPLIER = 3.0
DEFAULT_MAX_EVENTS_PER_MERCHANT = 10_000


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TransactionEvent:
    """Lightweight record of a merchant transaction."""

    amount: float
    is_fraud: bool
    is_chargeback: bool
    user_id: str
    timestamp: datetime


@dataclass
class ChargebackStats:
    """Per-merchant chargeback tracking."""

    total_transactions: int = 0
    total_chargebacks: int = 0
    chargeback_amounts: float = 0.0
    transaction_amounts: float = 0.0

    @property
    def ratio(self) -> float:
        if self.total_transactions == 0:
            return 0.0
        return self.total_chargebacks / self.total_transactions

    @property
    def amount_ratio(self) -> float:
        if self.transaction_amounts == 0.0:
            return 0.0
        return self.chargeback_amounts / self.transaction_amounts


@dataclass
class MerchantInfo:
    """Static metadata registered on first sight of a merchant."""

    merchant_id: str
    name: str
    mcc: str
    category: str
    country: str
    registered_at: datetime


@dataclass
class MerchantRiskProfile:
    """Full risk profile returned by the profiler."""

    merchant_id: str
    name: str
    mcc: str
    mcc_risk_tier: MCCRiskTier
    category: str
    country: str

    # Core scores
    risk_score: float
    reputation_score: float
    risk_factors: list[str] = field(default_factory=list)

    # Fraud / chargeback
    fraud_rate: float = 0.0
    chargeback_ratio: float = 0.0
    chargeback_amount_ratio: float = 0.0

    # Temporal
    is_new_merchant: bool = False
    merchant_age_days: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    # Volume
    total_transactions: int = 0
    total_volume: float = 0.0

    # Velocity
    velocity_current: int = 0
    velocity_mean: float = 0.0
    velocity_anomaly: bool = False

    assessed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "merchant_id": self.merchant_id,
            "name": self.name,
            "mcc": self.mcc,
            "mcc_risk_tier": self.mcc_risk_tier.value,
            "category": self.category,
            "country": self.country,
            "risk_score": round(self.risk_score, 4),
            "reputation_score": round(self.reputation_score, 4),
            "risk_factors": self.risk_factors,
            "fraud_rate": round(self.fraud_rate, 6),
            "chargeback_ratio": round(self.chargeback_ratio, 6),
            "chargeback_amount_ratio": round(self.chargeback_amount_ratio, 6),
            "is_new_merchant": self.is_new_merchant,
            "merchant_age_days": self.merchant_age_days,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "total_transactions": self.total_transactions,
            "total_volume": round(self.total_volume, 2),
            "velocity_current": self.velocity_current,
            "velocity_mean": round(self.velocity_mean, 2),
            "velocity_anomaly": self.velocity_anomaly,
            "assessed_at": self.assessed_at.isoformat(),
        }


# ── Helpers ──────────────────────────────────────────────────────────────────


def mcc_to_tier(mcc: str) -> MCCRiskTier:
    """Resolve an MCC code to its risk tier, defaulting to STANDARD."""
    return MCC_TIER_MAP.get(mcc, MCCRiskTier.STANDARD)


def is_high_risk_category(category: str) -> bool:
    """Return True if the category string is inherently high-risk."""
    return category.lower() in HIGH_RISK_CATEGORIES


# ── Service ──────────────────────────────────────────────────────────────────


class MerchantProfiler:
    """Merchant risk profiling service with MCC tiers and reputation scoring.

    Args:
        new_merchant_days: Number of days a merchant is considered "new" and
            subject to elevated scrutiny.
        chargeback_warn_threshold: Chargeback ratio that triggers a warning.
        chargeback_critical_threshold: Chargeback ratio that triggers critical alert.
        velocity_lookback_hours: Window for velocity baseline computation.
        velocity_stddev_multiplier: Standard-deviation multiplier for anomaly.
        max_events: Maximum events retained per merchant (ring buffer).
    """

    def __init__(
        self,
        *,
        new_merchant_days: int = DEFAULT_NEW_MERCHANT_DAYS,
        chargeback_warn_threshold: float = DEFAULT_CHARGEBACK_WARN_THRESHOLD,
        chargeback_critical_threshold: float = DEFAULT_CHARGEBACK_CRITICAL_THRESHOLD,
        velocity_lookback_hours: int = DEFAULT_VELOCITY_LOOKBACK_HOURS,
        velocity_stddev_multiplier: float = DEFAULT_VELOCITY_STDDEV_MULTIPLIER,
        max_events: int = DEFAULT_MAX_EVENTS_PER_MERCHANT,
    ) -> None:
        self._new_merchant_days = new_merchant_days
        self._chargeback_warn = chargeback_warn_threshold
        self._chargeback_critical = chargeback_critical_threshold
        self._velocity_lookback_hours = velocity_lookback_hours
        self._velocity_stddev_mult = velocity_stddev_multiplier
        self._max_events = max_events

        self._lock = Lock()
        self._merchants: dict[str, MerchantInfo] = {}
        self._events: dict[str, deque[TransactionEvent]] = defaultdict(
            lambda: deque(maxlen=max_events)
        )
        self._chargebacks: dict[str, ChargebackStats] = defaultdict(ChargebackStats)
        # Hourly bucket counts for velocity baseline: merchant_id -> list[int]
        self._hourly_buckets: dict[str, list[int]] = defaultdict(list)

    # ── Registration & ingestion ──────────────────────────────────────────

    def register_merchant(
        self,
        merchant_id: str,
        name: str,
        mcc: str,
        category: str = "general",
        country: str = "US",
        registered_at: datetime | None = None,
    ) -> MerchantInfo:
        """Register or update merchant metadata.

        If the merchant already exists, the call is idempotent and returns the
        existing record.
        """
        with self._lock:
            if merchant_id in self._merchants:
                return self._merchants[merchant_id]
            info = MerchantInfo(
                merchant_id=merchant_id,
                name=name,
                mcc=mcc,
                category=category.lower(),
                country=country.upper(),
                registered_at=registered_at or datetime.now(UTC),
            )
            self._merchants[merchant_id] = info
        logger.info("merchant_registered", merchant_id=merchant_id, mcc=mcc)
        return info

    def record_transaction(
        self,
        merchant_id: str,
        amount: float,
        user_id: str,
        *,
        is_fraud: bool = False,
        is_chargeback: bool = False,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a transaction for the given merchant.

        The merchant must have been registered first via ``register_merchant``.

        Raises:
            KeyError: If the merchant is not registered.
        """
        ts = timestamp or datetime.now(UTC)
        with self._lock:
            if merchant_id not in self._merchants:
                raise KeyError(f"Unknown merchant: {merchant_id!r}")
            self._events[merchant_id].append(
                TransactionEvent(
                    amount=amount,
                    is_fraud=is_fraud,
                    is_chargeback=is_chargeback,
                    user_id=user_id,
                    timestamp=ts,
                )
            )
            cb = self._chargebacks[merchant_id]
            cb.total_transactions += 1
            cb.transaction_amounts += amount
            if is_chargeback:
                cb.total_chargebacks += 1
                cb.chargeback_amounts += amount

    # ── Querying ──────────────────────────────────────────────────────────

    def get_chargeback_stats(self, merchant_id: str) -> ChargebackStats:
        """Return chargeback statistics for a merchant.

        Raises:
            KeyError: If the merchant is not registered.
        """
        with self._lock:
            if merchant_id not in self._merchants:
                raise KeyError(f"Unknown merchant: {merchant_id!r}")
            return self._chargebacks[merchant_id]

    def is_new_merchant(self, merchant_id: str, now: datetime | None = None) -> bool:
        """Return True if the merchant is within its elevated scrutiny period.

        Raises:
            KeyError: If the merchant is not registered.
        """
        now = now or datetime.now(UTC)
        with self._lock:
            if merchant_id not in self._merchants:
                raise KeyError(f"Unknown merchant: {merchant_id!r}")
            info = self._merchants[merchant_id]
        age = (now - info.registered_at).days
        return age < self._new_merchant_days

    def detect_velocity_anomaly(
        self, merchant_id: str, now: datetime | None = None
    ) -> tuple[bool, int, float]:
        """Check whether the merchant's recent velocity is anomalous.

        Returns:
            Tuple of (is_anomaly, current_velocity, baseline_mean).

        Raises:
            KeyError: If the merchant is not registered.
        """
        now = now or datetime.now(UTC)
        with self._lock:
            if merchant_id not in self._merchants:
                raise KeyError(f"Unknown merchant: {merchant_id!r}")
            events = list(self._events[merchant_id])

        if not events:
            return False, 0, 0.0

        lookback = now - timedelta(hours=self._velocity_lookback_hours)
        current_hour_start = now - timedelta(hours=1)

        # Bucket events into hours within the lookback window.
        hourly_counts: dict[int, int] = defaultdict(int)
        current_count = 0
        for ev in events:
            if ev.timestamp >= current_hour_start:
                current_count += 1
            if lookback <= ev.timestamp < current_hour_start:
                # Which hour bucket?
                hours_ago = int((now - ev.timestamp).total_seconds() // 3600)
                hourly_counts[hours_ago] += 1

        if len(hourly_counts) < 2:
            return False, current_count, 0.0

        counts = list(hourly_counts.values())
        mean_vel = statistics.mean(counts)
        stdev_vel = statistics.stdev(counts) if len(counts) >= 2 else 0.0

        if stdev_vel > 0:
            z = (current_count - mean_vel) / stdev_vel
            is_anomaly = z >= self._velocity_stddev_mult
        else:
            # No variance; flag if current far exceeds mean.
            is_anomaly = current_count > mean_vel * (1 + self._velocity_stddev_mult)

        return is_anomaly, current_count, mean_vel

    # ── Profiling ─────────────────────────────────────────────────────────

    def profile(
        self, merchant_id: str, now: datetime | None = None
    ) -> MerchantRiskProfile:
        """Build a full risk profile for the merchant.

        Raises:
            KeyError: If the merchant is not registered.
        """
        now = now or datetime.now(UTC)
        with self._lock:
            if merchant_id not in self._merchants:
                raise KeyError(f"Unknown merchant: {merchant_id!r}")
            info = self._merchants[merchant_id]
            events = list(self._events[merchant_id])
            cb = self._chargebacks[merchant_id]

        tier = mcc_to_tier(info.mcc)
        risk_factors: list[str] = []

        # ── Fraud rate ────────────────────────────────────────────────────
        total_txns = len(events)
        fraud_count = sum(1 for e in events if e.is_fraud)
        fraud_rate = fraud_count / total_txns if total_txns else 0.0
        total_volume = sum(e.amount for e in events)

        if fraud_rate >= 0.10:
            risk_factors.append(f"Fraud rate {fraud_rate:.2%} exceeds 10% threshold")
        elif fraud_rate >= 0.05:
            risk_factors.append(f"Elevated fraud rate: {fraud_rate:.2%}")

        # ── Chargeback ratio ──────────────────────────────────────────────
        cb_ratio = cb.ratio
        cb_amount_ratio = cb.amount_ratio

        if cb_ratio >= self._chargeback_critical:
            risk_factors.append(
                f"Chargeback ratio {cb_ratio:.2%} exceeds critical threshold "
                f"({self._chargeback_critical:.2%})"
            )
        elif cb_ratio >= self._chargeback_warn:
            risk_factors.append(
                f"Chargeback ratio {cb_ratio:.2%} exceeds warning threshold "
                f"({self._chargeback_warn:.2%})"
            )

        # ── MCC tier risk ────────────────────────────────────────────────
        if tier in (MCCRiskTier.CRITICAL, MCCRiskTier.HIGH):
            risk_factors.append(f"MCC {info.mcc} maps to {tier.value} risk tier")

        # ── High-risk category ───────────────────────────────────────────
        if is_high_risk_category(info.category):
            risk_factors.append(
                f"High-risk merchant category: {info.category}"
            )

        # ── New merchant scrutiny ────────────────────────────────────────
        merchant_age_days = (now - info.registered_at).days
        new_merchant = merchant_age_days < self._new_merchant_days
        if new_merchant:
            risk_factors.append(
                f"New merchant ({merchant_age_days} days old, "
                f"scrutiny period: {self._new_merchant_days} days)"
            )

        # ── Velocity anomaly ─────────────────────────────────────────────
        vel_anomaly, vel_current, vel_mean = self.detect_velocity_anomaly(
            merchant_id, now=now
        )
        if vel_anomaly:
            risk_factors.append(
                f"Velocity anomaly: {vel_current} txns in last hour vs "
                f"{vel_mean:.1f} baseline mean"
            )

        # ── Temporal stats ───────────────────────────────────────────────
        first_seen = min((e.timestamp for e in events), default=None)
        last_seen = max((e.timestamp for e in events), default=None)

        # ── Composite risk score ─────────────────────────────────────────
        risk_score = self._compute_risk_score(
            tier=tier,
            category=info.category,
            fraud_rate=fraud_rate,
            chargeback_ratio=cb_ratio,
            is_new=new_merchant,
            velocity_anomaly=vel_anomaly,
            merchant_age_days=merchant_age_days,
        )

        # ── Reputation score (inverse of risk, with smoothing) ───────────
        reputation_score = self._compute_reputation_score(
            fraud_rate=fraud_rate,
            chargeback_ratio=cb_ratio,
            merchant_age_days=merchant_age_days,
            total_transactions=total_txns,
            risk_score=risk_score,
        )

        return MerchantRiskProfile(
            merchant_id=merchant_id,
            name=info.name,
            mcc=info.mcc,
            mcc_risk_tier=tier,
            category=info.category,
            country=info.country,
            risk_score=risk_score,
            reputation_score=reputation_score,
            risk_factors=risk_factors,
            fraud_rate=fraud_rate,
            chargeback_ratio=cb_ratio,
            chargeback_amount_ratio=cb_amount_ratio,
            is_new_merchant=new_merchant,
            merchant_age_days=merchant_age_days,
            first_seen=first_seen,
            last_seen=last_seen,
            total_transactions=total_txns,
            total_volume=total_volume,
            velocity_current=vel_current,
            velocity_mean=vel_mean,
            velocity_anomaly=vel_anomaly,
            assessed_at=now,
        )

    def profile_all(self) -> list[MerchantRiskProfile]:
        """Profile every registered merchant and return sorted by risk descending."""
        with self._lock:
            ids = list(self._merchants.keys())
        profiles = [self.profile(mid) for mid in ids]
        profiles.sort(key=lambda p: p.risk_score, reverse=True)
        return profiles

    # ── Scoring internals ─────────────────────────────────────────────────

    @staticmethod
    def _compute_risk_score(
        *,
        tier: MCCRiskTier,
        category: str,
        fraud_rate: float,
        chargeback_ratio: float,
        is_new: bool,
        velocity_anomaly: bool,
        merchant_age_days: int,
    ) -> float:
        """Weighted composite risk score in [0, 1].

        Component weights:
            MCC tier risk       0.20
            Category risk       0.10
            Fraud rate          0.30
            Chargeback ratio    0.20
            New-merchant boost  0.10
            Velocity anomaly    0.10
        """
        tier_component = TIER_RISK_WEIGHT.get(tier, 0.20)
        category_component = 0.80 if is_high_risk_category(category) else 0.15
        fraud_component = min(fraud_rate / 0.15, 1.0)
        cb_component = min(chargeback_ratio / 0.02, 1.0)

        # New-merchant penalty: decays linearly over the scrutiny period.
        if is_new:
            new_component = max(0.0, 1.0 - merchant_age_days / DEFAULT_NEW_MERCHANT_DAYS)
        else:
            new_component = 0.0

        velocity_component = 1.0 if velocity_anomaly else 0.0

        score = (
            0.20 * tier_component
            + 0.10 * category_component
            + 0.30 * fraud_component
            + 0.20 * cb_component
            + 0.10 * new_component
            + 0.10 * velocity_component
        )

        return max(0.0, min(1.0, score))

    @staticmethod
    def _compute_reputation_score(
        *,
        fraud_rate: float,
        chargeback_ratio: float,
        merchant_age_days: int,
        total_transactions: int,
        risk_score: float,
    ) -> float:
        """Merchant reputation score in [0, 1] (1 = excellent).

        Combines:
          - Inverse of fraud and chargeback rates.
          - Age bonus (longer track record → higher reputation).
          - Transaction volume confidence (more data → more trust).
          - Inverse of composite risk score.
        """
        # Fraud / chargeback penalty (0 = bad, 1 = clean).
        clean_score = max(0.0, 1.0 - fraud_rate * 5.0 - chargeback_ratio * 20.0)

        # Age bonus: saturates at 365 days.
        age_factor = min(merchant_age_days / 365.0, 1.0)

        # Volume confidence: logarithmic growth, saturates around 1000 txns.
        if total_transactions > 0:
            volume_factor = min(math.log10(total_transactions + 1) / 3.0, 1.0)
        else:
            volume_factor = 0.0

        # Inverse risk
        safety_factor = 1.0 - risk_score

        reputation = (
            0.35 * clean_score
            + 0.20 * age_factor
            + 0.15 * volume_factor
            + 0.30 * safety_factor
        )

        return max(0.0, min(1.0, reputation))


__all__ = ["ChargebackStats", "MCCRiskTier", "MerchantInfo", "MerchantProfiler", "MerchantRiskProfile", "TransactionEvent", "is_high_risk_category", "mcc_to_tier"]
