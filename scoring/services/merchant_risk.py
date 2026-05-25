"""Merchant risk profiling service.

Builds and maintains risk profiles for merchants based on transaction history,
fraud rates, chargeback patterns, and velocity signals. Supports real-time
risk assessment and batch score updates.

In production, persistent storage would be backed by PostgreSQL or ClickHouse;
the in-memory implementation here is suitable for development and demonstration.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from threading import Lock
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

MERCHANT_RISK_SCORE = Histogram(
    "merchant_risk_score",
    "Distribution of merchant risk scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MERCHANT_RISK_LEVEL = Gauge(
    "merchant_risk_level_count",
    "Number of merchants at each risk level",
    ["level"],
)

MERCHANT_TRANSACTIONS_TOTAL = Counter(
    "merchant_transactions_recorded_total",
    "Total transactions recorded via the merchant risk service",
)

MERCHANT_ANOMALIES_DETECTED = Counter(
    "merchant_anomalies_detected_total",
    "Total merchant anomalies detected",
    ["anomaly_type"],
)

MERCHANT_RISK_UPDATES = Counter(
    "merchant_risk_score_updates_total",
    "Total batch risk-score update cycles run",
)


# ── Enumerations ──────────────────────────────────────────────────────────────


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MerchantCategory(StrEnum):
    GAMBLING = "gambling"
    CRYPTO = "crypto"
    ADULT = "adult"
    FIREARMS = "firearms"
    PHARMA = "pharma"
    TRAVEL = "travel"
    ELECTRONICS = "electronics"
    FASHION = "fashion"
    FOOD = "food"
    SOFTWARE = "software"
    FINANCIAL_SERVICES = "financial_services"
    GENERAL = "general"


# Base risk contribution from category alone (0–1 scale).
CATEGORY_BASE_RISK: dict[MerchantCategory, float] = {
    MerchantCategory.GAMBLING: 0.70,
    MerchantCategory.CRYPTO: 0.65,
    MerchantCategory.ADULT: 0.60,
    MerchantCategory.FIREARMS: 0.55,
    MerchantCategory.PHARMA: 0.50,
    MerchantCategory.FINANCIAL_SERVICES: 0.40,
    MerchantCategory.TRAVEL: 0.30,
    MerchantCategory.ELECTRONICS: 0.25,
    MerchantCategory.SOFTWARE: 0.20,
    MerchantCategory.FASHION: 0.15,
    MerchantCategory.FOOD: 0.10,
    MerchantCategory.GENERAL: 0.20,
}

# Countries with elevated baseline fraud risk (ISO-3166-1 alpha-2).
HIGH_RISK_COUNTRIES: frozenset[str] = frozenset(
    {"NG", "RO", "BD", "PK", "VN", "ID", "GH", "CM", "UA", "BY"}
)
MEDIUM_RISK_COUNTRIES: frozenset[str] = frozenset(
    {"IN", "BR", "MX", "CO", "PH", "EG", "ZA", "MA", "IR", "IQ"}
)


def _country_risk_factor(country: str) -> float:
    """Return an additive risk factor (0–0.20) for a given country code."""
    cc = country.upper()
    if cc in HIGH_RISK_COUNTRIES:
        return 0.20
    if cc in MEDIUM_RISK_COUNTRIES:
        return 0.10
    return 0.0


# ── Pydantic models ───────────────────────────────────────────────────────────


class MerchantProfile(BaseModel):
    """Aggregated risk profile for a single merchant."""

    merchant_id: str
    merchant_name: str
    category: MerchantCategory = MerchantCategory.GENERAL
    country: str = "US"

    # Volume / activity stats
    total_transactions: int = 0
    total_volume: float = 0.0
    avg_transaction_amount: float = 0.0

    # Risk indicators
    fraud_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    chargeback_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel = RiskLevel.LOW

    # Temporal
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Velocity (counts in window)
    velocity_1h: int = 0
    velocity_24h: int = 0
    unique_users_24h: int = 0

    @field_validator("country")
    @classmethod
    def normalise_country(cls, v: str) -> str:
        return v.upper()

    def to_dict(self) -> dict[str, Any]:
        return {
            "merchant_id": self.merchant_id,
            "merchant_name": self.merchant_name,
            "category": self.category.value,
            "country": self.country,
            "total_transactions": self.total_transactions,
            "total_volume": round(self.total_volume, 2),
            "avg_transaction_amount": round(self.avg_transaction_amount, 2),
            "fraud_rate": round(self.fraud_rate, 6),
            "chargeback_rate": round(self.chargeback_rate, 6),
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "velocity_1h": self.velocity_1h,
            "velocity_24h": self.velocity_24h,
            "unique_users_24h": self.unique_users_24h,
        }


class MerchantRiskAssessment(BaseModel):
    """Point-in-time risk assessment result for a merchant."""

    merchant_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    risk_factors: list[str] = Field(default_factory=list)

    # Comparative / contextual signals
    fraud_rate_percentile: float = Field(default=0.0, ge=0.0, le=100.0)
    volume_anomaly: bool = False
    category_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "merchant_id": self.merchant_id,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "risk_factors": self.risk_factors,
            "fraud_rate_percentile": round(self.fraud_rate_percentile, 2),
            "volume_anomaly": self.volume_anomaly,
            "category_risk": round(self.category_risk, 4),
            "assessed_at": self.assessed_at.isoformat(),
        }


class CategoryStats(BaseModel):
    """Aggregated statistics across all merchants in a category."""

    category: MerchantCategory
    merchant_count: int = 0
    avg_fraud_rate: float = 0.0
    avg_risk_score: float = 0.0
    total_volume: float = 0.0
    high_risk_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "merchant_count": self.merchant_count,
            "avg_fraud_rate": round(self.avg_fraud_rate, 6),
            "avg_risk_score": round(self.avg_risk_score, 4),
            "total_volume": round(self.total_volume, 2),
            "high_risk_count": self.high_risk_count,
        }


class MerchantAnomaly(BaseModel):
    """Detected anomaly for a specific merchant."""

    merchant_id: str
    anomaly_type: str
    description: str
    severity: RiskLevel
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "merchant_id": self.merchant_id,
            "anomaly_type": self.anomaly_type,
            "description": self.description,
            "severity": self.severity.value,
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata,
        }


# ── Internal event record ─────────────────────────────────────────────────────


class _TxnRecord:
    """Lightweight record of a single merchant transaction event."""

    __slots__ = ("amount", "fraud_score", "user_id", "timestamp")

    def __init__(
        self,
        amount: float,
        fraud_score: float,
        user_id: str,
        timestamp: datetime,
    ) -> None:
        self.amount = amount
        self.fraud_score = fraud_score
        self.user_id = user_id
        self.timestamp = timestamp


# ── Service ───────────────────────────────────────────────────────────────────


class MerchantRiskService:
    """Merchant risk profiling and anomaly detection service.

    Maintains in-memory merchant profiles derived from a rolling stream of
    transaction events. Risk scores are computed from a weighted combination of:

      - Category base risk
      - Country risk factor
      - Observed fraud rate (exponentially weighted)
      - Chargeback rate
      - Transaction velocity spikes
      - User diversity (low unique-user ratio → mule-account pattern)
      - Average transaction amount vs. category peer group

    Args:
        fraud_score_threshold: Minimum fraud_score for a transaction to be
            counted as fraudulent when computing fraud_rate. Default 0.7.
        velocity_spike_multiplier: A merchant whose 1-hour velocity exceeds
            this multiple of its rolling 24-hour hourly average is flagged as
            a velocity spike anomaly. Default 5.0.
        volume_z_score_threshold: Z-score above which a merchant's volume is
            considered anomalous vs. its category peers. Default 3.0.
    """

    def __init__(
        self,
        fraud_score_threshold: float = 0.7,
        velocity_spike_multiplier: float = 5.0,
        volume_z_score_threshold: float = 3.0,
    ) -> None:
        self._fraud_score_threshold = fraud_score_threshold
        self._velocity_spike_multiplier = velocity_spike_multiplier
        self._volume_z_score_threshold = volume_z_score_threshold

        self._lock = Lock()

        # Canonical profile store: merchant_id -> MerchantProfile
        self._profiles: dict[str, MerchantProfile] = {}

        # Rolling event deques: merchant_id -> deque[_TxnRecord]
        # We keep a maximum of 10 000 events per merchant to bound memory.
        self._events: dict[str, deque[_TxnRecord]] = defaultdict(
            lambda: deque(maxlen=10_000)
        )

        # Merchant metadata (name, category, country) supplied on first record
        self._metadata: dict[str, dict[str, str]] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def record_transaction(
        self,
        merchant_id: str,
        amount: float,
        fraud_score: float,
        user_id: str,
        timestamp: datetime | None = None,
        *,
        merchant_name: str | None = None,
        category: str | MerchantCategory = MerchantCategory.GENERAL,
        country: str = "US",
    ) -> None:
        """Record a transaction event for a merchant.

        Args:
            merchant_id: Unique merchant identifier.
            amount: Transaction amount in the platform's base currency.
            fraud_score: Model fraud score for this transaction (0–1).
            user_id: Identifier of the user who made the transaction.
            timestamp: Event timestamp; defaults to now (UTC).
            merchant_name: Display name; captured on first call if absent.
            category: Merchant category; used to initialise the profile.
            country: ISO-3166-1 alpha-2 country code.
        """
        ts = timestamp or datetime.now(timezone.utc)
        record = _TxnRecord(
            amount=amount,
            fraud_score=max(0.0, min(1.0, fraud_score)),
            user_id=user_id,
            timestamp=ts,
        )

        with self._lock:
            # Store metadata if this is the first event for this merchant.
            if merchant_id not in self._metadata:
                self._metadata[merchant_id] = {
                    "merchant_name": merchant_name or merchant_id,
                    "category": MerchantCategory(category).value
                    if isinstance(category, str)
                    else category.value,
                    "country": country.upper(),
                }
            self._events[merchant_id].append(record)

        MERCHANT_TRANSACTIONS_TOTAL.inc()
        logger.debug(
            "merchant_transaction_recorded",
            merchant_id=merchant_id,
            amount=amount,
            fraud_score=fraud_score,
        )

    def get_profile(self, merchant_id: str) -> MerchantProfile:
        """Return the current risk profile for a merchant.

        The profile is rebuilt from events on every call to ensure freshness.
        For high-throughput deployments, cache results with a short TTL.

        Raises:
            KeyError: If the merchant has no recorded events.
        """
        with self._lock:
            if merchant_id not in self._events or not self._events[merchant_id]:
                raise KeyError(f"Unknown merchant: {merchant_id!r}")
            events = list(self._events[merchant_id])
            meta = self._metadata[merchant_id]

        profile = self._build_profile(merchant_id, meta, events)

        with self._lock:
            self._profiles[merchant_id] = profile

        return profile

    def assess_risk(self, merchant_id: str) -> MerchantRiskAssessment:
        """Produce a detailed risk assessment for a merchant.

        Computes contextual signals such as fraud-rate percentile versus all
        known merchants and volume anomaly detection.

        Raises:
            KeyError: If the merchant has no recorded events.
        """
        profile = self.get_profile(merchant_id)

        with self._lock:
            all_profiles = list(self._profiles.values())

        risk_factors: list[str] = []
        category_risk = CATEGORY_BASE_RISK.get(profile.category, 0.20)

        # ── Fraud-rate percentile ──────────────────────────────────────────
        fraud_rates = sorted(p.fraud_rate for p in all_profiles)
        if fraud_rates:
            rank = sum(1 for r in fraud_rates if r <= profile.fraud_rate)
            percentile = (rank / len(fraud_rates)) * 100.0
        else:
            percentile = 0.0

        # ── Volume anomaly ────────────────────────────────────────────────
        peer_volumes = [
            p.total_volume
            for p in all_profiles
            if p.category == profile.category and p.merchant_id != merchant_id
        ]
        volume_anomaly = False
        if len(peer_volumes) >= 3:
            mean_vol = statistics.mean(peer_volumes)
            stdev_vol = statistics.stdev(peer_volumes)
            if stdev_vol > 0:
                z = (profile.total_volume - mean_vol) / stdev_vol
                if abs(z) >= self._volume_z_score_threshold:
                    volume_anomaly = True
                    risk_factors.append(
                        f"Volume z-score {z:.2f} deviates from category peers"
                    )

        # ── Identify discrete risk factors ────────────────────────────────
        if category_risk >= 0.60:
            risk_factors.append(
                f"High-risk merchant category: {profile.category.value}"
            )
        elif category_risk >= 0.40:
            risk_factors.append(
                f"Elevated-risk merchant category: {profile.category.value}"
            )

        country_factor = _country_risk_factor(profile.country)
        if country_factor >= 0.15:
            risk_factors.append(
                f"High-risk country of operation: {profile.country}"
            )
        elif country_factor > 0:
            risk_factors.append(
                f"Elevated-risk country: {profile.country}"
            )

        if profile.fraud_rate >= 0.10:
            risk_factors.append(
                f"Fraud rate {profile.fraud_rate:.2%} exceeds 10% threshold"
            )
        elif profile.fraud_rate >= 0.05:
            risk_factors.append(
                f"Elevated fraud rate: {profile.fraud_rate:.2%}"
            )

        if profile.chargeback_rate >= 0.02:
            risk_factors.append(
                f"Chargeback rate {profile.chargeback_rate:.2%} above 2% threshold"
            )

        if profile.velocity_1h > 500:
            risk_factors.append(
                f"High 1-hour transaction velocity: {profile.velocity_1h}"
            )

        if profile.unique_users_24h > 0:
            ratio = profile.velocity_24h / max(profile.unique_users_24h, 1)
            if ratio >= 10:
                risk_factors.append(
                    f"Low user diversity: {ratio:.1f} transactions per unique user (24 h)"
                )

        if percentile >= 90:
            risk_factors.append(
                f"Fraud rate in top {100 - percentile:.0f}% of all merchants"
            )

        logger.info(
            "merchant_risk_assessed",
            merchant_id=merchant_id,
            risk_score=profile.risk_score,
            risk_level=profile.risk_level.value,
            risk_factor_count=len(risk_factors),
        )

        return MerchantRiskAssessment(
            merchant_id=merchant_id,
            risk_score=profile.risk_score,
            risk_level=profile.risk_level,
            risk_factors=risk_factors,
            fraud_rate_percentile=percentile,
            volume_anomaly=volume_anomaly,
            category_risk=category_risk,
        )

    def get_top_risky_merchants(self, limit: int = 20) -> list[MerchantProfile]:
        """Return up to *limit* merchants ordered by risk score descending.

        Rebuilds all profiles from events before ranking so scores are fresh.
        """
        merchant_ids: list[str]
        with self._lock:
            merchant_ids = list(self._events.keys())

        profiles: list[MerchantProfile] = []
        for mid in merchant_ids:
            try:
                profiles.append(self.get_profile(mid))
            except KeyError:
                pass

        profiles.sort(key=lambda p: p.risk_score, reverse=True)
        return profiles[:limit]

    def get_category_stats(self) -> dict[str, CategoryStats]:
        """Return aggregated statistics grouped by merchant category."""
        with self._lock:
            all_profiles = list(self._profiles.values())

        # Ensure profiles are refreshed
        if not all_profiles:
            all_profiles = self.get_top_risky_merchants(limit=10_000)

        grouped: dict[MerchantCategory, list[MerchantProfile]] = defaultdict(list)
        for p in all_profiles:
            grouped[p.category].append(p)

        stats: dict[str, CategoryStats] = {}
        for cat, members in grouped.items():
            avg_fraud = statistics.mean(m.fraud_rate for m in members) if members else 0.0
            avg_risk = statistics.mean(m.risk_score for m in members) if members else 0.0
            total_vol = sum(m.total_volume for m in members)
            high_risk = sum(
                1 for m in members if m.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
            )
            stats[cat.value] = CategoryStats(
                category=cat,
                merchant_count=len(members),
                avg_fraud_rate=avg_fraud,
                avg_risk_score=avg_risk,
                total_volume=total_vol,
                high_risk_count=high_risk,
            )

        return stats

    def detect_merchant_anomalies(self) -> list[MerchantAnomaly]:
        """Scan all known merchants and return a list of detected anomalies.

        Anomaly types:
          - ``velocity_spike``: 1-hour velocity >> rolling 24-hour average.
          - ``fraud_surge``: Recent (1-hour) fraud rate much higher than baseline.
          - ``dormant_reactivation``: Merchant inactive for >30 days suddenly active.
          - ``single_user_dominance``: >80% of 24-hour transactions from one user.
        """
        now = datetime.now(timezone.utc)
        anomalies: list[MerchantAnomaly] = []

        with self._lock:
            merchant_ids = list(self._events.keys())

        for mid in merchant_ids:
            with self._lock:
                if mid not in self._events:
                    continue
                events = list(self._events[mid])

            if not events:
                continue

            cutoff_1h = now - timedelta(hours=1)
            cutoff_24h = now - timedelta(hours=24)

            events_1h = [e for e in events if e.timestamp >= cutoff_1h]
            events_24h = [e for e in events if e.timestamp >= cutoff_24h]
            events_before_24h = [e for e in events if e.timestamp < cutoff_24h]

            # ── Velocity spike ────────────────────────────────────────────
            hourly_avg_24h = len(events_24h) / 24.0
            if (
                len(events_1h) > 10
                and hourly_avg_24h > 0
                and len(events_1h) >= self._velocity_spike_multiplier * hourly_avg_24h
            ):
                anomalies.append(
                    MerchantAnomaly(
                        merchant_id=mid,
                        anomaly_type="velocity_spike",
                        description=(
                            f"1-hour volume ({len(events_1h)} txns) is "
                            f"{len(events_1h) / max(hourly_avg_24h, 1):.1f}× "
                            f"the 24-hour hourly average"
                        ),
                        severity=RiskLevel.HIGH,
                        metadata={
                            "velocity_1h": len(events_1h),
                            "hourly_avg_24h": round(hourly_avg_24h, 2),
                        },
                    )
                )
                MERCHANT_ANOMALIES_DETECTED.labels(anomaly_type="velocity_spike").inc()

            # ── Fraud surge ───────────────────────────────────────────────
            if events_1h and events_before_24h:
                fraud_rate_1h = sum(
                    1 for e in events_1h if e.fraud_score >= self._fraud_score_threshold
                ) / len(events_1h)
                baseline_fraud_rate = sum(
                    1 for e in events_before_24h
                    if e.fraud_score >= self._fraud_score_threshold
                ) / len(events_before_24h)
                if fraud_rate_1h >= 0.20 and fraud_rate_1h >= 3 * max(baseline_fraud_rate, 0.01):
                    anomalies.append(
                        MerchantAnomaly(
                            merchant_id=mid,
                            anomaly_type="fraud_surge",
                            description=(
                                f"1-hour fraud rate {fraud_rate_1h:.2%} is "
                                f"{fraud_rate_1h / max(baseline_fraud_rate, 0.001):.1f}× "
                                f"baseline ({baseline_fraud_rate:.2%})"
                            ),
                            severity=RiskLevel.CRITICAL,
                            metadata={
                                "fraud_rate_1h": round(fraud_rate_1h, 4),
                                "baseline_fraud_rate": round(baseline_fraud_rate, 4),
                            },
                        )
                    )
                    MERCHANT_ANOMALIES_DETECTED.labels(anomaly_type="fraud_surge").inc()

            # ── Dormant reactivation ──────────────────────────────────────
            if events_before_24h and events_1h:
                last_old_ts = max(e.timestamp for e in events_before_24h)
                gap_days = (now - last_old_ts).total_seconds() / 86_400
                if gap_days >= 30:
                    anomalies.append(
                        MerchantAnomaly(
                            merchant_id=mid,
                            anomaly_type="dormant_reactivation",
                            description=(
                                f"Merchant was dormant for {gap_days:.0f} days "
                                f"and has resumed activity"
                            ),
                            severity=RiskLevel.MEDIUM,
                            metadata={"dormant_days": round(gap_days, 1)},
                        )
                    )
                    MERCHANT_ANOMALIES_DETECTED.labels(
                        anomaly_type="dormant_reactivation"
                    ).inc()

            # ── Single user dominance ─────────────────────────────────────
            if len(events_24h) >= 20:
                user_counts: dict[str, int] = defaultdict(int)
                for e in events_24h:
                    user_counts[e.user_id] += 1
                top_user_count = max(user_counts.values())
                dominance = top_user_count / len(events_24h)
                if dominance >= 0.80:
                    top_user = max(user_counts, key=lambda u: user_counts[u])
                    anomalies.append(
                        MerchantAnomaly(
                            merchant_id=mid,
                            anomaly_type="single_user_dominance",
                            description=(
                                f"Single user accounts for {dominance:.0%} of "
                                f"24-hour transactions — possible mule account"
                            ),
                            severity=RiskLevel.HIGH,
                            metadata={
                                "dominant_user_id": top_user,
                                "dominance_ratio": round(dominance, 4),
                                "transactions_24h": len(events_24h),
                            },
                        )
                    )
                    MERCHANT_ANOMALIES_DETECTED.labels(
                        anomaly_type="single_user_dominance"
                    ).inc()

        logger.info("merchant_anomaly_scan_complete", anomaly_count=len(anomalies))
        return anomalies

    def update_risk_scores(self) -> int:
        """Batch-refresh risk scores for every known merchant.

        Returns:
            Number of profiles updated.
        """
        with self._lock:
            merchant_ids = list(self._events.keys())

        updated = 0
        for mid in merchant_ids:
            try:
                profile = self.get_profile(mid)
                MERCHANT_RISK_SCORE.observe(profile.risk_score)
                updated += 1
            except Exception:
                logger.exception("merchant_risk_score_update_failed", merchant_id=mid)

        # Refresh Prometheus level gauges
        level_counts: dict[str, int] = defaultdict(int)
        with self._lock:
            for p in self._profiles.values():
                level_counts[p.risk_level.value] += 1
        for level in RiskLevel:
            MERCHANT_RISK_LEVEL.labels(level=level.value).set(level_counts[level.value])

        MERCHANT_RISK_UPDATES.inc()
        logger.info("merchant_risk_batch_update_complete", updated=updated)
        return updated

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_profile(
        self,
        merchant_id: str,
        meta: dict[str, str],
        events: list[_TxnRecord],
    ) -> MerchantProfile:
        """Derive a MerchantProfile from the raw event list."""
        now = datetime.now(timezone.utc)
        category = MerchantCategory(meta.get("category", MerchantCategory.GENERAL))
        country = meta.get("country", "US")

        cutoff_1h = now - timedelta(hours=1)
        cutoff_24h = now - timedelta(hours=24)

        events_1h = [e for e in events if e.timestamp >= cutoff_1h]
        events_24h = [e for e in events if e.timestamp >= cutoff_24h]

        total_txns = len(events)
        total_volume = sum(e.amount for e in events)
        avg_amount = total_volume / total_txns if total_txns else 0.0

        fraud_count = sum(
            1 for e in events if e.fraud_score >= self._fraud_score_threshold
        )
        fraud_rate = fraud_count / total_txns if total_txns else 0.0

        # Chargeback rate is estimated from very-high fraud-score transactions
        # (score >= 0.90); in production this would come from acquirer data.
        chargeback_count = sum(1 for e in events if e.fraud_score >= 0.90)
        chargeback_rate = chargeback_count / total_txns if total_txns else 0.0

        velocity_1h = len(events_1h)
        velocity_24h = len(events_24h)
        unique_users_24h = len({e.user_id for e in events_24h})

        first_seen = min(e.timestamp for e in events)
        last_seen = max(e.timestamp for e in events)

        risk_score = self._compute_risk_score(
            category=category,
            country=country,
            fraud_rate=fraud_rate,
            chargeback_rate=chargeback_rate,
            velocity_1h=velocity_1h,
            velocity_24h=velocity_24h,
            unique_users_24h=unique_users_24h,
            avg_amount=avg_amount,
            total_txns=total_txns,
        )
        risk_level = _risk_level_from_score(risk_score)

        return MerchantProfile(
            merchant_id=merchant_id,
            merchant_name=meta.get("merchant_name", merchant_id),
            category=category,
            country=country,
            total_transactions=total_txns,
            total_volume=total_volume,
            avg_transaction_amount=avg_amount,
            fraud_rate=fraud_rate,
            chargeback_rate=chargeback_rate,
            risk_score=risk_score,
            risk_level=risk_level,
            first_seen=first_seen,
            last_seen=last_seen,
            velocity_1h=velocity_1h,
            velocity_24h=velocity_24h,
            unique_users_24h=unique_users_24h,
        )

    @staticmethod
    def _compute_risk_score(
        *,
        category: MerchantCategory,
        country: str,
        fraud_rate: float,
        chargeback_rate: float,
        velocity_1h: int,
        velocity_24h: int,
        unique_users_24h: int,
        avg_amount: float,
        total_txns: int,
    ) -> float:
        """Compute a composite risk score in [0, 1].

        Weights are heuristic and should be calibrated against labelled data
        in a production environment.
        """
        # ── Component 1: category base risk (weight 0.25) ─────────────────
        cat_risk = CATEGORY_BASE_RISK.get(category, 0.20)

        # ── Component 2: country risk (weight 0.10) ───────────────────────
        country_risk = _country_risk_factor(country) / 0.20  # normalise to 0-1

        # ── Component 3: fraud rate (weight 0.35) ─────────────────────────
        # Apply a sigmoid-like transform: aggressive scaling up to 0.15 fraud rate.
        fraud_component = min(fraud_rate / 0.15, 1.0)

        # ── Component 4: chargeback rate (weight 0.15) ────────────────────
        # 2% is the industry danger threshold.
        chargeback_component = min(chargeback_rate / 0.02, 1.0)

        # ── Component 5: velocity anomaly (weight 0.10) ───────────────────
        # Score rises with 1-hour velocity relative to a "normal" 50 txn/h.
        hourly_avg = velocity_24h / 24.0
        if hourly_avg > 0 and velocity_1h >= 5:
            spike_ratio = velocity_1h / hourly_avg
            velocity_component = min((spike_ratio - 1.0) / 9.0, 1.0) if spike_ratio > 1 else 0.0
        else:
            velocity_component = 0.0

        # ── Component 6: user diversity penalty (weight 0.05) ─────────────
        # Very low unique-user ratio with significant volume is suspicious.
        if velocity_24h >= 10 and unique_users_24h > 0:
            diversity_ratio = unique_users_24h / velocity_24h
            diversity_component = max(0.0, 1.0 - diversity_ratio * 5)  # low ratio -> high score
        else:
            diversity_component = 0.0

        score = (
            0.25 * cat_risk
            + 0.10 * country_risk
            + 0.35 * fraud_component
            + 0.15 * chargeback_component
            + 0.10 * velocity_component
            + 0.05 * diversity_component
        )

        # Cold-start penalty: distrust very low sample sizes
        if total_txns < 10:
            confidence = total_txns / 10.0
            base_cat = cat_risk * 0.25 + country_risk * 0.10
            score = confidence * score + (1 - confidence) * base_cat

        return max(0.0, min(1.0, score))


def _risk_level_from_score(score: float) -> RiskLevel:
    """Map a continuous risk score to a discrete risk level."""
    if score >= 0.75:
        return RiskLevel.CRITICAL
    if score >= 0.50:
        return RiskLevel.HIGH
    if score >= 0.25:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
