"""Behavioral biometrics analysis engine for user pattern profiling.

Builds per-user behavioral baselines from historical transactions and scores
new activity against established patterns.  Tracks:

  - Spending patterns by time-of-day and day-of-week
  - Typical transaction amounts (mean, stddev, percentiles)
  - Merchant category preferences
  - Payment method usage distributions
  - Deviation from baseline via composite scoring

All profiles adapt over time using exponential moving averages (EMA) so that
recent behavior is weighted more heavily than stale history.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_HOURS_IN_DAY = 24
_DAYS_IN_WEEK = 7
_DEFAULT_EMA_ALPHA = 0.1  # Weight for new observations (higher = faster adapt)
_MIN_OBSERVATIONS = 5  # Minimum transactions before scoring is meaningful
_MAX_PROFILES = 100_000  # Hard cap on tracked users


# ── Pydantic models ─────────────────────────────────────────────────────────


class TransactionRecord(BaseModel):
    """Minimal transaction data needed for behavioral profiling."""

    user_id: str
    amount: float = Field(..., ge=0.0)
    timestamp: datetime
    merchant_category: str = "unknown"
    payment_method: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeviationResult(BaseModel):
    """Result of scoring a transaction against a user's behavioral baseline."""

    user_id: str
    composite_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall deviation score (0 = normal, 1 = extreme outlier)",
    )
    amount_deviation: float = Field(
        ..., ge=0.0, le=1.0,
        description="How unusual the amount is relative to baseline",
    )
    time_deviation: float = Field(
        ..., ge=0.0, le=1.0,
        description="How unusual the time-of-day / day-of-week is",
    )
    category_deviation: float = Field(
        ..., ge=0.0, le=1.0,
        description="How unusual the merchant category is for this user",
    )
    payment_method_deviation: float = Field(
        ..., ge=0.0, le=1.0,
        description="How unusual the payment method is for this user",
    )
    observations: int = Field(
        ..., ge=0,
        description="Number of historical observations in the profile",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra context for explainability",
    )


class UserProfileSummary(BaseModel):
    """Human-readable summary of a user's behavioral profile."""

    user_id: str
    observation_count: int
    avg_amount: float
    stddev_amount: float
    preferred_hours: list[int] = Field(
        default_factory=list,
        description="Top 3 hours of the day by transaction frequency",
    )
    preferred_days: list[int] = Field(
        default_factory=list,
        description="Top 3 days of the week (0=Mon) by frequency",
    )
    top_categories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top merchant categories with proportion",
    )
    top_payment_methods: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top payment methods with proportion",
    )


# ── Internal data structures ─────────────────────────────────────────────────


@dataclass
class _EMAAccumulator:
    """Tracks an exponential moving average and variance."""

    mean: float = 0.0
    variance: float = 0.0
    count: int = 0
    alpha: float = _DEFAULT_EMA_ALPHA

    def update(self, value: float) -> None:
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.variance = 0.0
            return
        diff = value - self.mean
        self.mean += self.alpha * diff
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * diff * diff)

    @property
    def stddev(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


@dataclass
class _UserProfile:
    """Internal mutable profile for a single user."""

    user_id: str

    # Amount EMA
    amount_ema: _EMAAccumulator = field(default_factory=_EMAAccumulator)

    # Time-of-day distribution (24 buckets, EMA-smoothed counts)
    hour_counts: list[float] = field(
        default_factory=lambda: [0.0] * _HOURS_IN_DAY,
    )

    # Day-of-week distribution (7 buckets, EMA-smoothed counts)
    dow_counts: list[float] = field(
        default_factory=lambda: [0.0] * _DAYS_IN_WEEK,
    )

    # Merchant category distribution: category -> EMA-smoothed count
    category_counts: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # Payment method distribution: method -> EMA-smoothed count
    payment_method_counts: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # Total observation count
    observation_count: int = 0

    # EMA alpha (stored per profile so it can be tuned per user if desired)
    alpha: float = _DEFAULT_EMA_ALPHA


# ── BehavioralAnalytics engine ───────────────────────────────────────────────


class BehavioralAnalytics:
    """Thread-safe behavioral analytics engine.

    Maintains per-user spending profiles and scores new transactions against
    established baselines.  Designed to be instantiated once and registered
    on ``app.state`` during application lifespan startup.

    Args:
        alpha: EMA smoothing factor (0 < alpha <= 1). Higher values adapt
            faster to recent behavior.
        max_profiles: Hard limit on tracked user profiles; oldest profiles
            are evicted when this cap is reached.
        min_observations: Minimum observations before deviation scores are
            considered reliable.
    """

    def __init__(
        self,
        alpha: float = _DEFAULT_EMA_ALPHA,
        max_profiles: int = _MAX_PROFILES,
        min_observations: int = _MIN_OBSERVATIONS,
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        self._alpha = alpha
        self._max_profiles = max_profiles
        self._min_observations = min_observations
        self._lock = Lock()
        self._profiles: dict[str, _UserProfile] = {}
        # Track insertion order for eviction
        self._insertion_order: list[str] = []

        logger.info(
            "behavioral_analytics.initialized",
            alpha=alpha,
            max_profiles=max_profiles,
            min_observations=min_observations,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, txn: TransactionRecord) -> None:
        """Ingest a transaction and update the user's behavioral profile.

        This method is idempotent with respect to profile structure — calling
        it multiple times with the same transaction simply updates the EMA
        further.
        """
        with self._lock:
            profile = self._get_or_create_profile(txn.user_id)
            self._update_profile(profile, txn)

        logger.debug(
            "behavioral_analytics.ingested",
            user_id=txn.user_id,
            observations=profile.observation_count,
        )

    def score(self, txn: TransactionRecord) -> DeviationResult:
        """Score a transaction against the user's established behavioral baseline.

        Does **not** update the profile — call :meth:`ingest` separately if
        the transaction should be incorporated into the baseline after scoring.

        Returns a :class:`DeviationResult` with per-dimension deviation scores
        and a composite score.
        """
        with self._lock:
            profile = self._profiles.get(txn.user_id)

        if profile is None or profile.observation_count < self._min_observations:
            return DeviationResult(
                user_id=txn.user_id,
                composite_score=0.0,
                amount_deviation=0.0,
                time_deviation=0.0,
                category_deviation=0.0,
                payment_method_deviation=0.0,
                observations=profile.observation_count if profile else 0,
                details={"reason": "insufficient_history"},
            )

        amount_dev = self._score_amount(profile, txn.amount)
        time_dev = self._score_time(profile, txn.timestamp)
        cat_dev = self._score_category(profile, txn.merchant_category)
        pm_dev = self._score_payment_method(profile, txn.payment_method)

        # Weighted composite: amount is most important, then time, then category/payment
        composite = (
            0.40 * amount_dev
            + 0.25 * time_dev
            + 0.20 * cat_dev
            + 0.15 * pm_dev
        )
        composite = round(min(composite, 1.0), 4)

        return DeviationResult(
            user_id=txn.user_id,
            composite_score=composite,
            amount_deviation=round(amount_dev, 4),
            time_deviation=round(time_dev, 4),
            category_deviation=round(cat_dev, 4),
            payment_method_deviation=round(pm_dev, 4),
            observations=profile.observation_count,
            details={
                "amount_mean": round(profile.amount_ema.mean, 2),
                "amount_stddev": round(profile.amount_ema.stddev, 2),
            },
        )

    def score_and_ingest(self, txn: TransactionRecord) -> DeviationResult:
        """Score and then ingest in a single atomic operation.

        This is the recommended method for the online scoring pipeline — it
        scores against the current baseline and then updates it.
        """
        result = self.score(txn)
        self.ingest(txn)
        return result

    def get_profile_summary(self, user_id: str) -> UserProfileSummary | None:
        """Return a human-readable summary of the user's behavioral profile.

        Returns None if no profile exists for the given user.
        """
        with self._lock:
            profile = self._profiles.get(user_id)

        if profile is None:
            return None

        # Top hours
        hour_indexed = list(enumerate(profile.hour_counts))
        hour_indexed.sort(key=lambda x: x[1], reverse=True)
        preferred_hours = [h for h, _ in hour_indexed[:3] if hour_indexed[0][1] > 0]

        # Top days
        dow_indexed = list(enumerate(profile.dow_counts))
        dow_indexed.sort(key=lambda x: x[1], reverse=True)
        preferred_days = [d for d, _ in dow_indexed[:3] if dow_indexed[0][1] > 0]

        # Top categories
        total_cat = sum(profile.category_counts.values()) or 1.0
        sorted_cats = sorted(
            profile.category_counts.items(), key=lambda kv: kv[1], reverse=True,
        )
        top_categories = [
            {"category": cat, "proportion": round(cnt / total_cat, 4)}
            for cat, cnt in sorted_cats[:5]
        ]

        # Top payment methods
        total_pm = sum(profile.payment_method_counts.values()) or 1.0
        sorted_pms = sorted(
            profile.payment_method_counts.items(), key=lambda kv: kv[1], reverse=True,
        )
        top_payment_methods = [
            {"method": pm, "proportion": round(cnt / total_pm, 4)}
            for pm, cnt in sorted_pms[:5]
        ]

        return UserProfileSummary(
            user_id=user_id,
            observation_count=profile.observation_count,
            avg_amount=round(profile.amount_ema.mean, 2),
            stddev_amount=round(profile.amount_ema.stddev, 2),
            preferred_hours=preferred_hours,
            preferred_days=preferred_days,
            top_categories=top_categories,
            top_payment_methods=top_payment_methods,
        )

    def get_all_user_ids(self) -> list[str]:
        """Return a list of all tracked user IDs."""
        with self._lock:
            return list(self._profiles.keys())

    def remove_profile(self, user_id: str) -> bool:
        """Remove a user's profile. Returns True if it existed."""
        with self._lock:
            if user_id in self._profiles:
                del self._profiles[user_id]
                if user_id in self._insertion_order:
                    self._insertion_order.remove(user_id)
                return True
            return False

    @property
    def profile_count(self) -> int:
        """Number of user profiles currently tracked."""
        return len(self._profiles)

    # ── Internal: profile management ──────────────────────────────────────────

    def _get_or_create_profile(self, user_id: str) -> _UserProfile:
        """Return the profile for *user_id*, creating it if needed.

        Must be called under ``self._lock``.
        """
        if user_id in self._profiles:
            return self._profiles[user_id]

        # Evict oldest profiles if at capacity
        if len(self._profiles) >= self._max_profiles:
            evict_count = self._max_profiles // 10
            for _ in range(evict_count):
                if self._insertion_order:
                    old_uid = self._insertion_order.pop(0)
                    self._profiles.pop(old_uid, None)
            logger.warning(
                "behavioral_analytics.profile_eviction",
                evicted=evict_count,
            )

        profile = _UserProfile(
            user_id=user_id,
            alpha=self._alpha,
            amount_ema=_EMAAccumulator(alpha=self._alpha),
        )
        self._profiles[user_id] = profile
        self._insertion_order.append(user_id)
        return profile

    def _update_profile(self, profile: _UserProfile, txn: TransactionRecord) -> None:
        """Update all dimensions of *profile* with data from *txn*.

        Must be called under ``self._lock``.
        """
        profile.observation_count += 1

        # 1. Amount EMA
        profile.amount_ema.update(txn.amount)

        # 2. Time-of-day (EMA-smoothed distribution)
        hour = txn.timestamp.hour
        for h in range(_HOURS_IN_DAY):
            if h == hour:
                profile.hour_counts[h] += 1.0
            else:
                profile.hour_counts[h] *= (1 - profile.alpha)

        # 3. Day-of-week
        dow = txn.timestamp.weekday()  # 0=Monday
        for d in range(_DAYS_IN_WEEK):
            if d == dow:
                profile.dow_counts[d] += 1.0
            else:
                profile.dow_counts[d] *= (1 - profile.alpha)

        # 4. Merchant category
        for cat in profile.category_counts:
            profile.category_counts[cat] *= (1 - profile.alpha)
        profile.category_counts[txn.merchant_category] = (
            profile.category_counts.get(txn.merchant_category, 0.0) + 1.0
        )

        # 5. Payment method
        for pm in profile.payment_method_counts:
            profile.payment_method_counts[pm] *= (1 - profile.alpha)
        profile.payment_method_counts[txn.payment_method] = (
            profile.payment_method_counts.get(txn.payment_method, 0.0) + 1.0
        )

    # ── Internal: deviation scoring ──────────────────────────────────────────

    @staticmethod
    def _score_amount(profile: _UserProfile, amount: float) -> float:
        """Score how unusual *amount* is relative to the EMA baseline.

        Uses a z-score approach: deviation = sigmoid(|z| - 2) so that
        amounts within ~2 stddevs are considered normal.
        """
        ema = profile.amount_ema
        if ema.stddev < 1e-9:
            # No variance established yet — check raw difference
            if ema.mean < 1e-9:
                return 0.0
            ratio = abs(amount - ema.mean) / max(ema.mean, 1.0)
            return min(ratio / 5.0, 1.0)

        z = abs(amount - ema.mean) / ema.stddev
        # Sigmoid mapping: values below z=2 map to near-zero
        return _sigmoid(z - 2.0)

    @staticmethod
    def _score_time(profile: _UserProfile, ts: datetime) -> float:
        """Score how unusual the time-of-day and day-of-week are.

        Combines hour and day-of-week distributions.  A transaction at a
        time the user has never transacted before gets a high score.
        """
        hour = ts.hour
        dow = ts.weekday()

        hour_total = sum(profile.hour_counts) or 1.0
        hour_proportion = profile.hour_counts[hour] / hour_total

        dow_total = sum(profile.dow_counts) or 1.0
        dow_proportion = profile.dow_counts[dow] / dow_total

        # Invert proportions: low proportion = high deviation
        hour_dev = 1.0 - min(hour_proportion * _HOURS_IN_DAY, 1.0)
        dow_dev = 1.0 - min(dow_proportion * _DAYS_IN_WEEK, 1.0)

        return 0.6 * hour_dev + 0.4 * dow_dev

    @staticmethod
    def _score_category(profile: _UserProfile, category: str) -> float:
        """Score how unusual *category* is for this user."""
        total = sum(profile.category_counts.values()) or 1.0
        proportion = profile.category_counts.get(category, 0.0) / total

        if proportion < 1e-9:
            return 1.0  # Never-seen category is fully anomalous
        # Scale: proportion 1.0 -> 0.0 deviation, proportion ~0 -> ~1.0
        return max(0.0, 1.0 - proportion * len(profile.category_counts))

    @staticmethod
    def _score_payment_method(profile: _UserProfile, method: str) -> float:
        """Score how unusual *method* is for this user."""
        total = sum(profile.payment_method_counts.values()) or 1.0
        proportion = profile.payment_method_counts.get(method, 0.0) / total

        if proportion < 1e-9:
            return 1.0  # Never-seen method
        return max(0.0, 1.0 - proportion * len(profile.payment_method_counts))


# ── Module-level helpers ─────────────────────────────────────────────────────


def _sigmoid(x: float) -> float:
    """Standard sigmoid function clamped to [0, 1]."""
    if x < -10:
        return 0.0
    if x > 10:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


__all__ = ["BehavioralAnalytics", "DeviationResult", "TransactionRecord", "UserProfileSummary"]
