"""Geographic risk scoring service for fraud detection.

Provides location-based risk analysis including:
  - Distance calculation between consecutive transactions (Haversine)
  - Impossible travel detection (transactions in distant locations within
    unrealistically short time windows)
  - Configurable country risk tiers with per-tier scoring
  - Cross-border transaction detection
  - Timezone anomaly detection (transaction timestamp vs. local timezone)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

import structlog
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

__all__ = [
    "GeoRiskLevel",
    "GeoLocation",
    "TransactionGeoData",
    "ImpossibleTravelResult",
    "GeoRiskAssessment",
    "GeoRiskService",
    "get_geo_risk_service",
]

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

GEO_RISK_ASSESSMENTS_TOTAL = Counter(
    "geo_risk_assessments_total",
    "Total geographic risk assessments performed",
    ["risk_level"],
)

IMPOSSIBLE_TRAVEL_DETECTIONS_TOTAL = Counter(
    "geo_risk_impossible_travel_detections_total",
    "Total impossible travel events detected",
)

CROSS_BORDER_DETECTIONS_TOTAL = Counter(
    "geo_risk_cross_border_detections_total",
    "Total cross-border transactions detected",
)

TIMEZONE_ANOMALIES_TOTAL = Counter(
    "geo_risk_timezone_anomalies_total",
    "Total timezone anomalies detected",
)

GEO_DISTANCE_HISTOGRAM = Histogram(
    "geo_risk_distance_km",
    "Distribution of distances between consecutive transactions (km)",
    buckets=(0, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000),
)

GEO_RISK_SCORE_HISTOGRAM = Histogram(
    "geo_risk_score_distribution",
    "Distribution of geographic risk scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ── Country risk tiers ───────────────────────────────────────────────────────
# Tier 1: Sanctioned / highest risk
# Tier 2: FATF grey-list / elevated risk
# Tier 3: Moderate risk
# Tier 4: Low risk (default for unlisted countries)

_DEFAULT_COUNTRY_RISK_TIERS: dict[int, frozenset[str]] = {
    1: frozenset({
        "KP",  # North Korea
        "IR",  # Iran
        "SY",  # Syria
        "SS",  # South Sudan
    }),
    2: frozenset({
        "AF",  # Afghanistan
        "MM",  # Myanmar
        "SO",  # Somalia
        "YE",  # Yemen
        "SD",  # Sudan
        "LY",  # Libya
        "CU",  # Cuba
        "VE",  # Venezuela
        "RU",  # Russia
        "BY",  # Belarus
    }),
    3: frozenset({
        "NG",  # Nigeria
        "PK",  # Pakistan
        "HT",  # Haiti
        "JM",  # Jamaica
        "PA",  # Panama
        "TT",  # Trinidad & Tobago
        "CD",  # DR Congo
        "MZ",  # Mozambique
        "CM",  # Cameroon
        "TZ",  # Tanzania
    }),
}

_TIER_SCORES: dict[int, float] = {
    1: 1.0,
    2: 0.75,
    3: 0.45,
    4: 0.10,  # default tier for unlisted countries
}

# ── Timezone offsets (approximate UTC offset in hours per country) ────────────
# Used for timezone anomaly detection. In production, use a proper tz database.

_COUNTRY_UTC_OFFSETS: dict[str, list[float]] = {
    "US": [-10, -9, -8, -7, -6, -5, -4],
    "CA": [-8, -7, -6, -5, -4, -3.5],
    "GB": [0],
    "DE": [1],
    "FR": [1],
    "JP": [9],
    "AU": [8, 9.5, 10, 10.5, 11],
    "IN": [5.5],
    "CN": [8],
    "BR": [-5, -4, -3, -2],
    "RU": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "KR": [9],
    "SG": [8],
    "AE": [4],
    "SA": [3],
    "NG": [1],
    "ZA": [2],
    "MX": [-8, -7, -6, -5],
    "AR": [-3],
    "KP": [9],
    "IR": [3.5],
    "SY": [2],
    "NZ": [12, 13],
}

# Maximum plausible speed for a human traveller (commercial flight ~900 km/h,
# with buffer for supersonic / private jet edge cases).
_MAX_PLAUSIBLE_SPEED_KMH: float = 1200.0

# Minimum time gap (seconds) below which we skip impossible-travel checks
# (to avoid false positives from near-simultaneous duplicate events).
_MIN_TIME_GAP_SECONDS: float = 60.0

# Earth's mean radius in kilometres.
_EARTH_RADIUS_KM: float = 6371.0


# ── Enumerations ──────────────────────────────────────────────────────────────


class GeoRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Pydantic models ──────────────────────────────────────────────────────────


class GeoLocation(BaseModel):
    """A geographic coordinate with optional country metadata."""

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    country_code: str = Field(default="XX", max_length=2)

    model_config = {"frozen": True}


class TransactionGeoData(BaseModel):
    """Geographic data attached to a single transaction."""

    transaction_id: str = Field(default_factory=lambda: str(uuid4()))
    location: GeoLocation
    timestamp: datetime
    user_id: str = ""


class ImpossibleTravelResult(BaseModel):
    """Result of an impossible travel check between two transactions."""

    is_impossible: bool = False
    distance_km: float = 0.0
    time_gap_seconds: float = 0.0
    required_speed_kmh: float = 0.0
    max_plausible_speed_kmh: float = _MAX_PLAUSIBLE_SPEED_KMH


class GeoRiskAssessment(BaseModel):
    """Full geographic risk assessment for a transaction."""

    assessment_id: str = Field(default_factory=lambda: str(uuid4()))
    transaction_id: str = ""
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: GeoRiskLevel
    country_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    country_risk_tier: int = 4
    is_cross_border: bool = False
    impossible_travel: ImpossibleTravelResult | None = None
    timezone_anomaly: bool = False
    risk_signals: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


# ── Service ──────────────────────────────────────────────────────────────────


class GeoRiskService:
    """Geographic risk scoring and impossible travel detection service.

    All public methods are synchronous and stateless aside from configuration
    provided at initialisation time.

    Args:
        country_risk_tiers: Mapping of tier number (1-4) to sets of ISO 3166-1
            alpha-2 country codes. If ``None``, built-in defaults are used.
        tier_scores: Mapping of tier number to risk score [0, 1]. Falls back
            to the built-in ``_TIER_SCORES`` if ``None``.
        max_plausible_speed_kmh: Maximum speed (km/h) considered physically
            plausible for a traveller. Defaults to 1200 km/h.
    """

    def __init__(
        self,
        country_risk_tiers: dict[int, frozenset[str]] | None = None,
        tier_scores: dict[int, float] | None = None,
        max_plausible_speed_kmh: float = _MAX_PLAUSIBLE_SPEED_KMH,
    ) -> None:
        self._tiers = country_risk_tiers or _DEFAULT_COUNTRY_RISK_TIERS
        self._tier_scores = tier_scores or _TIER_SCORES
        self._max_speed = max_plausible_speed_kmh
        self._log = logger.bind(service="GeoRiskService")

    # ── Public API ────────────────────────────────────────────────────────────

    def assess(
        self,
        transaction: TransactionGeoData,
        previous_transaction: TransactionGeoData | None = None,
    ) -> GeoRiskAssessment:
        """Produce a full geographic risk assessment for *transaction*.

        When *previous_transaction* is supplied the service also checks for
        impossible travel and cross-border movement.
        """
        signals: list[str] = []
        details: dict[str, Any] = {}
        score_components: list[float] = []

        # ── Country risk ─────────────────────────────────────────────────────
        country = transaction.location.country_code.upper()
        tier = self._get_country_tier(country)
        country_score = self._tier_scores.get(tier, self._tier_scores.get(4, 0.10))
        score_components.append(country_score)
        details["country"] = country
        details["country_tier"] = tier

        if tier <= 2:
            signals.append("high_risk_country")

        # ── Cross-border detection ───────────────────────────────────────────
        is_cross_border = False
        if previous_transaction is not None:
            prev_country = previous_transaction.location.country_code.upper()
            if prev_country != country:
                is_cross_border = True
                signals.append("cross_border_transaction")
                CROSS_BORDER_DETECTIONS_TOTAL.inc()
                # Cross-border adds a small risk bump
                score_components.append(0.20)
                details["previous_country"] = prev_country

        # ── Impossible travel ────────────────────────────────────────────────
        impossible_travel: ImpossibleTravelResult | None = None
        if previous_transaction is not None:
            impossible_travel = self.check_impossible_travel(
                transaction, previous_transaction
            )
            if impossible_travel.distance_km > 0:
                GEO_DISTANCE_HISTOGRAM.observe(impossible_travel.distance_km)
                details["distance_km"] = round(impossible_travel.distance_km, 2)
                details["time_gap_seconds"] = impossible_travel.time_gap_seconds

            if impossible_travel.is_impossible:
                signals.append("impossible_travel")
                score_components.append(0.90)
                IMPOSSIBLE_TRAVEL_DETECTIONS_TOTAL.inc()

        # ── Timezone anomaly ─────────────────────────────────────────────────
        tz_anomaly = self.detect_timezone_anomaly(
            transaction.timestamp, country
        )
        if tz_anomaly:
            signals.append("timezone_anomaly")
            score_components.append(0.30)
            TIMEZONE_ANOMALIES_TOTAL.inc()

        # ── Composite score ──────────────────────────────────────────────────
        if score_components:
            max_component = max(score_components)
            mean_component = sum(score_components) / len(score_components)
            raw_score = 0.7 * max_component + 0.3 * mean_component
        else:
            raw_score = 0.05

        risk_score = round(min(1.0, raw_score), 4)
        risk_level = self._score_to_level(risk_score)

        GEO_RISK_ASSESSMENTS_TOTAL.labels(risk_level=risk_level.value).inc()
        GEO_RISK_SCORE_HISTOGRAM.observe(risk_score)

        self._log.info(
            "geo_risk_assessed",
            transaction_id=transaction.transaction_id,
            country=country,
            risk_score=risk_score,
            risk_level=risk_level.value,
            signals=signals,
        )

        return GeoRiskAssessment(
            transaction_id=transaction.transaction_id,
            risk_score=risk_score,
            risk_level=risk_level,
            country_risk_score=country_score,
            country_risk_tier=tier,
            is_cross_border=is_cross_border,
            impossible_travel=impossible_travel,
            timezone_anomaly=tz_anomaly,
            risk_signals=signals,
            details=details,
        )

    def calculate_distance(self, loc1: GeoLocation, loc2: GeoLocation) -> float:
        """Compute the great-circle distance between two locations (Haversine).

        Returns:
            Distance in kilometres.
        """
        lat1 = math.radians(loc1.latitude)
        lat2 = math.radians(loc2.latitude)
        d_lat = math.radians(loc2.latitude - loc1.latitude)
        d_lon = math.radians(loc2.longitude - loc1.longitude)

        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return _EARTH_RADIUS_KM * c

    def check_impossible_travel(
        self,
        current: TransactionGeoData,
        previous: TransactionGeoData,
    ) -> ImpossibleTravelResult:
        """Determine whether travel between *previous* and *current* is physically impossible.

        Compares the required travel speed against ``max_plausible_speed_kmh``.
        If the time gap is below ``_MIN_TIME_GAP_SECONDS``, the check is skipped
        to avoid false positives from near-simultaneous events.
        """
        distance_km = self.calculate_distance(
            current.location, previous.location
        )

        # Ensure both timestamps are offset-aware before subtracting.
        cur_ts = current.timestamp if current.timestamp.tzinfo else current.timestamp.replace(tzinfo=UTC)
        prev_ts = previous.timestamp if previous.timestamp.tzinfo else previous.timestamp.replace(tzinfo=UTC)

        time_gap = abs((cur_ts - prev_ts).total_seconds())

        if time_gap < _MIN_TIME_GAP_SECONDS:
            return ImpossibleTravelResult(
                is_impossible=False,
                distance_km=distance_km,
                time_gap_seconds=time_gap,
                required_speed_kmh=0.0,
                max_plausible_speed_kmh=self._max_speed,
            )

        required_speed = distance_km / (time_gap / 3600.0)
        is_impossible = required_speed > self._max_speed

        return ImpossibleTravelResult(
            is_impossible=is_impossible,
            distance_km=round(distance_km, 2),
            time_gap_seconds=time_gap,
            required_speed_kmh=round(required_speed, 2),
            max_plausible_speed_kmh=self._max_speed,
        )

    def get_country_risk_score(self, country_code: str) -> float:
        """Return the risk score for *country_code* based on its tier."""
        tier = self._get_country_tier(country_code.upper())
        return self._tier_scores.get(tier, self._tier_scores.get(4, 0.10))

    def is_cross_border(self, country_a: str, country_b: str) -> bool:
        """Return True if *country_a* and *country_b* are different countries."""
        return country_a.upper().strip() != country_b.upper().strip()

    def detect_timezone_anomaly(
        self,
        transaction_time: datetime,
        country_code: str,
    ) -> bool:
        """Detect whether *transaction_time* is anomalous for *country_code*.

        A transaction is considered anomalous when the local hour (derived from
        the country's expected UTC offset) falls outside normal waking hours
        (04:00 – 23:59). If the country has multiple time zones, the check
        passes (no anomaly) if *any* zone is within waking hours.

        Returns ``True`` when an anomaly is detected.
        """
        code = country_code.upper()
        offsets = _COUNTRY_UTC_OFFSETS.get(code)
        if offsets is None:
            # Unknown country — cannot determine anomaly; skip.
            return False

        # Ensure offset-aware
        ts = transaction_time if transaction_time.tzinfo else transaction_time.replace(tzinfo=UTC)

        for offset_hours in offsets:
            local_time = ts + timedelta(hours=offset_hours)
            local_hour = local_time.hour
            if 4 <= local_hour < 24:
                return False  # at least one zone is within waking hours

        return True

    def detect_impossible_travel_batch(
        self,
        transactions: list[TransactionGeoData],
    ) -> list[ImpossibleTravelResult]:
        """Check a chronologically-ordered list of transactions for impossible travel.

        Returns one ``ImpossibleTravelResult`` per consecutive pair.
        """
        results: list[ImpossibleTravelResult] = []
        sorted_txns = sorted(transactions, key=lambda t: t.timestamp)
        for i in range(1, len(sorted_txns)):
            results.append(
                self.check_impossible_travel(sorted_txns[i], sorted_txns[i - 1])
            )
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_country_tier(self, country_code: str) -> int:
        """Return the risk tier (1-4) for *country_code*."""
        code = country_code.upper()
        for tier, countries in self._tiers.items():
            if code in countries:
                return tier
        return 4  # default — low risk

    @staticmethod
    def _score_to_level(score: float) -> GeoRiskLevel:
        if score >= 0.75:
            return GeoRiskLevel.CRITICAL
        if score >= 0.50:
            return GeoRiskLevel.HIGH
        if score >= 0.25:
            return GeoRiskLevel.MEDIUM
        return GeoRiskLevel.LOW


# ── Module-level singleton ────────────────────────────────────────────────────

_default_service: GeoRiskService | None = None


def get_geo_risk_service() -> GeoRiskService:
    """Return the module-level singleton, creating it on first call."""
    global _default_service
    if _default_service is None:
        _default_service = GeoRiskService()
    return _default_service
