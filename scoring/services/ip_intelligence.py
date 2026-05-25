"""IP intelligence and geolocation risk scoring service.

Provides comprehensive IP analysis including:
  - IP classification (private/public, IPv4/IPv6, loopback, link-local)
  - VPN, proxy, and Tor exit node detection via configurable block lists
  - ASN/ISP risk classification for datacenter and hosting ranges
  - Geo-distance calculation between two locations (Haversine formula)
  - IP velocity tracking — how many distinct users arrive from one IP
  - Composite IP risk scoring across multiple signals
  - Abuse history tracking with time-decayed reputation scores
"""

from __future__ import annotations

import ipaddress
import math
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

import structlog
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

IP_ANALYSES_TOTAL = Counter(
    "ip_intelligence_analyses_total",
    "Total IP addresses analyzed",
    ["ip_version", "classification"],
)

IP_RISK_SCORE_HISTOGRAM = Histogram(
    "ip_intelligence_risk_score",
    "Distribution of IP risk scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

IP_VPN_DETECTED_TOTAL = Counter(
    "ip_intelligence_vpn_detected_total",
    "Total IPs identified as VPN/proxy/Tor",
    ["detection_type"],  # vpn, proxy, tor, datacenter
)

IP_HIGH_VELOCITY_TOTAL = Counter(
    "ip_intelligence_high_velocity_total",
    "Total IPs that exceeded velocity thresholds",
)

IP_GEO_DISTANCE_HISTOGRAM = Histogram(
    "ip_intelligence_geo_distance_km",
    "Geolocation distance between consecutive logins (km)",
    buckets=(0, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000),
)

IP_ABUSE_REPORTS_TOTAL = Counter(
    "ip_intelligence_abuse_reports_total",
    "Total abuse events recorded against IPs",
    ["abuse_type"],
)

IP_ACTIVE_HIGH_RISK = Gauge(
    "ip_intelligence_active_high_risk_count",
    "Number of IPs currently classified as high risk (score >= 0.7)",
)


# ── Enumerations ──────────────────────────────────────────────────────────────


class IPVersion(StrEnum):
    V4 = "ipv4"
    V6 = "ipv6"


class IPClassification(StrEnum):
    PRIVATE = "private"
    PUBLIC = "public"
    LOOPBACK = "loopback"
    LINK_LOCAL = "link_local"
    MULTICAST = "multicast"
    RESERVED = "reserved"


class IPRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AbuseType(StrEnum):
    FRAUD = "fraud"
    BRUTE_FORCE = "brute_force"
    SCRAPING = "scraping"
    SPAM = "spam"
    CREDENTIAL_STUFFING = "credential_stuffing"


# ── Pydantic models ───────────────────────────────────────────────────────────


class GeoLocation(BaseModel):
    """Geographic location derived from IP geolocation lookup."""

    model_config = {"frozen": True}

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    country_code: str = Field(default="XX", max_length=2)
    country_name: str = Field(default="Unknown")
    region: str = Field(default="")
    city: str = Field(default="")
    postal_code: str = Field(default="")
    timezone: str = Field(default="UTC")
    is_eu: bool = False

    @property
    def display(self) -> str:
        parts = [p for p in (self.city, self.region, self.country_code) if p]
        return ", ".join(parts) if parts else "Unknown"


class IPIntelligence(BaseModel):
    """Full intelligence report for a single IP address."""

    model_config = {"frozen": True}

    ip_address: str
    ip_version: IPVersion
    classification: IPClassification
    is_routable: bool
    is_vpn: bool = False
    is_proxy: bool = False
    is_tor: bool = False
    is_datacenter: bool = False
    asn: int | None = None
    asn_org: str | None = None
    isp: str | None = None
    geo: GeoLocation | None = None
    abuse_score: float = Field(default=0.0, ge=0.0, le=1.0)
    abuse_report_count: int = 0
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_suspicious_infrastructure(self) -> bool:
        return self.is_vpn or self.is_proxy or self.is_tor or self.is_datacenter


class IPRiskAssessment(BaseModel):
    """Composite risk assessment for an IP address in context of a user."""

    model_config = {"frozen": False}

    assessment_id: str = Field(default_factory=lambda: str(uuid4()))
    ip_address: str
    user_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: IPRiskLevel
    risk_signals: list[str] = Field(default_factory=list)
    velocity_count: int = 0
    geo_distance_km: float | None = None
    recommended_action: str = "allow"
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "ip_address": self.ip_address,
            "user_id": self.user_id,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "risk_signals": self.risk_signals,
            "velocity_count": self.velocity_count,
            "geo_distance_km": round(self.geo_distance_km, 2) if self.geo_distance_km is not None else None,
            "recommended_action": self.recommended_action,
            "assessed_at": self.assessed_at.isoformat(),
            "details": self.details,
        }


# ── Demo data: known VPN/proxy/Tor CIDRs and datacenter ASNs ─────────────────
# In production these come from threat-intel feeds (e.g. MaxMind, AbuseIPDB,
# IPQualityScore, or internal threat intelligence pipelines).

_DEMO_VPN_CIDRS: list[str] = [
    "185.220.0.0/16",   # Common Tor/VPN range
    "104.21.0.0/16",    # Cloudflare proxy
    "198.11.0.0/16",    # Commercial VPN range
    "45.142.0.0/16",    # VPN provider range
]

_DEMO_TOR_EXIT_IPS: set[str] = {
    "185.220.101.1",
    "185.220.101.2",
    "185.220.101.34",
    "199.87.154.255",
    "171.25.193.77",
}

_DEMO_PROXY_CIDRS: list[str] = [
    "103.21.244.0/22",  # Cloudflare shared proxy
    "162.158.0.0/15",
    "172.64.0.0/13",
]

# ASNs that predominantly serve datacenter/hosting traffic
_DATACENTER_ASNS: set[int] = {
    14061,   # DigitalOcean
    16509,   # Amazon AWS
    15169,   # Google Cloud
    8075,    # Microsoft Azure
    20940,   # Akamai
    13335,   # Cloudflare
    2914,    # NTT America
    3356,    # Lumen/Level3
    6939,    # Hurricane Electric
    36351,   # SoftLayer
    63949,   # Linode/Akamai
    24940,   # Hetzner
    51167,   # Contabo
}

# High-risk country codes (configurable; simplified demo set)
_HIGH_RISK_COUNTRIES: set[str] = {
    "KP",  # North Korea
    "IR",  # Iran
    "CU",  # Cuba
    "SY",  # Syria
}


# ── Velocity tracking ─────────────────────────────────────────────────────────


class _VelocityTracker:
    """Tracks distinct user IDs seen from each IP within rolling windows.

    Uses a deque per IP storing (timestamp, user_id) tuples. Entries older
    than the longest tracked window are evicted lazily.
    """

    _MAX_WINDOW_SECONDS: int = 3600  # 1 hour — longest window we support

    def __init__(self) -> None:
        self._events: dict[str, deque[tuple[float, str]]] = defaultdict(deque)

    def record(self, ip: str, user_id: str) -> None:
        now = time.monotonic()
        self._events[ip].append((now, user_id))
        self._evict(ip, now)

    def count_distinct_users(self, ip: str, window_seconds: int) -> int:
        now = time.monotonic()
        self._evict(ip, now)
        cutoff = now - window_seconds
        seen: set[str] = set()
        for ts, uid in self._events[ip]:
            if ts >= cutoff:
                seen.add(uid)
        return len(seen)

    def _evict(self, ip: str, now: float) -> None:
        cutoff = now - self._MAX_WINDOW_SECONDS
        dq = self._events[ip]
        while dq and dq[0][0] < cutoff:
            dq.popleft()


# ── Abuse reputation store ────────────────────────────────────────────────────


class _AbuseStore:
    """In-process abuse score store with time decay.

    Scores decay by 50% every 24 hours so old signals fade out naturally.
    In production this would be persisted to Redis with ZSET-based TTL.
    """

    _DECAY_HALF_LIFE_HOURS: float = 24.0

    def __init__(self) -> None:
        # ip -> list of (timestamp_unix, raw_weight)
        self._records: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def record_abuse(self, ip: str, abuse_type: AbuseType, weight: float = 0.25) -> None:
        self._records[ip].append((time.time(), weight))
        IP_ABUSE_REPORTS_TOTAL.labels(abuse_type=abuse_type.value).inc()

    def get_score(self, ip: str) -> tuple[float, int]:
        """Return (decayed_abuse_score 0-1, total_report_count)."""
        records = self._records.get(ip)
        if not records:
            return 0.0, 0
        now = time.time()
        half_life_seconds = self._DECAY_HALF_LIFE_HOURS * 3600
        total = 0.0
        for ts, weight in records:
            age_seconds = now - ts
            decay = math.exp(-math.log(2) * age_seconds / half_life_seconds)
            total += weight * decay
        return min(1.0, total), len(records)


# ── IP intelligence service ───────────────────────────────────────────────────


class IPIntelligenceService:
    """Production-grade IP intelligence and geolocation risk scoring service.

    Classifies IP addresses, detects anonymization infrastructure (VPN/proxy/Tor),
    scores IP reputation based on abuse history, and computes velocity and
    geolocation signals to produce a composite fraud risk score.

    Args:
        velocity_threshold_soft: Distinct users per IP per hour that triggers
            a medium-risk velocity signal (default: 10).
        velocity_threshold_hard: Distinct users per IP per hour that triggers
            a high-risk velocity signal (default: 50).
        high_risk_velocity_window: Rolling window in seconds for velocity checks
            (default: 3600 = 1 hour).
        additional_vpn_cidrs: Extra VPN CIDRs to load on top of built-in list.
        additional_tor_ips: Extra Tor exit node IPs to load.
        additional_proxy_cidrs: Extra proxy CIDRs to load.
    """

    def __init__(
        self,
        velocity_threshold_soft: int = 10,
        velocity_threshold_hard: int = 50,
        high_risk_velocity_window: int = 3600,
        additional_vpn_cidrs: list[str] | None = None,
        additional_tor_ips: set[str] | None = None,
        additional_proxy_cidrs: list[str] | None = None,
    ) -> None:
        self._velocity_soft = velocity_threshold_soft
        self._velocity_hard = velocity_threshold_hard
        self._velocity_window = high_risk_velocity_window
        self._velocity = _VelocityTracker()
        self._abuse = _AbuseStore()

        # Compile CIDR networks once at startup for O(n) lookup
        self._vpn_networks = self._compile_networks(
            _DEMO_VPN_CIDRS + (additional_vpn_cidrs or [])
        )
        self._proxy_networks = self._compile_networks(
            _DEMO_PROXY_CIDRS + (additional_proxy_cidrs or [])
        )
        self._tor_exits: set[str] = set(_DEMO_TOR_EXIT_IPS) | (additional_tor_ips or set())

        self._high_risk_ip_cache: dict[str, bool] = {}  # ip -> is_high_risk
        self._active_high_risk: int = 0

        logger.info(
            "ip_intelligence_service_initialized",
            vpn_networks=len(self._vpn_networks),
            proxy_networks=len(self._proxy_networks),
            tor_exits=len(self._tor_exits),
            datacenter_asns=len(_DATACENTER_ASNS),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(self, ip_address: str) -> IPIntelligence:
        """Parse and analyze an IP address, returning full intelligence report.

        This is the primary entry point. It classifies the IP, checks all
        anonymization detection signals, looks up geo/ASN data, and retrieves
        the current abuse reputation score.

        Args:
            ip_address: A valid IPv4 or IPv6 address string.

        Returns:
            IPIntelligence populated with all available signals.

        Raises:
            ValueError: If ip_address is not a valid IP address.
        """
        try:
            parsed = ipaddress.ip_address(ip_address)
        except ValueError as exc:
            raise ValueError(f"Invalid IP address: {ip_address!r}") from exc

        ip_version = IPVersion.V4 if parsed.version == 4 else IPVersion.V6
        classification = self._classify(parsed)
        is_routable = not (
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_multicast
            or parsed.is_reserved
            or parsed.is_unspecified
        )

        # Anonymization detection (only meaningful for public IPs)
        _is_vpn = is_routable and self.is_vpn(ip_address)
        _is_proxy = is_routable and self.is_proxy(ip_address)
        _is_tor = is_routable and self.is_tor(ip_address)

        # ASN classification (demo: deterministic mock for demonstration)
        asn, asn_org, isp = self._lookup_asn_mock(ip_address)
        _is_datacenter = is_routable and self.is_datacenter(ip_address, asn)

        # Geo lookup (demo: deterministic mock)
        geo = self._geo_lookup_mock(ip_address) if is_routable else None

        # Abuse reputation
        abuse_score, abuse_count = self._abuse.get_score(ip_address)

        intel = IPIntelligence(
            ip_address=ip_address,
            ip_version=ip_version,
            classification=classification,
            is_routable=is_routable,
            is_vpn=_is_vpn,
            is_proxy=_is_proxy,
            is_tor=_is_tor,
            is_datacenter=_is_datacenter,
            asn=asn,
            asn_org=asn_org,
            isp=isp,
            geo=geo,
            abuse_score=abuse_score,
            abuse_report_count=abuse_count,
        )

        # Prometheus
        IP_ANALYSES_TOTAL.labels(
            ip_version=ip_version.value,
            classification=classification.value,
        ).inc()

        if _is_vpn:
            IP_VPN_DETECTED_TOTAL.labels(detection_type="vpn").inc()
        if _is_proxy:
            IP_VPN_DETECTED_TOTAL.labels(detection_type="proxy").inc()
        if _is_tor:
            IP_VPN_DETECTED_TOTAL.labels(detection_type="tor").inc()
        if _is_datacenter:
            IP_VPN_DETECTED_TOTAL.labels(detection_type="datacenter").inc()

        logger.debug(
            "ip_analyzed",
            ip=ip_address,
            version=ip_version.value,
            classification=classification.value,
            is_vpn=_is_vpn,
            is_tor=_is_tor,
            is_datacenter=_is_datacenter,
            abuse_score=round(abuse_score, 3),
        )

        return intel

    async def assess_risk(
        self,
        ip_intel: IPIntelligence,
        user_id: str,
        previous_geo: GeoLocation | None = None,
    ) -> IPRiskAssessment:
        """Compute a composite risk score for an IP in context of a user.

        Combines anonymization signals, abuse reputation, velocity, geolocation
        distance, and country-level risk into a single risk score [0, 1].

        Args:
            ip_intel: Intelligence report from :meth:`analyze`.
            user_id: The user ID attempting to authenticate or transact.
            previous_geo: Optional most recent known location for this user —
                used to compute geolocation velocity distance.

        Returns:
            IPRiskAssessment with composite score, level, signals, and action.
        """
        signals: list[str] = []
        score_components: list[float] = []
        details: dict[str, Any] = {}

        # ── Non-routable IPs ──────────────────────────────────────────────────
        if not ip_intel.is_routable:
            # Private/loopback IPs are usually internal traffic — low risk
            assessment = IPRiskAssessment(
                ip_address=ip_intel.ip_address,
                user_id=user_id,
                risk_score=0.05,
                risk_level=IPRiskLevel.LOW,
                risk_signals=["non_routable_ip"],
                recommended_action="allow",
                details={"classification": ip_intel.classification.value},
            )
            IP_RISK_SCORE_HISTOGRAM.observe(0.05)
            return assessment

        # ── Anonymization infrastructure ─────────────────────────────────────
        if ip_intel.is_tor:
            score_components.append(0.85)
            signals.append("tor_exit_node")
        if ip_intel.is_vpn:
            score_components.append(0.55)
            signals.append("known_vpn")
        if ip_intel.is_proxy:
            score_components.append(0.45)
            signals.append("known_proxy")
        if ip_intel.is_datacenter:
            score_components.append(0.30)
            signals.append("datacenter_asn")
            details["asn"] = ip_intel.asn
            details["asn_org"] = ip_intel.asn_org

        # ── Abuse reputation ──────────────────────────────────────────────────
        if ip_intel.abuse_score > 0:
            score_components.append(ip_intel.abuse_score * 0.75)
            signals.append("abuse_history")
            details["abuse_score"] = round(ip_intel.abuse_score, 4)
            details["abuse_report_count"] = ip_intel.abuse_report_count

        # ── Country-level risk ────────────────────────────────────────────────
        if ip_intel.geo and ip_intel.geo.country_code in _HIGH_RISK_COUNTRIES:
            score_components.append(0.70)
            signals.append("high_risk_country")
            details["country"] = ip_intel.geo.country_code

        # ── Velocity check ────────────────────────────────────────────────────
        self._velocity.record(ip_intel.ip_address, user_id)
        velocity = self._velocity.count_distinct_users(
            ip_intel.ip_address, self._velocity_window
        )
        details["velocity_users_per_hour"] = velocity

        if velocity >= self._velocity_hard:
            score_components.append(0.80)
            signals.append("extreme_ip_velocity")
            IP_HIGH_VELOCITY_TOTAL.inc()
        elif velocity >= self._velocity_soft:
            score_components.append(0.45)
            signals.append("high_ip_velocity")

        # ── Geolocation distance ──────────────────────────────────────────────
        geo_distance_km: float | None = None
        if ip_intel.geo and previous_geo:
            geo_distance_km = self.compute_geo_distance(ip_intel.geo, previous_geo)
            details["geo_distance_km"] = round(geo_distance_km, 2)
            IP_GEO_DISTANCE_HISTOGRAM.observe(geo_distance_km)

            if geo_distance_km > 5000:
                score_components.append(0.60)
                signals.append("impossible_travel_distance")
            elif geo_distance_km > 1000:
                score_components.append(0.25)
                signals.append("large_geo_distance")

        # ── Composite score (weighted maximum + mean blend) ───────────────────
        if score_components:
            max_component = max(score_components)
            mean_component = sum(score_components) / len(score_components)
            # Blend: 70% max signal + 30% mean to reward multiple weaker signals
            raw_score = 0.7 * max_component + 0.3 * mean_component
        else:
            raw_score = 0.02  # Baseline for clean public IPs

        risk_score = min(1.0, raw_score)
        risk_level = self._score_to_level(risk_score)
        recommended_action = self._recommend_action(risk_level, signals)

        # Update high-risk gauge
        was_high_risk = self._high_risk_ip_cache.get(ip_intel.ip_address, False)
        is_now_high_risk = risk_score >= 0.7
        if is_now_high_risk and not was_high_risk:
            self._active_high_risk += 1
            IP_ACTIVE_HIGH_RISK.set(self._active_high_risk)
        elif not is_now_high_risk and was_high_risk:
            self._active_high_risk = max(0, self._active_high_risk - 1)
            IP_ACTIVE_HIGH_RISK.set(self._active_high_risk)
        self._high_risk_ip_cache[ip_intel.ip_address] = is_now_high_risk

        IP_RISK_SCORE_HISTOGRAM.observe(risk_score)

        assessment = IPRiskAssessment(
            ip_address=ip_intel.ip_address,
            user_id=user_id,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_signals=signals,
            velocity_count=velocity,
            geo_distance_km=geo_distance_km,
            recommended_action=recommended_action,
            details=details,
        )

        logger.info(
            "ip_risk_assessed",
            ip=ip_intel.ip_address,
            user_id=user_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level.value,
            signals=signals,
            action=recommended_action,
        )

        return assessment

    def compute_geo_distance(self, loc1: GeoLocation, loc2: GeoLocation) -> float:
        """Compute the great-circle distance between two locations (Haversine).

        Args:
            loc1: First geographic location.
            loc2: Second geographic location.

        Returns:
            Distance in kilometres.
        """
        r_km = 6371.0  # Earth's mean radius in km
        lat1 = math.radians(loc1.latitude)
        lat2 = math.radians(loc2.latitude)
        d_lat = math.radians(loc2.latitude - loc1.latitude)
        d_lon = math.radians(loc2.longitude - loc1.longitude)

        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r_km * c

    def is_vpn(self, ip: str) -> bool:
        """Return True if the IP falls within a known VPN CIDR range."""
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(addr in network for network in self._vpn_networks)

    def is_proxy(self, ip: str) -> bool:
        """Return True if the IP falls within a known proxy CIDR range."""
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(addr in network for network in self._proxy_networks)

    def is_tor(self, ip: str) -> bool:
        """Return True if the IP is a known Tor exit node."""
        return ip in self._tor_exits

    def is_datacenter(self, ip: str, asn: int | None = None) -> bool:
        """Return True if the IP originates from a datacenter/hosting ASN.

        Args:
            ip: IP address string (used for future CIDR-based datacenter detection).
            asn: Autonomous System Number from ASN lookup.
        """
        if asn is not None and asn in _DATACENTER_ASNS:
            return True
        # Future: check datacenter CIDR ranges here
        return False

    def get_ip_velocity(self, ip: str, window_seconds: int = 3600) -> int:
        """Return the number of distinct user IDs seen from this IP in the window.

        Args:
            ip: IP address to query.
            window_seconds: Rolling time window length in seconds.

        Returns:
            Distinct user count in the window.
        """
        return self._velocity.count_distinct_users(ip, window_seconds)

    def record_abuse(
        self,
        ip: str,
        abuse_type: AbuseType,
        weight: float = 0.25,
    ) -> None:
        """Record an abuse event against an IP, affecting future risk scoring.

        Args:
            ip: IP address of the abusive actor.
            abuse_type: Nature of the abuse.
            weight: Severity weight in [0, 1]; multiple events accumulate.
        """
        self._abuse.record_abuse(ip, abuse_type, weight)
        logger.warning(
            "ip_abuse_recorded",
            ip=ip,
            abuse_type=abuse_type.value,
            weight=weight,
        )

    def add_vpn_cidr(self, cidr: str) -> None:
        """Dynamically add a VPN CIDR to the detection list at runtime."""
        try:
            network = ipaddress.ip_network(cidr, strict=False)
            self._vpn_networks.append(network)
            logger.info("vpn_cidr_added", cidr=cidr)
        except ValueError as exc:
            logger.error("vpn_cidr_invalid", cidr=cidr, error=str(exc))

    def add_tor_exit(self, ip: str) -> None:
        """Dynamically add a Tor exit node IP to the detection set."""
        self._tor_exits.add(ip)

    @property
    def stats(self) -> dict[str, Any]:
        """Return operational statistics for monitoring and diagnostics."""
        return {
            "vpn_networks_loaded": len(self._vpn_networks),
            "proxy_networks_loaded": len(self._proxy_networks),
            "tor_exits_loaded": len(self._tor_exits),
            "datacenter_asns_loaded": len(_DATACENTER_ASNS),
            "active_high_risk_ips": self._active_high_risk,
            "velocity_window_seconds": self._velocity_window,
            "velocity_threshold_soft": self._velocity_soft,
            "velocity_threshold_hard": self._velocity_hard,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _classify(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> IPClassification:
        if addr.is_loopback:
            return IPClassification.LOOPBACK
        if addr.is_link_local:
            return IPClassification.LINK_LOCAL
        if addr.is_multicast:
            return IPClassification.MULTICAST
        if addr.is_private:
            return IPClassification.PRIVATE
        if addr.is_reserved or addr.is_unspecified:
            return IPClassification.RESERVED
        return IPClassification.PUBLIC

    @staticmethod
    def _compile_networks(
        cidrs: list[str],
    ) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
        networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for cidr in cidrs:
            try:
                networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                logger.warning("invalid_cidr_skipped", cidr=cidr)
        return networks

    @staticmethod
    def _score_to_level(score: float) -> IPRiskLevel:
        if score >= 0.75:
            return IPRiskLevel.CRITICAL
        if score >= 0.50:
            return IPRiskLevel.HIGH
        if score >= 0.25:
            return IPRiskLevel.MEDIUM
        return IPRiskLevel.LOW

    @staticmethod
    def _recommend_action(level: IPRiskLevel, signals: list[str]) -> str:
        if level == IPRiskLevel.CRITICAL:
            return "block"
        if level == IPRiskLevel.HIGH:
            # Tor is non-negotiable; everything else gets stepped-up auth
            if "tor_exit_node" in signals or "extreme_ip_velocity" in signals:
                return "block"
            return "challenge"
        if level == IPRiskLevel.MEDIUM:
            return "review"
        return "allow"

    @staticmethod
    def _lookup_asn_mock(
        ip: str,
    ) -> tuple[int | None, str | None, str | None]:
        """Deterministic ASN mock based on first octet for demonstration.

        Replace with a real MaxMind GeoIP2 or ip-api.com call in production.
        """
        try:
            first_octet = int(ip.split(".")[0])
        except (ValueError, IndexError):
            return None, None, None

        # Map first octet ranges to known datacenter ASNs for demo purposes
        if 104 <= first_octet <= 104:
            return 13335, "Cloudflare, Inc.", "Cloudflare"
        if first_octet in (52, 54):
            return 16509, "Amazon.com, Inc.", "Amazon AWS"
        if first_octet == 34:
            return 15169, "Google LLC", "Google Cloud"
        if first_octet == 185:
            return 24940, "Hetzner Online GmbH", "Hetzner"
        if first_octet == 45:
            return 14061, "DigitalOcean, LLC", "DigitalOcean"
        return None, None, None

    @staticmethod
    def _geo_lookup_mock(ip: str) -> GeoLocation:
        """Deterministic geo mock for demonstration.

        Replace with MaxMind GeoIP2 City database or an external geo API.
        """
        try:
            first_octet = int(ip.split(".")[0])
            _second_octet = int(ip.split(".")[1]) if "." in ip else 0
        except (ValueError, IndexError):
            first_octet, _second_octet = 1, 1

        # Deterministic mapping for consistent unit testing
        if first_octet < 50:
            return GeoLocation(
                latitude=37.7749, longitude=-122.4194,
                country_code="US", country_name="United States",
                region="California", city="San Francisco",
                timezone="America/Los_Angeles", is_eu=False,
            )
        if first_octet < 100:
            return GeoLocation(
                latitude=51.5074, longitude=-0.1278,
                country_code="GB", country_name="United Kingdom",
                region="England", city="London",
                timezone="Europe/London", is_eu=False,
            )
        if first_octet < 150:
            return GeoLocation(
                latitude=48.8566, longitude=2.3522,
                country_code="FR", country_name="France",
                region="Île-de-France", city="Paris",
                timezone="Europe/Paris", is_eu=True,
            )
        if first_octet < 185:
            return GeoLocation(
                latitude=35.6762, longitude=139.6503,
                country_code="JP", country_name="Japan",
                region="Tokyo", city="Tokyo",
                timezone="Asia/Tokyo", is_eu=False,
            )
        # 185.x — common Tor/Eastern European ranges in demo
        return GeoLocation(
            latitude=50.4501, longitude=30.5234,
            country_code="UA", country_name="Ukraine",
            region="Kyiv City", city="Kyiv",
            timezone="Europe/Kyiv", is_eu=False,
        )
