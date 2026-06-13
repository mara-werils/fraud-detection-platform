"""Device fingerprinting and risk scoring engine.

Computes stable device fingerprints from browser/client signals, tracks
device history per user, detects anomalies (new devices, shared devices,
spoofing indicators), and maintains a reputation score for each device.
All storage is in-memory by default, with a Redis-compatible interface
for production deployment.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

import structlog
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, field_validator, model_validator

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_FINGERPRINT_COMPUTED = Counter(
    "device_fingerprint_computed_total",
    "Total number of device fingerprints computed",
)
_RISK_ASSESSED = Counter(
    "device_risk_assessed_total",
    "Total number of device risk assessments performed",
    ["risk_level"],
)
_ANOMALY_DETECTED = Counter(
    "device_anomaly_detected_total",
    "Total anomalies detected during device assessment",
    ["anomaly_type"],
)
_ASSESSMENT_LATENCY = Histogram(
    "device_risk_assessment_latency_seconds",
    "Time taken to perform a full device risk assessment",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)
_KNOWN_DEVICES = Gauge(
    "device_known_devices_total",
    "Total number of unique device fingerprints tracked",
)
_SHARED_DEVICES = Gauge(
    "device_shared_devices_total",
    "Number of devices associated with more than one user",
)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(StrEnum):
    NEW_DEVICE = "new_device"
    SHARED_DEVICE = "shared_device"
    IMPOSSIBLE_ATTRIBUTES = "impossible_attributes"
    HEADLESS_BROWSER = "headless_browser"
    BOT_SIGNALS = "bot_signals"
    TIMEZONE_MISMATCH = "timezone_mismatch"
    RAPID_USER_SWITCH = "rapid_user_switch"
    ATTRIBUTE_TAMPERING = "attribute_tampering"
    EMULATOR_DETECTED = "emulator_detected"
    VPN_OR_PROXY = "vpn_or_proxy"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class BrowserSignals(BaseModel):
    """Raw browser/client signals collected on the client side."""

    user_agent: str = Field(..., min_length=1, max_length=1024)
    screen_width: int = Field(..., ge=0, le=32768)
    screen_height: int = Field(..., ge=0, le=32768)
    color_depth: int = Field(default=24, ge=1, le=64)
    pixel_ratio: float = Field(default=1.0, ge=0.1, le=10.0)
    timezone_offset: int = Field(default=0, ge=-840, le=840)  # minutes from UTC
    timezone_name: str = Field(default="", max_length=128)
    language: str = Field(default="en", max_length=64)
    languages: list[str] = Field(default_factory=list)
    platform: str = Field(default="", max_length=128)
    hardware_concurrency: int = Field(default=4, ge=1, le=256)
    device_memory: float | None = Field(default=None, ge=0.125, le=1024.0)
    max_touch_points: int = Field(default=0, ge=0, le=256)
    plugins: list[str] = Field(default_factory=list)
    canvas_hash: str = Field(default="", max_length=128)
    webgl_vendor: str = Field(default="", max_length=256)
    webgl_renderer: str = Field(default="", max_length=256)
    audio_hash: str = Field(default="", max_length=128)
    fonts_hash: str = Field(default="", max_length=128)
    do_not_track: str | None = Field(default=None)
    cookie_enabled: bool = Field(default=True)
    local_storage_available: bool = Field(default=True)
    session_storage_available: bool = Field(default=True)
    indexed_db_available: bool = Field(default=True)
    # Network-layer signals (injected server-side)
    ip_address: str = Field(default="", max_length=45)
    ip_is_proxy: bool = Field(default=False)
    ip_is_vpn: bool = Field(default=False)
    ip_is_tor: bool = Field(default=False)
    ip_country: str = Field(default="", max_length=2)
    ip_asn: str = Field(default="", max_length=64)

    @field_validator("user_agent")
    @classmethod
    def user_agent_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_agent must not be blank")
        return v

    @field_validator("languages", mode="before")
    @classmethod
    def coerce_languages(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",") if lang.strip()]
        return v or []


class ParsedUserAgent(BaseModel):
    """Structured breakdown of a parsed user-agent string."""

    raw: str
    browser_family: str = ""
    browser_version: str = ""
    os_family: str = ""
    os_version: str = ""
    device_type: str = "desktop"  # desktop | mobile | tablet | bot
    is_bot: bool = False
    is_headless: bool = False


class DeviceFingerprint(BaseModel):
    """Stable, privacy-preserving device fingerprint derived from raw signals."""

    # Identity
    device_id: str = Field(..., description="SHA-256 of the stable signal set")
    fuzzy_id: str = Field(..., description="SHA-256 of a softer signal set tolerant to minor changes")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Parsed attributes
    parsed_ua: ParsedUserAgent
    screen_resolution: str  # e.g. "1920x1080"
    color_depth: int
    pixel_ratio: float
    timezone_offset: int
    timezone_name: str
    language: str
    platform: str
    hardware_concurrency: int
    device_memory: float | None
    max_touch_points: int
    has_touch: bool

    # Entropy hashes (never raw data)
    canvas_hash: str
    webgl_vendor: str
    webgl_renderer: str
    audio_hash: str
    fonts_hash: str
    plugins_hash: str

    # Network
    ip_country: str
    ip_asn: str
    ip_is_proxy: bool
    ip_is_vpn: bool
    ip_is_tor: bool

    # Computed signals
    entropy_score: float = Field(..., ge=0.0, le=1.0, description="How unique this device appears")
    spoofing_indicators: list[str] = Field(default_factory=list)
    reputation_score: float = Field(default=1.0, ge=0.0, le=1.0, description="1.0 = pristine; 0.0 = banned")

    # Seen-with metadata
    seen_with_users: list[str] = Field(default_factory=list)
    seen_count: int = Field(default=1, ge=1)

    def touch(self) -> None:
        """Update last_seen_at and increment counter."""
        self.last_seen_at = datetime.now(UTC)
        self.seen_count += 1

    @property
    def age_days(self) -> float:
        """Days since the device was first seen."""
        delta = datetime.now(UTC) - self.created_at
        return delta.total_seconds() / 86_400


class DeviceAnomaly(BaseModel):
    """A single anomaly detected for a device/user pair."""

    anomaly_type: AnomalyType
    severity: float = Field(..., ge=0.0, le=1.0)
    description: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class DeviceRiskAssessment(BaseModel):
    """Full risk assessment for a (user_id, device_fingerprint) pair."""

    user_id: str
    device_id: str
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    risk_score: float = Field(..., ge=0.0, le=1.0, description="0 = no risk; 1 = maximum risk")
    risk_level: RiskLevel
    anomalies: list[DeviceAnomaly] = Field(default_factory=list)

    is_new_device: bool
    is_shared_device: bool
    is_trusted: bool

    signal_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Contribution of each signal component to the final score",
    )
    recommendation: str = ""

    @model_validator(mode="after")
    def set_recommendation(self) -> DeviceRiskAssessment:
        if not self.recommendation:
            if self.risk_level == RiskLevel.CRITICAL:
                self.recommendation = "block"
            elif self.risk_level == RiskLevel.HIGH:
                self.recommendation = "step_up_auth"
            elif self.risk_level == RiskLevel.MEDIUM:
                self.recommendation = "monitor"
            else:
                self.recommendation = "allow"
        return self

# ---------------------------------------------------------------------------
# Storage protocol — swap for Redis in production
# ---------------------------------------------------------------------------


class DeviceStore(Protocol):
    """Minimal async key-value store used by DeviceFingerprintService."""

    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None: ...
    async def sadd(self, key: str, *members: str) -> None: ...
    async def smembers(self, key: str) -> set[str]: ...


class InMemoryDeviceStore:
    """In-process store for MVP / testing; not suitable for multi-process deployments."""

    def __init__(self) -> None:
        self._kv: dict[str, Any] = {}
        self._sets: dict[str, set[str]] = defaultdict(set)
        self._expires: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            if key in self._expires and time.monotonic() > self._expires[key]:
                self._kv.pop(key, None)
                self._expires.pop(key, None)
                return None
            return self._kv.get(key)

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        async with self._lock:
            self._kv[key] = value
            if ttl_seconds is not None:
                self._expires[key] = time.monotonic() + ttl_seconds

    async def sadd(self, key: str, *members: str) -> None:
        async with self._lock:
            self._sets[key].update(members)

    async def smembers(self, key: str) -> set[str]:
        async with self._lock:
            return set(self._sets.get(key, set()))

# ---------------------------------------------------------------------------
# User-agent parsing (lightweight, no external dependency)
# ---------------------------------------------------------------------------

_BOT_PATTERNS = re.compile(
    r"(bot|crawl|spider|slurp|feedfetcher|mediapartners|archiver|wget|curl"
    r"|python-requests|go-http|java/|okhttp|scrapy|phantomjs|headlesschrome"
    r"|slimerjs|htmlunit|selenium|puppeteer)",
    re.IGNORECASE,
)
_HEADLESS_PATTERNS = re.compile(
    r"(headlesschrome|phantomjs|slimerjs|htmlunit|selenium|webdriver)",
    re.IGNORECASE,
)
_MOBILE_PATTERNS = re.compile(
    r"(android|iphone|ipad|mobile|blackberry|windows phone)",
    re.IGNORECASE,
)
_TABLET_PATTERNS = re.compile(r"(ipad|tablet|kindle)", re.IGNORECASE)


def _parse_user_agent(ua: str) -> ParsedUserAgent:
    """Lightweight user-agent parser that avoids external dependencies."""
    is_bot = bool(_BOT_PATTERNS.search(ua))
    is_headless = bool(_HEADLESS_PATTERNS.search(ua))

    # Device type
    if is_bot or is_headless:
        device_type = "bot"
    elif _TABLET_PATTERNS.search(ua):
        device_type = "tablet"
    elif _MOBILE_PATTERNS.search(ua):
        device_type = "mobile"
    else:
        device_type = "desktop"

    # Browser family
    browser_family, browser_version = "", ""
    for pattern, family in [
        (r"Edg(?:e)?/([\d.]+)", "Edge"),
        (r"OPR/([\d.]+)", "Opera"),
        (r"Firefox/([\d.]+)", "Firefox"),
        (r"Chrome/([\d.]+)", "Chrome"),
        (r"Safari/([\d.]+)", "Safari"),
        (r"MSIE ([\d.]+)", "IE"),
        (r"Trident/.*rv:([\d.]+)", "IE"),
    ]:
        m = re.search(pattern, ua)
        if m:
            browser_family = family
            browser_version = m.group(1)
            break

    # OS family
    os_family, os_version = "", ""
    for pattern, family in [
        (r"Windows NT ([\d.]+)", "Windows"),
        (r"Mac OS X ([\d_.]+)", "macOS"),
        (r"Android ([\d.]+)", "Android"),
        (r"iPhone OS ([\d_]+)", "iOS"),
        (r"iPad.*OS ([\d_]+)", "iOS"),
        (r"Linux", "Linux"),
        (r"CrOS", "ChromeOS"),
    ]:
        m = re.search(pattern, ua)
        if m:
            os_family = family
            try:
                os_version = m.group(1).replace("_", ".")
            except IndexError:
                os_version = ""
            break

    return ParsedUserAgent(
        raw=ua,
        browser_family=browser_family,
        browser_version=browser_version,
        os_family=os_family,
        os_version=os_version,
        device_type=device_type,
        is_bot=is_bot,
        is_headless=is_headless,
    )

# ---------------------------------------------------------------------------
# Fingerprint computation helpers
# ---------------------------------------------------------------------------


def _sha256(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


def _compute_entropy_score(signals: BrowserSignals, parsed_ua: ParsedUserAgent) -> float:
    """Heuristic entropy score: how unique/identifiable this device is (0–1)."""
    score = 0.0
    weights = {
        "canvas": 0.20,
        "webgl": 0.15,
        "audio": 0.10,
        "fonts": 0.10,
        "plugins": 0.05,
        "resolution": 0.05,
        "timezone": 0.05,
        "language": 0.05,
        "hardware": 0.10,
        "ua_detail": 0.15,
    }
    if signals.canvas_hash:
        score += weights["canvas"]
    if signals.webgl_vendor or signals.webgl_renderer:
        score += weights["webgl"]
    if signals.audio_hash:
        score += weights["audio"]
    if signals.fonts_hash:
        score += weights["fonts"]
    if signals.plugins:
        score += weights["plugins"]
    if signals.screen_width and signals.screen_height:
        score += weights["resolution"]
    if signals.timezone_name:
        score += weights["timezone"]
    if signals.language:
        score += weights["language"]
    if signals.hardware_concurrency and signals.device_memory:
        score += weights["hardware"]
    if parsed_ua.browser_version and parsed_ua.os_version:
        score += weights["ua_detail"]
    return min(score, 1.0)


def _detect_spoofing_indicators(signals: BrowserSignals, parsed_ua: ParsedUserAgent) -> list[str]:
    """Return a list of spoofing indicator strings."""
    indicators: list[str] = []

    # Headless / bot
    if parsed_ua.is_headless:
        indicators.append("headless_browser_detected")
    if parsed_ua.is_bot:
        indicators.append("bot_user_agent")

    # Navigator properties missing — common in automation frameworks
    if not signals.canvas_hash and not signals.webgl_vendor:
        indicators.append("no_canvas_or_webgl")

    # Touch inconsistency: mobile UA but zero touch points
    if parsed_ua.device_type in ("mobile", "tablet") and signals.max_touch_points == 0:
        indicators.append("mobile_ua_no_touch")

    # Desktop UA but touch points present (possible spoofed UA)
    if parsed_ua.device_type == "desktop" and signals.max_touch_points > 0:
        indicators.append("desktop_ua_with_touch")

    # Timezone offset inconsistency (name present but offset is 0 for non-UTC zones)
    if (
        signals.timezone_name
        and "UTC" not in signals.timezone_name.upper()
        and signals.timezone_offset == 0
    ):
        indicators.append("timezone_offset_mismatch")

    # Network-level proxy/VPN/Tor
    if signals.ip_is_tor:
        indicators.append("tor_exit_node")
    elif signals.ip_is_vpn:
        indicators.append("vpn_detected")
    elif signals.ip_is_proxy:
        indicators.append("proxy_detected")

    # Implausible screen resolution
    if signals.screen_width > 0 and signals.screen_height > 0:
        ratio = signals.screen_width / signals.screen_height
        if ratio < 0.3 or ratio > 5.0:
            indicators.append("implausible_screen_ratio")

    # Unrealistic hardware concurrency
    if signals.hardware_concurrency > 128:
        indicators.append("implausible_cpu_cores")

    # Cookie/storage disabled while claiming to be a real browser
    if not signals.cookie_enabled and not signals.local_storage_available:
        indicators.append("storage_fully_disabled")

    return indicators

# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class DeviceFingerprintService:
    """Production device fingerprinting and risk scoring service.

    Responsibilities:
    - Compute stable SHA-256 device IDs from browser signals.
    - Assess risk by combining device history, anomaly detection, and
      reputation scores.
    - Track which users have been associated with a given device and vice versa.
    - Detect shared devices (fraud rings) and spoofing indicators.
    - Export Prometheus metrics for observability.

    Args:
        store: Async storage backend.  Defaults to InMemoryDeviceStore.
        new_device_risk_weight: Score contribution when a device has never been seen. Default 0.3.
        shared_device_risk_weight: Score contribution when a device is used by multiple users. Default 0.4.
        spoofing_risk_weight: Score contribution per spoofing indicator. Default 0.15.
        max_users_per_device: Number of distinct users before a device is considered shared. Default 3.
        device_ttl_days: Days after last-seen to expire an inactive device. Default 90.
    """

    def __init__(
        self,
        store: DeviceStore | None = None,
        new_device_risk_weight: float = 0.30,
        shared_device_risk_weight: float = 0.40,
        spoofing_risk_weight: float = 0.15,
        max_users_per_device: int = 3,
        device_ttl_days: int = 90,
    ) -> None:
        self._store: DeviceStore = store or InMemoryDeviceStore()
        self._new_device_risk_weight = new_device_risk_weight
        self._shared_device_risk_weight = shared_device_risk_weight
        self._spoofing_risk_weight = spoofing_risk_weight
        self._max_users_per_device = max_users_per_device
        self._device_ttl_seconds = device_ttl_days * 86_400

        self._log = logger.bind(component="DeviceFingerprintService")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compute_fingerprint(self, raw_signals: dict[str, Any]) -> DeviceFingerprint:
        """Parse raw client signals and return a structured DeviceFingerprint.

        Args:
            raw_signals: Dictionary of browser/client signals, typically
                collected by a JS agent or mobile SDK.

        Returns:
            A fully computed DeviceFingerprint with stable device_id.
        """
        signals = BrowserSignals.model_validate(raw_signals)
        parsed_ua = _parse_user_agent(signals.user_agent)

        # Stable device ID — excludes volatile fields (IP, timestamp)
        device_id = _sha256(
            signals.user_agent,
            f"{signals.screen_width}x{signals.screen_height}",
            str(signals.color_depth),
            str(signals.timezone_offset),
            signals.timezone_name,
            signals.language,
            signals.platform,
            str(signals.hardware_concurrency),
            signals.canvas_hash,
            signals.webgl_vendor,
            signals.webgl_renderer,
            signals.audio_hash,
            signals.fonts_hash,
        )

        # Fuzzy ID — tolerant to minor UA changes (browser auto-updates)
        fuzzy_id = _sha256(
            parsed_ua.browser_family,
            parsed_ua.os_family,
            f"{signals.screen_width}x{signals.screen_height}",
            str(signals.timezone_offset),
            signals.language,
            signals.canvas_hash,
        )

        plugins_hash = _sha256(*sorted(signals.plugins)) if signals.plugins else ""
        entropy_score = _compute_entropy_score(signals, parsed_ua)
        spoofing_indicators = _detect_spoofing_indicators(signals, parsed_ua)

        # Check if we already know this device so we can preserve history
        existing: DeviceFingerprint | None = await self._load_device(device_id)
        now = datetime.now(UTC)

        if existing is not None:
            existing.touch()
            fp = existing
        else:
            fp = DeviceFingerprint(
                device_id=device_id,
                fuzzy_id=fuzzy_id,
                created_at=now,
                last_seen_at=now,
                parsed_ua=parsed_ua,
                screen_resolution=f"{signals.screen_width}x{signals.screen_height}",
                color_depth=signals.color_depth,
                pixel_ratio=signals.pixel_ratio,
                timezone_offset=signals.timezone_offset,
                timezone_name=signals.timezone_name,
                language=signals.language,
                platform=signals.platform,
                hardware_concurrency=signals.hardware_concurrency,
                device_memory=signals.device_memory,
                max_touch_points=signals.max_touch_points,
                has_touch=signals.max_touch_points > 0,
                canvas_hash=signals.canvas_hash,
                webgl_vendor=signals.webgl_vendor,
                webgl_renderer=signals.webgl_renderer,
                audio_hash=signals.audio_hash,
                fonts_hash=signals.fonts_hash,
                plugins_hash=plugins_hash,
                ip_country=signals.ip_country,
                ip_asn=signals.ip_asn,
                ip_is_proxy=signals.ip_is_proxy,
                ip_is_vpn=signals.ip_is_vpn,
                ip_is_tor=signals.ip_is_tor,
                entropy_score=entropy_score,
                spoofing_indicators=spoofing_indicators,
            )

        # Always refresh mutable fields
        fp.spoofing_indicators = spoofing_indicators
        fp.entropy_score = entropy_score

        _FINGERPRINT_COMPUTED.inc()
        self._log.debug(
            "fingerprint_computed",
            device_id=device_id[:16],
            is_new=existing is None,
            spoofing_count=len(spoofing_indicators),
        )
        return fp

    async def assess_risk(
        self, user_id: str, device_fingerprint: DeviceFingerprint
    ) -> DeviceRiskAssessment:
        """Compute a risk score for a (user_id, device) pair.

        Args:
            user_id: Authenticated or provisional user identifier.
            device_fingerprint: Previously computed DeviceFingerprint.

        Returns:
            DeviceRiskAssessment with risk_score, risk_level, and anomalies.
        """
        start = time.monotonic()
        device_id = device_fingerprint.device_id
        fp = device_fingerprint

        anomalies: list[DeviceAnomaly] = []
        signal_scores: dict[str, float] = {}

        # 1. New device signal
        is_new = fp.seen_count == 1 and fp.age_days < 0.01
        if is_new:
            anomalies.append(DeviceAnomaly(
                anomaly_type=AnomalyType.NEW_DEVICE,
                severity=self._new_device_risk_weight,
                description="Device has never been seen before",
                evidence={"device_id": device_id[:16]},
            ))
            signal_scores["new_device"] = self._new_device_risk_weight

        # 2. Shared device (multiple users)
        all_users = await self._get_device_users(device_id)
        other_users = all_users - {user_id}
        is_shared = len(other_users) >= self._max_users_per_device
        if other_users:
            shared_severity = min(len(other_users) / 10.0, 1.0) * self._shared_device_risk_weight
            anomalies.append(DeviceAnomaly(
                anomaly_type=AnomalyType.SHARED_DEVICE,
                severity=shared_severity,
                description=f"Device associated with {len(other_users)} other user(s)",
                evidence={"other_user_count": len(other_users)},
            ))
            signal_scores["shared_device"] = shared_severity

        # 3. Rapid user switching (same device, multiple users in short window)
        recent_switches = await self._count_recent_user_switches(device_id)
        if recent_switches > 2:
            switch_severity = min(recent_switches / 5.0, 1.0) * 0.25
            anomalies.append(DeviceAnomaly(
                anomaly_type=AnomalyType.RAPID_USER_SWITCH,
                severity=switch_severity,
                description=f"Device switched between {recent_switches} users recently",
                evidence={"switch_count": recent_switches},
            ))
            signal_scores["rapid_user_switch"] = switch_severity

        # 4. Spoofing indicators
        spoofing_score = 0.0
        for indicator in fp.spoofing_indicators:
            per_indicator = self._spoofing_risk_weight / max(len(fp.spoofing_indicators), 1)
            spoofing_score += per_indicator

            anomaly_type = (
                AnomalyType.HEADLESS_BROWSER
                if "headless" in indicator
                else AnomalyType.BOT_SIGNALS
                if "bot" in indicator
                else AnomalyType.VPN_OR_PROXY
                if any(x in indicator for x in ("vpn", "proxy", "tor"))
                else AnomalyType.TIMEZONE_MISMATCH
                if "timezone" in indicator
                else AnomalyType.ATTRIBUTE_TAMPERING
            )
            anomalies.append(DeviceAnomaly(
                anomaly_type=anomaly_type,
                severity=min(per_indicator, 1.0),
                description=f"Spoofing signal: {indicator}",
                evidence={"indicator": indicator},
            ))

        if spoofing_score > 0:
            signal_scores["spoofing"] = min(spoofing_score, 1.0)

        # 5. Emulator heuristics
        emulator_score = self._emulator_score(fp)
        if emulator_score > 0.1:
            anomalies.append(DeviceAnomaly(
                anomaly_type=AnomalyType.EMULATOR_DETECTED,
                severity=emulator_score,
                description="Device attributes suggest an emulator or virtual machine",
                evidence={"emulator_score": round(emulator_score, 3)},
            ))
            signal_scores["emulator"] = emulator_score

        # 6. Reputation degradation
        reputation_penalty = 1.0 - fp.reputation_score
        if reputation_penalty > 0:
            signal_scores["reputation"] = reputation_penalty

        # --- Final score ---
        raw_score = sum(signal_scores.values())
        # Apply reputation as a multiplier on top
        risk_score = min(raw_score * (0.5 + 0.5 * reputation_penalty + 0.5), 1.0)
        risk_score = min(risk_score, 1.0)

        # Determine level
        if risk_score >= 0.75:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.50:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.25:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        is_trusted = (
            risk_level == RiskLevel.LOW
            and not is_new
            and not is_shared
            and fp.reputation_score >= 0.9
        )

        # Emit metrics
        _RISK_ASSESSED.labels(risk_level=risk_level.value).inc()
        for anomaly in anomalies:
            _ANOMALY_DETECTED.labels(anomaly_type=anomaly.anomaly_type.value).inc()

        elapsed = time.monotonic() - start
        _ASSESSMENT_LATENCY.observe(elapsed)

        assessment = DeviceRiskAssessment(
            user_id=user_id,
            device_id=device_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            anomalies=anomalies,
            is_new_device=is_new,
            is_shared_device=is_shared,
            is_trusted=is_trusted,
            signal_breakdown={k: round(v, 4) for k, v in signal_scores.items()},
        )

        self._log.info(
            "device_risk_assessed",
            user_id=user_id,
            device_id=device_id[:16],
            risk_score=assessment.risk_score,
            risk_level=risk_level.value,
            anomaly_count=len(anomalies),
            latency_ms=round(elapsed * 1000, 2),
        )
        return assessment

    async def register_device(self, user_id: str, device_fingerprint: DeviceFingerprint) -> None:
        """Persist a device fingerprint and associate it with a user.

        Args:
            user_id: The user who presented this device.
            device_fingerprint: The computed fingerprint to store.
        """
        device_id = device_fingerprint.device_id

        # Associate user <-> device bidirectionally
        await self._store.sadd(f"device:{device_id}:users", user_id)
        await self._store.sadd(f"user:{user_id}:devices", device_id)

        # Record this as a recent event for rapid-switch detection
        event_key = f"device:{device_id}:user_events"
        events: list[dict[str, Any]] = await self._store.get(event_key) or []
        events.append({"user_id": user_id, "ts": time.time()})
        # Keep only last 50 events
        events = events[-50:]
        await self._store.set(event_key, events, ttl_seconds=self._device_ttl_seconds)

        # Update seen_with_users on the fingerprint
        all_users = await self._get_device_users(device_id)
        device_fingerprint.seen_with_users = sorted(all_users)

        # Persist fingerprint
        await self._store.set(
            f"device:{device_id}:fingerprint",
            device_fingerprint.model_dump(mode="json"),
            ttl_seconds=self._device_ttl_seconds,
        )

        # Update gauges
        all_device_keys: set[str] = await self._store.smembers("all_devices")
        _KNOWN_DEVICES.set(len(all_device_keys) + 1)
        await self._store.sadd("all_devices", device_id)

        self._log.debug(
            "device_registered",
            user_id=user_id,
            device_id=device_id[:16],
            seen_count=device_fingerprint.seen_count,
        )

    async def get_device_history(self, user_id: str) -> list[DeviceFingerprint]:
        """Return all known device fingerprints for a user.

        Args:
            user_id: The user identifier.

        Returns:
            List of DeviceFingerprint objects, newest first.
        """
        device_ids = await self._store.smembers(f"user:{user_id}:devices")
        fingerprints: list[DeviceFingerprint] = []
        for device_id in device_ids:
            fp = await self._load_device(device_id)
            if fp is not None:
                fingerprints.append(fp)
        fingerprints.sort(key=lambda f: f.last_seen_at, reverse=True)
        return fingerprints

    async def is_device_shared(self, device_id: str) -> bool:
        """Return True if more than one user has been associated with this device.

        Args:
            device_id: The stable device identifier.

        Returns:
            True when the device is shared across multiple users.
        """
        users = await self._get_device_users(device_id)
        is_shared = len(users) > 1
        if is_shared:
            _SHARED_DEVICES.inc()
        return is_shared

    async def update_reputation(self, device_id: str, delta: float) -> float:
        """Adjust the reputation score for a device.

        Args:
            device_id: Device to update.
            delta: Positive values improve reputation; negative degrade it.

        Returns:
            The updated reputation score clamped to [0, 1].
        """
        fp = await self._load_device(device_id)
        if fp is None:
            self._log.warning("update_reputation_device_not_found", device_id=device_id[:16])
            return 0.5

        new_score = max(0.0, min(1.0, fp.reputation_score + delta))
        fp.reputation_score = new_score
        await self._store.set(
            f"device:{device_id}:fingerprint",
            fp.model_dump(mode="json"),
            ttl_seconds=self._device_ttl_seconds,
        )
        self._log.info(
            "device_reputation_updated",
            device_id=device_id[:16],
            old_score=round(fp.reputation_score - delta, 4),
            new_score=round(new_score, 4),
            delta=delta,
        )
        return new_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_device(self, device_id: str) -> DeviceFingerprint | None:
        data = await self._store.get(f"device:{device_id}:fingerprint")
        if data is None:
            return None
        return DeviceFingerprint.model_validate(data)

    async def _get_device_users(self, device_id: str) -> set[str]:
        return await self._store.smembers(f"device:{device_id}:users")

    async def _count_recent_user_switches(self, device_id: str, window_seconds: int = 3600) -> int:
        """Count distinct users who used this device within the last window."""
        events: list[dict[str, Any]] = await self._store.get(f"device:{device_id}:user_events") or []
        cutoff = time.time() - window_seconds
        recent_users: set[str] = {
            e["user_id"] for e in events if e.get("ts", 0) >= cutoff
        }
        return len(recent_users)

    @staticmethod
    def _emulator_score(fp: DeviceFingerprint) -> float:
        """Heuristic score for emulator/VM likelihood."""
        score = 0.0

        # WebGL vendor strings common in software renderers / VMs
        low_gl = fp.webgl_vendor.lower()
        if any(x in low_gl for x in ("mesa", "swiftshader", "llvm", "softpipe", "virtualbox")):
            score += 0.35

        # Suspiciously round hardware: exactly 1 core, tiny memory
        if fp.hardware_concurrency == 1 and fp.device_memory is not None and fp.device_memory <= 0.5:
            score += 0.25

        # Screen resolution typical of emulators / headless
        if fp.screen_resolution in ("0x0", "800x600", "1024x768", "480x800"):
            score += 0.20

        # No canvas, no audio, no fonts — all blanked simultaneously
        blank_count = sum([
            fp.canvas_hash == "",
            fp.audio_hash == "",
            fp.fonts_hash == "",
        ])
        if blank_count == 3:
            score += 0.25

        return min(score, 1.0)


__all__ = ["AnomalyType", "BrowserSignals", "DeviceAnomaly", "DeviceFingerprint", "DeviceFingerprintService", "DeviceRiskAssessment", "InMemoryDeviceStore", "ParsedUserAgent", "RiskLevel"]
