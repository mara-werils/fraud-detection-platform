"""Transaction enrichment: IP geolocation, device fingerprinting, merchant risk scoring."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Merchant category risk table (static)
# ---------------------------------------------------------------------------

_MERCHANT_CATEGORY_RISK: dict[str, float] = {
    "gambling": 0.95,
    "cryptocurrency": 0.90,
    "money_transfer": 0.85,
    "adult_entertainment": 0.80,
    "pawn_shop": 0.75,
    "wire_transfer": 0.75,
    "jewelry": 0.60,
    "electronics": 0.55,
    "travel": 0.50,
    "gas_station": 0.40,
    "restaurant": 0.25,
    "grocery": 0.20,
    "healthcare": 0.15,
    "utilities": 0.10,
    "education": 0.05,
    "government": 0.05,
}

_DEFAULT_CATEGORY_RISK = 0.30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GeoInfo:
    """Geolocation information resolved from IP or transaction coordinates."""

    country: str = "unknown"
    region: str = "unknown"
    city: str = "unknown"
    latitude: float = 0.0
    longitude: float = 0.0
    is_vpn: bool = False
    is_proxy: bool = False


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Parsed device fingerprint information."""

    raw_user_agent: str = ""
    os_family: str = "unknown"
    browser_family: str = "unknown"
    device_type: str = "unknown"
    fingerprint_hash: str = ""


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    """Complete enrichment output for a single transaction."""

    geo: GeoInfo = field(default_factory=GeoInfo)
    device: DeviceInfo = field(default_factory=DeviceInfo)
    merchant_risk_score: float = _DEFAULT_CATEGORY_RISK
    enrichment_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# GeoIP lookup
# ---------------------------------------------------------------------------


class GeoIPResolver:
    """MaxMind GeoLite2 IP geolocation resolver.

    Falls back to transaction lat/lon when the GeoIP database is unavailable.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._reader: Any | None = None

    async def start(self) -> None:
        """Open the GeoIP database if it exists."""
        path = Path(self._db_path)
        if path.is_file():
            try:
                import geoip2.database  # type: ignore[import-untyped]

                self._reader = geoip2.database.Reader(str(path))
                await logger.ainfo("geoip_db_loaded", path=self._db_path)
            except ImportError:
                await logger.awarning(
                    "geoip2_not_installed",
                    msg="Install geoip2 for IP geolocation; falling back to lat/lon.",
                )
            except Exception as exc:
                await logger.awarning("geoip_db_open_failed", error=str(exc))
        else:
            await logger.ainfo(
                "geoip_db_not_found",
                path=self._db_path,
                msg="Will use transaction lat/lon for geolocation.",
            )

    async def close(self) -> None:
        """Close the GeoIP database reader."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            await logger.ainfo("geoip_db_closed")

    def resolve(
        self,
        ip_address: str | None,
        fallback_lat: float | None = None,
        fallback_lon: float | None = None,
    ) -> GeoInfo:
        """Resolve geolocation from an IP address, falling back to lat/lon."""
        if self._reader is not None and ip_address:
            try:
                resp = self._reader.city(ip_address)
                return GeoInfo(
                    country=resp.country.iso_code or "unknown",
                    region=resp.subdivisions.most_specific.name or "unknown" if resp.subdivisions else "unknown",
                    city=resp.city.name or "unknown" if resp.city else "unknown",
                    latitude=resp.location.latitude or 0.0 if resp.location else 0.0,
                    longitude=resp.location.longitude or 0.0 if resp.location else 0.0,
                    is_vpn=False,
                    is_proxy=False,
                )
            except Exception:
                pass  # fall through to coordinate-based fallback

        # Fallback: use transaction coordinates
        lat = fallback_lat or 0.0
        lon = fallback_lon or 0.0
        return GeoInfo(
            country="unknown",
            region="unknown",
            city="unknown",
            latitude=lat,
            longitude=lon,
            is_vpn=False,
            is_proxy=False,
        )


# ---------------------------------------------------------------------------
# Device fingerprint analysis
# ---------------------------------------------------------------------------

# Simple user-agent parsing without heavyweight dependency
_OS_TOKENS: list[tuple[str, str]] = [
    ("Windows", "Windows"),
    ("Macintosh", "macOS"),
    ("Mac OS X", "macOS"),
    ("Linux", "Linux"),
    ("Android", "Android"),
    ("iPhone", "iOS"),
    ("iPad", "iPadOS"),
]

_BROWSER_TOKENS: list[tuple[str, str]] = [
    ("Edg/", "Edge"),
    ("OPR/", "Opera"),
    ("Chrome/", "Chrome"),
    ("Firefox/", "Firefox"),
    ("Safari/", "Safari"),
]


def parse_device_info(
    user_agent: str | None,
    device_id: str | None,
) -> DeviceInfo:
    """Parse user-agent string and device ID into structured device info."""
    ua = user_agent or ""

    os_family = "unknown"
    for token, name in _OS_TOKENS:
        if token in ua:
            os_family = name
            break

    browser_family = "unknown"
    for token, name in _BROWSER_TOKENS:
        if token in ua:
            browser_family = name
            break

    # Device type heuristic
    device_type = "unknown"
    ua_lower = ua.lower()
    if any(kw in ua_lower for kw in ("mobile", "android", "iphone")):
        device_type = "mobile"
    elif any(kw in ua_lower for kw in ("tablet", "ipad")):
        device_type = "tablet"
    elif ua:
        device_type = "desktop"

    # Fingerprint hash from device_id or user-agent
    raw = device_id or ua
    fingerprint_hash = hashlib.sha256(raw.encode()).hexdigest()[:16] if raw else ""

    return DeviceInfo(
        raw_user_agent=ua,
        os_family=os_family,
        browser_family=browser_family,
        device_type=device_type,
        fingerprint_hash=fingerprint_hash,
    )


# ---------------------------------------------------------------------------
# Merchant risk scoring
# ---------------------------------------------------------------------------


def compute_merchant_risk(merchant_category: str | None) -> float:
    """Return a risk score for the given merchant category."""
    if not merchant_category:
        return _DEFAULT_CATEGORY_RISK
    return _MERCHANT_CATEGORY_RISK.get(merchant_category.lower(), _DEFAULT_CATEGORY_RISK)


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------


class TransactionEnricher:
    """Orchestrates all enrichment steps for incoming transactions."""

    def __init__(self, geoip_db_path: str) -> None:
        self._geo_resolver = GeoIPResolver(geoip_db_path)

    async def start(self) -> None:
        """Initialize enrichment resources."""
        await self._geo_resolver.start()
        await logger.ainfo("transaction_enricher_started")

    async def close(self) -> None:
        """Release enrichment resources."""
        await self._geo_resolver.close()
        await logger.ainfo("transaction_enricher_closed")

    def enrich(self, txn: dict[str, Any]) -> EnrichmentResult:
        """Enrich a raw transaction dictionary with geo, device, and risk data.

        Args:
            txn: Deserialized transaction from Kafka.

        Returns:
            EnrichmentResult with all enrichment fields populated.
        """
        import time

        start = time.monotonic()

        geo = self._geo_resolver.resolve(
            ip_address=txn.get("ip_address"),
            fallback_lat=txn.get("latitude"),
            fallback_lon=txn.get("longitude"),
        )

        device = parse_device_info(
            user_agent=txn.get("metadata", {}).get("user_agent"),
            device_id=txn.get("device_id"),
        )

        merchant_risk = compute_merchant_risk(txn.get("merchant_category"))

        latency_ms = (time.monotonic() - start) * 1000.0

        return EnrichmentResult(
            geo=geo,
            device=device,
            merchant_risk_score=merchant_risk,
            enrichment_latency_ms=latency_ms,
        )
