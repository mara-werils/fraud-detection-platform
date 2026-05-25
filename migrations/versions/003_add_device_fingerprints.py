"""Add device fingerprint storage tables.

Revision ID: 003
Revises:     002
Created:     2026-05-25 00:00:00 UTC

Description
-----------
Adds three tables that power device-based fraud detection signals:

device_fingerprints
    Canonical record of a unique device, keyed by a stable fingerprint hash
    computed from browser/OS/hardware signals (TLS fingerprint, canvas hash,
    font list hash, screen geometry, etc.).  The ``attributes`` JSONB column
    stores the full raw signal set for future feature extraction without
    schema changes.

    Risk signals are pre-computed and stored here so that the hot-path
    scoring engine can look them up with a single indexed query instead of
    re-computing from raw attributes on every transaction.

device_user_associations
    Many-to-many relationship between devices and merchant-supplied user IDs.
    A single device may be used by multiple users (shared device, account
    takeover) and a user may have multiple trusted devices (multi-device user).

    The ``association_type`` column distinguishes between:
      - trusted: user explicitly linked this device
      - observed: device seen on ≥1 transaction for this user
      - suspicious: flagged by a rule or analyst

    Counts and timestamps enable velocity signals:
      - How many users has this device ever been associated with?
      - How recently did a new user appear on this device?

ip_intelligence_cache
    Cached enrichment data for IPv4/IPv6 addresses queried from external
    threat-intelligence providers (MaxMind, IPinfo, AbuseIPDB, etc.).
    Using a local cache avoids redundant paid API calls and stays within
    latency budgets.

    Cache invalidation is handled by the ``expires_at`` column.  The scoring
    service skips rows where ``expires_at < NOW()``.

    The ``risk_signals`` JSONB column holds provider-specific threat scores,
    ASN classification (residential / datacenter / vpn / tor / proxy),
    and geolocation for the hot-path.

Rollback Notes
--------------
Drops all three tables.  No other tables reference these as FK targets.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

# ---------------------------------------------------------------------------
# Revision identifiers
# ---------------------------------------------------------------------------

revision: str = "003"
down_revision: str | Sequence[str] | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TSTZ = sa.TIMESTAMP(timezone=True)
_UUID = pg.UUID(as_uuid=False)
_JSONB = pg.JSONB(none_as_null=True)
_INET = pg.INET()


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------

def upgrade() -> None:
    # ── device_fingerprints ───────────────────────────────────────────────────
    op.create_table(
        "device_fingerprints",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        # Stable hash of the combined device signals.  Computed client-side and
        # transmitted in the X-Device-Fingerprint request header.
        # SHA-256 hex digest (64 chars) or a vendor-specific token.
        sa.Column("fingerprint_hash", sa.String(128), nullable=False),
        # Tenant for data isolation.  NULL = platform-level device (rare).
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
        # ── Device classification signals ─────────────────────────────────────
        # Browser family: Chrome | Firefox | Safari | Edge | unknown
        sa.Column("browser_family", sa.String(100), nullable=True),
        # OS family: Windows | macOS | iOS | Android | Linux | unknown
        sa.Column("os_family", sa.String(100), nullable=True),
        # Device category: desktop | mobile | tablet | bot | unknown
        sa.Column("device_type", sa.String(50), nullable=True),
        # User-Agent string at first seen (truncated to 500 chars).
        sa.Column("user_agent", sa.String(500), nullable=True),
        # Timezone offset in minutes from UTC (e.g. -300 for EST).
        sa.Column("timezone_offset_minutes", sa.SmallInteger, nullable=True),
        # IANA timezone name if extracted (e.g. "America/New_York").
        sa.Column("timezone_name", sa.String(100), nullable=True),
        # Screen resolution as "WIDTHxHEIGHT" (e.g. "1920x1080").
        sa.Column("screen_resolution", sa.String(20), nullable=True),
        # Number of logical CPU cores reported by the browser.
        sa.Column("cpu_cores", sa.SmallInteger, nullable=True),
        # Device memory bucket in GB (browser API rounds to nearest power of 2).
        sa.Column("device_memory_gb", sa.SmallInteger, nullable=True),
        # Canvas fingerprint hash (uniquely identifies GPU + driver combo).
        sa.Column("canvas_hash", sa.String(64), nullable=True),
        # WebGL renderer string hash.
        sa.Column("webgl_hash", sa.String(64), nullable=True),
        # Audio fingerprint hash.
        sa.Column("audio_hash", sa.String(64), nullable=True),
        # Installed font list hash.
        sa.Column("font_hash", sa.String(64), nullable=True),
        # TLS/JA3 fingerprint hash (network-layer device identifier).
        sa.Column("tls_fingerprint", sa.String(64), nullable=True),
        # ── Full raw signals ──────────────────────────────────────────────────
        # Complete attribute bag from the client SDK — enables retroactive
        # feature engineering without another collection event.
        sa.Column("attributes", _JSONB, nullable=False, server_default="{}"),
        # ── Pre-computed risk signals ─────────────────────────────────────────
        # Composite risk score [0.0, 1.0] updated whenever new signals arrive.
        sa.Column(
            "risk_score",
            sa.Float,
            nullable=False,
            server_default="0.0",
        ),
        # JSON object of named risk signals: {headless_browser, emulator, ...}
        sa.Column("risk_signals", _JSONB, nullable=False, server_default="{}"),
        # ── Temporal tracking ─────────────────────────────────────────────────
        sa.Column("first_seen_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("last_seen_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        # How many unique transaction IDs have used this fingerprint.
        sa.Column("transaction_count", sa.BigInteger, nullable=False, server_default="0"),
        # How many distinct org-scoped user IDs have used this fingerprint.
        sa.Column("unique_user_count", sa.Integer, nullable=False, server_default="0"),
        # How many distinct IP addresses have been seen with this fingerprint.
        sa.Column("unique_ip_count", sa.Integer, nullable=False, server_default="0"),
        # Analyst label override: trusted | suspicious | blocked | NULL
        sa.Column("manual_label", sa.String(50), nullable=True),
        sa.Column("labeled_by", sa.String(255), nullable=True),
        sa.Column("labeled_at", _TSTZ, nullable=True),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_unique_constraint(
        "uq_device_fingerprints_hash_org",
        "device_fingerprints",
        ["fingerprint_hash", "org_id"],
    )
    op.create_index(
        "ix_device_fingerprints_hash",
        "device_fingerprints",
        ["fingerprint_hash"],
    )
    op.create_index(
        "ix_device_fingerprints_org_id",
        "device_fingerprints",
        ["org_id"],
    )
    op.create_index(
        "ix_device_fingerprints_risk_score",
        "device_fingerprints",
        ["risk_score"],
    )
    op.create_index(
        "ix_device_fingerprints_last_seen",
        "device_fingerprints",
        ["last_seen_at"],
    )
    op.create_index(
        "ix_device_fingerprints_canvas",
        "device_fingerprints",
        ["canvas_hash"],
        postgresql_where=sa.text("canvas_hash IS NOT NULL"),
    )
    op.create_index(
        "ix_device_fingerprints_tls",
        "device_fingerprints",
        ["tls_fingerprint"],
        postgresql_where=sa.text("tls_fingerprint IS NOT NULL"),
    )
    # GIN for full JSONB attribute searches.
    op.create_index(
        "ix_device_fingerprints_risk_signals_gin",
        "device_fingerprints",
        ["risk_signals"],
        postgresql_using="gin",
    )

    # ── device_user_associations ──────────────────────────────────────────────
    op.create_table(
        "device_user_associations",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "device_id",
            _UUID,
            sa.ForeignKey("device_fingerprints.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Merchant-supplied user identifier (string, not FK — external system).
        sa.Column("user_id", sa.String(100), nullable=False),
        # trusted | observed | suspicious | blocked
        sa.Column(
            "association_type",
            sa.String(50),
            nullable=False,
            server_default="observed",
        ),
        # Confidence score for ML-derived associations [0.0, 1.0].
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("first_seen_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("last_seen_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("transaction_count", sa.Integer, nullable=False, server_default="0"),
        # Context from the last transaction that updated this association.
        sa.Column("last_transaction_id", sa.String(100), nullable=True),
        sa.Column("last_ip_address", _INET, nullable=True),
        # Notes added by an analyst when manually labelling.
        sa.Column("analyst_notes", sa.Text, nullable=True),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_unique_constraint(
        "uq_device_user_assoc",
        "device_user_associations",
        ["device_id", "org_id", "user_id"],
    )
    op.create_index(
        "ix_device_user_assoc_device",
        "device_user_associations",
        ["device_id"],
    )
    op.create_index(
        "ix_device_user_assoc_user",
        "device_user_associations",
        ["org_id", "user_id"],
    )
    op.create_index(
        "ix_device_user_assoc_type",
        "device_user_associations",
        ["association_type"],
    )
    op.create_index(
        "ix_device_user_assoc_last_seen",
        "device_user_associations",
        ["last_seen_at"],
    )
    # Partial index for suspicious/blocked associations — used by risk engine.
    op.create_index(
        "ix_device_user_assoc_risky",
        "device_user_associations",
        ["org_id", "device_id"],
        postgresql_where=sa.text("association_type IN ('suspicious', 'blocked')"),
    )

    # ── ip_intelligence_cache ─────────────────────────────────────────────────
    op.create_table(
        "ip_intelligence_cache",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        # PostgreSQL native INET type stores both IPv4 and IPv6.
        sa.Column("ip_address", _INET, nullable=False),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
        # ── Geolocation ───────────────────────────────────────────────────────
        sa.Column("country_code", sa.String(2), nullable=True),  # ISO 3166-1 alpha-2
        sa.Column("country_name", sa.String(100), nullable=True),
        sa.Column("region", sa.String(100), nullable=True),
        sa.Column("city", sa.String(100), nullable=True),
        sa.Column("postal_code", sa.String(20), nullable=True),
        sa.Column("latitude", sa.Numeric(9, 6), nullable=True),
        sa.Column("longitude", sa.Numeric(9, 6), nullable=True),
        sa.Column("timezone", sa.String(100), nullable=True),
        # ── Network metadata ──────────────────────────────────────────────────
        # Autonomous System Number (e.g. 15169 for Google).
        sa.Column("asn", sa.Integer, nullable=True),
        sa.Column("asn_org", sa.String(255), nullable=True),
        # ISP or hosting provider description.
        sa.Column("isp", sa.String(255), nullable=True),
        # Network classification: residential | datacenter | cdn | mobile | unknown
        sa.Column("network_type", sa.String(50), nullable=True),
        # ── Threat intelligence flags ─────────────────────────────────────────
        sa.Column("is_vpn", sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
        sa.Column("is_tor", sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
        sa.Column("is_proxy", sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
        sa.Column("is_datacenter", sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
        sa.Column("is_bogon", sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
        # Normalised abuse confidence score [0, 100] from AbuseIPDB or similar.
        sa.Column("abuse_confidence_score", sa.SmallInteger, nullable=True),
        # Number of distinct reports in the threat-intel provider's database.
        sa.Column("abuse_report_count", sa.Integer, nullable=True),
        # Pre-computed composite risk score for hot-path lookup [0.0, 1.0].
        sa.Column("risk_score", sa.Float, nullable=False, server_default="0.0"),
        # Full enrichment payload from the upstream provider (MaxMind, IPinfo, etc.)
        sa.Column("raw_response", _JSONB, nullable=False, server_default="{}"),
        # Named risk signals extracted from raw_response for feature engineering.
        sa.Column("risk_signals", _JSONB, nullable=False, server_default="{}"),
        # ── Cache lifecycle ───────────────────────────────────────────────────
        # Intelligence source: maxmind | ipinfo | abuseipdb | composite
        sa.Column("source", sa.String(100), nullable=False, server_default="composite"),
        sa.Column("fetched_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        # Scoring service skips rows where expires_at < NOW().
        # Typical TTL: 24h for residential IPs, 1h for datacenter/VPN.
        sa.Column("expires_at", _TSTZ, nullable=False),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_unique_constraint(
        "uq_ip_cache_ip_org",
        "ip_intelligence_cache",
        ["ip_address", "org_id"],
    )
    op.create_index(
        "ix_ip_cache_ip_address",
        "ip_intelligence_cache",
        ["ip_address"],
    )
    op.create_index(
        "ix_ip_cache_expires",
        "ip_intelligence_cache",
        ["expires_at"],
    )
    op.create_index(
        "ix_ip_cache_country",
        "ip_intelligence_cache",
        ["country_code"],
    )
    op.create_index(
        "ix_ip_cache_risk_score",
        "ip_intelligence_cache",
        ["risk_score"],
    )
    # Partial index for IPs with high abuse scores — used by block-list checks.
    op.create_index(
        "ix_ip_cache_high_abuse",
        "ip_intelligence_cache",
        ["ip_address", "abuse_confidence_score"],
        postgresql_where=sa.text("abuse_confidence_score >= 75"),
    )
    # Partial index for anonymising proxies — used by rule engine.
    op.create_index(
        "ix_ip_cache_anon_proxy",
        "ip_intelligence_cache",
        ["ip_address"],
        postgresql_where=sa.text("is_vpn OR is_tor OR is_proxy"),
    )


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------

def downgrade() -> None:
    op.drop_table("ip_intelligence_cache")
    op.drop_table("device_user_associations")
    op.drop_table("device_fingerprints")
