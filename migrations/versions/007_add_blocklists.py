"""Add blocklist entries table for IP, BIN, email, device, country, and merchant blocking.

Revision ID: 007
Revises:     006
Created:     2026-06-08 00:00:00 UTC

Description
-----------
Adds the ``blocklist_entries`` table which powers the platform's real-time
allow/block list engine.  The scoring pipeline checks incoming transactions
against this table (typically via a Redis-backed cache) to short-circuit
scoring when a known-bad (or known-good) entity is encountered.

Supported list types:

  * **ip** — IPv4/IPv6 addresses.  When ``cidr_range`` is populated the
    entry matches an entire subnet (e.g. ``192.168.0.0/16``).
  * **bin** — Bank Identification Number (first 6-8 digits of a card).
  * **email** — Full email address or domain pattern.
  * **device** — Device fingerprint hash.
  * **country** — ISO 3166-1 alpha-2 country code.
  * **merchant** — Merchant identifier or MCC code.

The ``is_allowlist`` boolean flips the semantics: when TRUE the entry acts
as an allowlist override, exempting the matched entity from fraud checks.
This is useful for whitelisting known-good corporate IPs or trusted
merchant IDs.

Expiry is optional — ``expires_at`` enables time-boxed blocks (e.g. block
a BIN range for 30 days during an active fraud ring investigation).  A
background cleanup job archives expired entries.

Rollback Notes
--------------
Drops the ``blocklist_entries`` table.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

# ---------------------------------------------------------------------------
# Revision identifiers
# ---------------------------------------------------------------------------

revision: str = "007"
down_revision: str | Sequence[str] | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TSTZ = sa.TIMESTAMP(timezone=True)
_UUID = pg.UUID(as_uuid=False)


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------

def upgrade() -> None:
    op.create_table(
        "blocklist_entries",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        # Type of entity being blocked/allowed.
        # ip | bin | email | device | country | merchant
        sa.Column("list_type", sa.String(50), nullable=False),
        # The blocked/allowed value — IP address, BIN, email, device hash, etc.
        sa.Column("value", sa.String(512), nullable=False),
        # Optional CIDR notation for IP-range blocks (e.g. "10.0.0.0/8").
        # NULL for non-IP list types.
        sa.Column("cidr_range", sa.String(50), nullable=True),
        # Reason for the block — free-text for audit / analyst review.
        sa.Column("reason", sa.Text, nullable=True),
        # Identity of the person or system that created the entry.
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        # Optional expiration for time-boxed blocks.
        sa.Column("expires_at", _TSTZ, nullable=True),
        # When TRUE the entry acts as an allowlist (whitelist) override.
        sa.Column("is_allowlist", sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
    )

    # Primary lookup: type + value (used by the scoring pipeline cache loader).
    op.create_index(
        "ix_blocklist_entries_type_value",
        "blocklist_entries",
        ["list_type", "value"],
    )
    # Filter by list type.
    op.create_index("ix_blocklist_entries_list_type", "blocklist_entries", ["list_type"])
    # Find entries expiring soon for cleanup jobs.
    op.create_index(
        "ix_blocklist_entries_expires_at",
        "blocklist_entries",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )
    # Separate index for allowlist vs. blocklist queries.
    op.create_index(
        "ix_blocklist_entries_is_allowlist",
        "blocklist_entries",
        ["is_allowlist", "list_type"],
    )
    # CIDR range lookup for IP-type entries.
    op.create_index(
        "ix_blocklist_entries_cidr",
        "blocklist_entries",
        ["cidr_range"],
        postgresql_where=sa.text("cidr_range IS NOT NULL"),
    )


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------

def downgrade() -> None:
    op.drop_table("blocklist_entries")
