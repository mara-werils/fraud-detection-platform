"""Add API keys table for service authentication and rate limiting.

Revision ID: 005
Revises:     004
Created:     2026-06-08 00:00:00 UTC

Description
-----------
Adds a single ``api_keys`` table that stores hashed API keys used by external
clients and internal services to authenticate against the platform's REST API.

Each key is stored as a SHA-256 hash (``key_hash``); the raw key is never
persisted.  A short ``key_prefix`` (first 8 characters) is kept in plaintext
so that operators can identify a key from a partial value shown in logs or
support tickets.

Key features:

  * **Scoped access** — the ``scopes`` JSONB column holds an array of
    permission strings (e.g. ``["scoring:read", "cases:write"]``).  The
    auth middleware intersects this list with the endpoint's required scope
    before granting access.
  * **Per-key rate limiting** — ``rate_limit`` stores the maximum number of
    requests per minute.  NULL means the tenant-level default applies.
  * **Tenant isolation** — ``tenant_id`` links the key to a tenant so that
    all queries are automatically scoped.
  * **Expiry & revocation** — ``expires_at`` enables time-boxed keys;
    ``revoked_at`` supports immediate invalidation without deleting the row
    (preserving an audit trail).
  * **Usage tracking** — ``last_used_at`` is updated on every successful
    authentication so stale keys can be identified and rotated.

Rollback Notes
--------------
Drops the ``api_keys`` table.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

# ---------------------------------------------------------------------------
# Revision identifiers
# ---------------------------------------------------------------------------

revision: str = "005"
down_revision: str | Sequence[str] | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TSTZ = sa.TIMESTAMP(timezone=True)
_UUID = pg.UUID(as_uuid=False)
_JSONB = pg.JSONB(none_as_null=True)


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------

def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        # SHA-256 hash of the raw API key — used for constant-time lookup.
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        # First 8 characters of the raw key for identification in logs / UI.
        sa.Column("key_prefix", sa.String(16), nullable=False),
        # Human-readable label: "production-scoring-service", "analyst-dashboard"
        sa.Column("name", sa.String(255), nullable=False),
        # Permission scopes: ["scoring:read", "cases:write", "admin:*"]
        sa.Column("scopes", _JSONB, nullable=False, server_default="[]"),
        # Maximum requests per minute.  NULL = use tenant default.
        sa.Column("rate_limit", sa.Integer, nullable=True),
        # Tenant that owns this key.
        sa.Column(
            "tenant_id",
            _UUID,
            nullable=False,
        ),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        # Optional expiration — NULL means the key does not expire.
        sa.Column("expires_at", _TSTZ, nullable=True),
        # Soft-revocation timestamp — set when an operator revokes the key.
        sa.Column("revoked_at", _TSTZ, nullable=True),
        # Updated on every successful authentication.
        sa.Column("last_used_at", _TSTZ, nullable=True),
    )

    # Fast lookup by hash during authentication.
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)
    # Filter keys by tenant.
    op.create_index("ix_api_keys_tenant_id", "api_keys", ["tenant_id"])
    # Identify keys by prefix (support / debugging).
    op.create_index("ix_api_keys_key_prefix", "api_keys", ["key_prefix"])
    # Find expired keys for cleanup jobs.
    op.create_index(
        "ix_api_keys_expires_at",
        "api_keys",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )
    # Find revoked keys.
    op.create_index(
        "ix_api_keys_revoked_at",
        "api_keys",
        ["revoked_at"],
        postgresql_where=sa.text("revoked_at IS NOT NULL"),
    )
    # Scopes GIN index for containment queries (@>).
    op.create_index(
        "ix_api_keys_scopes_gin",
        "api_keys",
        ["scopes"],
        postgresql_using="gin",
    )


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------

def downgrade() -> None:
    op.drop_table("api_keys")
