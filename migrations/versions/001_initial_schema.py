"""Initial schema — core platform tables.

Revision ID: 001
Revises:     (none — first migration)
Created:     2026-05-25 00:00:00 UTC

Description
-----------
Creates the full baseline schema for the Fraud Detection Platform:

  * organizations    — multi-tenant root; every other table is scoped to an org
  * users            — human operators / analysts with JWT-based auth
  * api_keys         — machine-to-machine authentication tokens
  * transactions     — immutable scored transaction log
  * cases            — fraud investigation workflow items
  * feedback         — analyst label decisions used for model retraining
  * alerts           — real-time notifications linked to transactions / cases
  * audit_logs       — tamper-evident activity log (GDPR Art. 30 registry)
  * usage_records    — per-request metering for billing and quota enforcement

All primary keys are UUID v4.  Timestamps use TIMESTAMP WITH TIME ZONE so that
UTC offsets are stored and portable across regions.  JSONB columns are used
instead of JSON for indexing capability and storage efficiency.

Rollback Notes
--------------
``downgrade()`` drops all tables and extensions.  This is a **complete data
loss** operation.  Never run on production without a full backup.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from alembic import op

# ---------------------------------------------------------------------------
# Revision identifiers
# ---------------------------------------------------------------------------

revision: str = "001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TSTZ = sa.TIMESTAMP(timezone=True)
_UUID = pg.UUID(as_uuid=False)  # stored as native pg UUID, returned as str in raw SQL
_JSONB = pg.JSONB(none_as_null=True)


def _now_utc() -> sa.TextClause:
    return sa.text("NOW() AT TIME ZONE 'UTC'")


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------

def upgrade() -> None:
    # ── Prerequisites ────────────────────────────────────────────────────────
    # uuid-ossp gives us gen_random_uuid() on older Postgres.
    # pgcrypto is useful for key hashing; installed here for completeness.
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    # ── organizations ────────────────────────────────────────────────────────
    op.create_table(
        "organizations",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        # Subscription plan: starter | growth | enterprise
        sa.Column("plan", sa.String(50), nullable=False, server_default="starter"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("TRUE")),
        sa.Column("rate_limit_rps", sa.Integer, nullable=False, server_default="100"),
        sa.Column("monthly_quota", sa.Integer, nullable=False, server_default="1000000"),
        # Flexible key/value bag for org-level feature flags and preferences.
        sa.Column("settings", _JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_unique_constraint("uq_organizations_name", "organizations", ["name"])
    op.create_unique_constraint("uq_organizations_slug", "organizations", ["slug"])
    op.create_index("ix_organizations_slug", "organizations", ["slug"])
    op.create_index("ix_organizations_is_active", "organizations", ["is_active"])

    # ── users ────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False, server_default=""),
        # Roles: superadmin | admin | analyst | viewer
        sa.Column("role", sa.String(50), nullable=False, server_default="analyst"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("TRUE")),
        sa.Column("is_verified", sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
        sa.Column("last_login", _TSTZ, nullable=True),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_unique_constraint("uq_users_email", "users", ["email"])
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_org_id", "users", ["org_id"])
    op.create_index("ix_users_org_role", "users", ["org_id", "role"])

    # ── api_keys ─────────────────────────────────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_by",
            _UUID,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        # SHA-256 hex digest of the raw key — never store the raw key.
        sa.Column("key_hash", sa.String(64), nullable=False),
        # First 12 chars of the key for UI display (e.g. "fdp_live_xxxx").
        sa.Column("key_prefix", sa.String(12), nullable=False),
        # JSON array of permission strings: ["score:write", "cases:read", ...]
        sa.Column("scopes", _JSONB, nullable=False, server_default="[]"),
        sa.Column("rate_limit", sa.Integer, nullable=False, server_default="1000"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("TRUE")),
        sa.Column("last_used", _TSTZ, nullable=True),
        sa.Column("expires_at", _TSTZ, nullable=True),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_unique_constraint("uq_api_keys_key_hash", "api_keys", ["key_hash"])
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"])
    op.create_index("ix_api_keys_org_id", "api_keys", ["org_id"])
    op.create_index("ix_api_keys_is_active", "api_keys", ["org_id", "is_active"])

    # ── transactions ─────────────────────────────────────────────────────────
    op.create_table(
        "transactions",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Merchant-supplied transaction identifier.
        sa.Column("transaction_id", sa.String(100), nullable=False),
        # Merchant-supplied end-user identifier (pseudonymized after erasure).
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("amount", sa.Numeric(18, 4), nullable=False),
        sa.Column("currency", sa.String(10), nullable=False, server_default="USD"),
        sa.Column("merchant_id", sa.String(100), nullable=True),
        # Model output: 0.0 (legit) → 1.0 (fraud).
        sa.Column("fraud_score", sa.Float, nullable=False),
        # allow | review | block
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("is_flagged", sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
        # Snapshot of the model version at scoring time for reproducibility.
        sa.Column("model_version", sa.String(50), nullable=False, server_default=""),
        # Full feature vector used for scoring — enables post-hoc debugging.
        sa.Column("features", _JSONB, nullable=False, server_default="{}"),
        # Top-N risk factors returned by the model / rule engine.
        sa.Column("risk_factors", _JSONB, nullable=False, server_default="[]"),
        # Client-reported transaction timestamp (may differ from scored_at).
        sa.Column("timestamp", _TSTZ, nullable=False),
        sa.Column("scored_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_transactions_org_user", "transactions", ["org_id", "user_id"])
    op.create_index("ix_transactions_fraud_score", "transactions", ["fraud_score"])
    op.create_index("ix_transactions_timestamp", "transactions", ["timestamp"])
    op.create_index("ix_transactions_txn_id", "transactions", ["transaction_id"])
    op.create_index(
        "ix_transactions_org_timestamp",
        "transactions",
        ["org_id", "timestamp"],
    )
    # Partial index to quickly locate high-risk transactions without a full scan.
    op.create_index(
        "ix_transactions_high_risk",
        "transactions",
        ["org_id", "fraud_score"],
        postgresql_where=sa.text("fraud_score >= 0.5"),
    )

    # ── cases ─────────────────────────────────────────────────────────────────
    op.create_table(
        "cases",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Human-readable, stable case identifier (e.g. "CASE-2026-0042").
        sa.Column("case_id", sa.String(100), nullable=False),
        sa.Column("transaction_id", sa.String(100), nullable=False),
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("fraud_score", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("amount", sa.Numeric(18, 4), nullable=False, server_default="0"),
        sa.Column("currency", sa.String(10), nullable=False, server_default="USD"),
        # open | in_review | escalated | closed | cancelled
        sa.Column("status", sa.String(50), nullable=False, server_default="open"),
        # low | medium | high | critical
        sa.Column("priority", sa.String(20), nullable=False, server_default="medium"),
        # Analyst email or UUID who owns this case.
        sa.Column("assigned_to", sa.String(255), nullable=True),
        # Analyst comments — append-only JSON array: [{author, text, ts}]
        sa.Column("notes", _JSONB, nullable=False, server_default="[]"),
        # Structured event log: [{type, actor, data, ts}]
        sa.Column("events", _JSONB, nullable=False, server_default="[]"),
        # Free-form tags for categorisation (e.g. ["card_not_present", "velocity"])
        sa.Column("tags", _JSONB, nullable=False, server_default="[]"),
        # Ground-truth label set when case is resolved.
        sa.Column("is_fraud", sa.Boolean, nullable=True),
        sa.Column("resolved_at", _TSTZ, nullable=True),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_unique_constraint("uq_cases_case_id", "cases", ["case_id"])
    op.create_index("ix_cases_case_id", "cases", ["case_id"])
    op.create_index("ix_cases_org_status", "cases", ["org_id", "status"])
    op.create_index("ix_cases_org_created", "cases", ["org_id", "created_at"])
    op.create_index("ix_cases_transaction_id", "cases", ["transaction_id"])
    op.create_index("ix_cases_assigned_to", "cases", ["assigned_to"])

    # ── feedback ─────────────────────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("transaction_id", sa.String(100), nullable=False),
        # User ID of the analyst who provided the label.
        sa.Column("analyst_id", sa.String(255), nullable=False),
        # allow | review | block (system's original decision)
        sa.Column("original_decision", sa.String(20), nullable=False),
        # allow | block (analyst's ground-truth label)
        sa.Column("analyst_decision", sa.String(20), nullable=False),
        sa.Column("is_fraud", sa.Boolean, nullable=False),
        sa.Column("notes", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_feedback_txn_id", "feedback", ["transaction_id"])
    op.create_index("ix_feedback_org_id", "feedback", ["org_id"])
    op.create_index("ix_feedback_analyst_id", "feedback", ["analyst_id"])
    op.create_index("ix_feedback_org_created", "feedback", ["org_id", "created_at"])

    # ── alerts ───────────────────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # alert type: fraud_detected | velocity_breach | rule_trigger | anomaly
        sa.Column("alert_type", sa.String(100), nullable=False),
        # References a transaction or case (denormalised for query speed).
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", sa.String(100), nullable=False),
        # low | medium | high | critical
        sa.Column("severity", sa.String(20), nullable=False, server_default="medium"),
        # new | acknowledged | resolved | suppressed
        sa.Column("status", sa.String(50), nullable=False, server_default="new"),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("body", sa.Text, nullable=False, server_default=""),
        # Structured metadata: rule name, threshold, context, etc.
        sa.Column("metadata", _JSONB, nullable=False, server_default="{}"),
        sa.Column("acknowledged_by", sa.String(255), nullable=True),
        sa.Column("acknowledged_at", _TSTZ, nullable=True),
        sa.Column("resolved_at", _TSTZ, nullable=True),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_alerts_org_status", "alerts", ["org_id", "status"])
    op.create_index("ix_alerts_entity", "alerts", ["entity_type", "entity_id"])
    op.create_index("ix_alerts_org_created", "alerts", ["org_id", "created_at"])
    op.create_index("ix_alerts_severity", "alerts", ["severity"])
    op.create_index(
        "ix_alerts_open",
        "alerts",
        ["org_id", "created_at"],
        postgresql_where=sa.text("status = 'new'"),
    )

    # ── audit_logs ───────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "user_id",
            _UUID,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # Dot-separated action code: auth.login, case.created, etc.
        sa.Column("action", sa.String(100), nullable=False),
        # Entity category: user | transaction | case | api_key | organization
        sa.Column("resource_type", sa.String(100), nullable=False),
        sa.Column("resource_id", sa.String(255), nullable=True),
        # Arbitrary structured context for the event.
        sa.Column("details", _JSONB, nullable=False, server_default="{}"),
        # IPv4 or IPv6 address (max 45 chars).
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(500), nullable=True),
        sa.Column("timestamp", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_audit_org_timestamp", "audit_logs", ["org_id", "timestamp"])
    op.create_index("ix_audit_action", "audit_logs", ["action"])
    op.create_index("ix_audit_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_resource", "audit_logs", ["resource_type", "resource_id"])
    # GIN index allows fast JSONB contains queries (@>) on the details column.
    op.create_index(
        "ix_audit_details_gin",
        "audit_logs",
        ["details"],
        postgresql_using="gin",
    )

    # ── usage_records ─────────────────────────────────────────────────────────
    op.create_table(
        "usage_records",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("endpoint", sa.String(255), nullable=False),
        sa.Column("method", sa.String(10), nullable=False),
        sa.Column("status_code", sa.SmallInteger, nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("request_size", sa.Integer, nullable=False, server_default="0"),
        sa.Column("response_size", sa.Integer, nullable=False, server_default="0"),
        # Counted unit for quota/billing purposes (usually 1 per request).
        sa.Column("billed_units", sa.Integer, nullable=False, server_default="1"),
        sa.Column("timestamp", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_usage_org_timestamp", "usage_records", ["org_id", "timestamp"])
    op.create_index("ix_usage_endpoint", "usage_records", ["endpoint"])
    op.create_index("ix_usage_status_code", "usage_records", ["status_code"])


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------

def downgrade() -> None:
    """Drop all baseline tables in reverse dependency order.

    WARNING: Complete data loss.  Requires explicit confirmation in production.
    """
    op.drop_table("usage_records")
    op.drop_table("audit_logs")
    op.drop_table("alerts")
    op.drop_table("feedback")
    op.drop_table("cases")
    op.drop_table("transactions")
    op.drop_table("api_keys")
    op.drop_table("users")
    op.drop_table("organizations")
