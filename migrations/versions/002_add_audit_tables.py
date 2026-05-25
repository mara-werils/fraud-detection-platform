"""Add audit trail tables — event sourcing and data retention policies.

Revision ID: 002
Revises:     001
Created:     2026-05-25 00:00:00 UTC

Description
-----------
Adds two tables that support the compliance and event-sourcing layer:

audit_events
    Event-sourcing style immutable log.  Unlike ``audit_logs`` (which records
    *who did what* for security monitoring), ``audit_events`` captures *business
    domain events* in a strongly-typed, replayable format.  Every event has:
      - a canonical ``event_type`` (e.g. "transaction.scored", "case.resolved")
      - an ``aggregate_type`` / ``aggregate_id`` pair identifying the root entity
      - a full ``payload`` JSONB blob (immutable snapshot at event time)
      - a ``sequence_number`` monotonically increasing per aggregate for ordering
      - a ``correlation_id`` to group causally related events across aggregates
      - a ``causation_id`` pointing to the upstream event that caused this one

    Consumers can rebuild any read model by replaying events from
    ``sequence_number = 1`` for a given aggregate, or subscribe to a Kafka
    topic that mirrors these rows.

data_retention_policies
    Per-org, per-table configuration for GDPR Art. 5(1)(e) storage limitation.
    Overrides the platform defaults in ``scoring/services/gdpr.py``.  The
    scheduled purge job reads this table instead of hard-coded constants.

Rollback Notes
--------------
Drops both tables — no FK references exist from other tables, so rollback is
safe.  Existing audit_logs rows are unaffected.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

# ---------------------------------------------------------------------------
# Revision identifiers
# ---------------------------------------------------------------------------

revision: str = "002"
down_revision: str | Sequence[str] | None = "001"
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
    # ── audit_events ─────────────────────────────────────────────────────────
    # Event sourcing log — immutable, append-only.
    op.create_table(
        "audit_events",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        # Domain event name in dot-notation: transaction.scored | case.resolved
        sa.Column("event_type", sa.String(150), nullable=False),
        # Root aggregate type: transaction | case | user | organization | api_key
        sa.Column("aggregate_type", sa.String(100), nullable=False),
        # UUID or external string identifier of the root aggregate.
        sa.Column("aggregate_id", sa.String(255), nullable=False),
        # Tenant scoping — NULL for platform-level events (e.g. system.startup).
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # Actor who triggered the event (user UUID, api_key UUID, or "system").
        sa.Column("actor_type", sa.String(50), nullable=False, server_default="system"),
        sa.Column("actor_id", sa.String(255), nullable=True),
        # Monotonically increasing per aggregate_type + aggregate_id.
        # Application code uses SELECT MAX(sequence_number) + 1 with a lock.
        sa.Column(
            "sequence_number",
            sa.BigInteger,
            nullable=False,
            server_default="1",
        ),
        # Immutable payload snapshot at the time the event was emitted.
        # Schema varies by event_type — document in separate event catalog.
        sa.Column("payload", _JSONB, nullable=False, server_default="{}"),
        # Metadata: schema version, source service, environment, etc.
        sa.Column("metadata", _JSONB, nullable=False, server_default="{}"),
        # Groups causally related events spanning multiple aggregates.
        sa.Column("correlation_id", _UUID, nullable=True),
        # ID of the upstream event that directly caused this event.
        sa.Column("causation_id", _UUID, nullable=True),
        # Idempotency key — application supplies this to prevent duplicate events.
        sa.Column("idempotency_key", sa.String(255), nullable=True),
        # Schema version of the payload for forward-compatible replay.
        sa.Column("schema_version", sa.SmallInteger, nullable=False, server_default="1"),
        sa.Column("occurred_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("recorded_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    # Unique constraint on sequence numbers per aggregate — enforces ordering.
    op.create_unique_constraint(
        "uq_audit_events_aggregate_seq",
        "audit_events",
        ["aggregate_type", "aggregate_id", "sequence_number"],
    )
    # Unique idempotency key (when provided) prevents duplicate event ingestion.
    op.create_unique_constraint(
        "uq_audit_events_idempotency",
        "audit_events",
        ["idempotency_key"],
        # NULL values are not considered equal in PostgreSQL, so we add a
        # partial index to enforce uniqueness only for non-NULL keys.
    )
    op.create_index(
        "ix_audit_events_aggregate",
        "audit_events",
        ["aggregate_type", "aggregate_id", "sequence_number"],
    )
    op.create_index(
        "ix_audit_events_event_type",
        "audit_events",
        ["event_type"],
    )
    op.create_index(
        "ix_audit_events_org_occurred",
        "audit_events",
        ["org_id", "occurred_at"],
    )
    op.create_index(
        "ix_audit_events_correlation",
        "audit_events",
        ["correlation_id"],
        postgresql_where=sa.text("correlation_id IS NOT NULL"),
    )
    op.create_index(
        "ix_audit_events_occurred_at",
        "audit_events",
        ["occurred_at"],
    )
    # GIN index for JSONB payload queries (@> operator).
    op.create_index(
        "ix_audit_events_payload_gin",
        "audit_events",
        ["payload"],
        postgresql_using="gin",
    )

    # ── data_retention_policies ───────────────────────────────────────────────
    op.create_table(
        "data_retention_policies",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # PostgreSQL table name this policy applies to.
        # Must match an actual table: transactions | cases | feedback |
        # usage_records | audit_logs | audit_events
        sa.Column("table_name", sa.String(100), nullable=False),
        # Number of days to retain rows.  NULL = retain indefinitely.
        sa.Column("retention_days", sa.Integer, nullable=True),
        # Column used for age comparison (default: "created_at" or "timestamp").
        sa.Column("age_column", sa.String(100), nullable=False, server_default="created_at"),
        # SQL WHERE fragment appended to the DELETE — allows partial purges.
        # Example: "status = 'closed'" to only purge resolved cases.
        sa.Column("filter_clause", sa.Text, nullable=True),
        # active | paused | archived
        sa.Column("policy_status", sa.String(50), nullable=False, server_default="active"),
        # Legal basis under GDPR Art. 6 for retaining this category.
        sa.Column(
            "legal_basis",
            sa.String(255),
            nullable=False,
            server_default="legitimate_interest_fraud_prevention",
        ),
        # Who approved this retention policy and when.
        sa.Column("approved_by", sa.String(255), nullable=True),
        sa.Column("approved_at", _TSTZ, nullable=True),
        # Last time the automated purge job ran this policy successfully.
        sa.Column("last_purge_at", _TSTZ, nullable=True),
        # Row count deleted in the most recent purge run (for audit reporting).
        sa.Column("last_purge_rows", sa.BigInteger, nullable=True),
        sa.Column("notes", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    # Enforce one policy per (org, table) pair.
    op.create_unique_constraint(
        "uq_retention_org_table",
        "data_retention_policies",
        ["org_id", "table_name"],
    )
    op.create_index(
        "ix_retention_org_id",
        "data_retention_policies",
        ["org_id"],
    )
    op.create_index(
        "ix_retention_status",
        "data_retention_policies",
        ["policy_status"],
        postgresql_where=sa.text("policy_status = 'active'"),
    )

    # ── Seed platform-default retention policies ───────────────────────────────
    # These match RETENTION_DAYS in scoring/services/gdpr.py and act as
    # defaults for newly onboarded orgs until they configure their own.
    op.execute(
        sa.text(
            """
            INSERT INTO data_retention_policies
                (org_id, table_name, retention_days, age_column, legal_basis, approved_by, notes)
            SELECT
                o.id,
                t.table_name,
                t.days,
                t.age_col,
                'legitimate_interest_fraud_prevention',
                'system',
                'Platform default — auto-seeded by migration 002'
            FROM organizations o
            CROSS JOIN (
                VALUES
                    ('transactions',   365,  'timestamp'),
                    ('cases',          730,  'created_at'),
                    ('feedback',       730,  'created_at'),
                    ('usage_records',   90,  'timestamp'),
                    ('audit_logs',    2555,  'timestamp'),
                    ('audit_events',  2555,  'occurred_at')
            ) AS t(table_name, days, age_col)
            ON CONFLICT (org_id, table_name) DO NOTHING
            """
        )
    )


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------

def downgrade() -> None:
    op.drop_table("data_retention_policies")
    op.drop_table("audit_events")
