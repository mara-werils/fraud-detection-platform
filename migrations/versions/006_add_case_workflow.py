"""Add enhanced case workflow — assignment, SLA, notes, and timeline.

Revision ID: 006
Revises:     005
Created:     2026-06-08 00:00:00 UTC

Description
-----------
Extends the existing ``cases`` table with columns that support a full
investigation workflow and adds two new child tables for audit-friendly
collaboration.

cases (new columns)
    * ``assigned_to`` — analyst currently responsible for the case.
    * ``priority`` — integer priority (1 = critical … 5 = informational).
    * ``sla_deadline`` — timestamp by which the case must be resolved to
      meet the SLA contract.
    * ``resolved_at`` — when the case was closed / resolved.
    * ``resolution_type`` — outcome classification: confirmed_fraud,
      false_positive, chargeback, no_action, escalated.
    * ``template_id`` — optional FK-free reference to a case template that
      pre-populates fields when a case is opened.

case_notes
    Free-form analyst notes attached to a case.  Each note is an immutable
    append-only record (no UPDATE expected) so that the audit trail is
    preserved.

case_timeline
    Machine-generated event log for a case.  Every state change (status
    transition, assignment, priority change, SLA breach) is recorded as a
    timeline entry with a structured ``details`` JSONB payload.

Rollback Notes
--------------
Drops ``case_timeline`` and ``case_notes`` tables, then removes the six
columns added to ``cases``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

# ---------------------------------------------------------------------------
# Revision identifiers
# ---------------------------------------------------------------------------

revision: str = "006"
down_revision: str | Sequence[str] | None = "005"
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
    # ── Extend cases table ───────────────────────────────────────────────────
    op.add_column("cases", sa.Column("assigned_to", sa.String(255), nullable=True))
    op.add_column("cases", sa.Column("priority", sa.Integer, nullable=True))
    op.add_column("cases", sa.Column("sla_deadline", _TSTZ, nullable=True))
    op.add_column("cases", sa.Column("resolved_at", _TSTZ, nullable=True))
    op.add_column("cases", sa.Column("resolution_type", sa.String(50), nullable=True))
    op.add_column("cases", sa.Column("template_id", _UUID, nullable=True))

    op.create_index("ix_cases_assigned_to", "cases", ["assigned_to"])
    op.create_index("ix_cases_priority", "cases", ["priority"])
    op.create_index(
        "ix_cases_sla_deadline",
        "cases",
        ["sla_deadline"],
        postgresql_where=sa.text("sla_deadline IS NOT NULL"),
    )
    op.create_index(
        "ix_cases_resolved_at",
        "cases",
        ["resolved_at"],
        postgresql_where=sa.text("resolved_at IS NOT NULL"),
    )
    op.create_index("ix_cases_resolution_type", "cases", ["resolution_type"])
    op.create_index(
        "ix_cases_template_id",
        "cases",
        ["template_id"],
        postgresql_where=sa.text("template_id IS NOT NULL"),
    )

    # ── case_notes ───────────────────────────────────────────────────────────
    op.create_table(
        "case_notes",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "case_id",
            _UUID,
            sa.ForeignKey("cases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Analyst or system identity that authored the note.
        sa.Column("author", sa.String(255), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_case_notes_case_id", "case_notes", ["case_id"])
    op.create_index("ix_case_notes_author", "case_notes", ["author"])
    op.create_index("ix_case_notes_created_at", "case_notes", ["created_at"])

    # ── case_timeline ────────────────────────────────────────────────────────
    op.create_table(
        "case_timeline",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "case_id",
            _UUID,
            sa.ForeignKey("cases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Event classification: status_change | assignment | priority_change |
        # sla_breach | note_added | escalation | resolution
        sa.Column("event_type", sa.String(100), nullable=False),
        # Structured payload with event-specific data.
        # e.g. {"from_status": "open", "to_status": "in_review", "changed_by": "analyst@co.com"}
        sa.Column("details", _JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_case_timeline_case_id", "case_timeline", ["case_id"])
    op.create_index("ix_case_timeline_event_type", "case_timeline", ["event_type"])
    op.create_index("ix_case_timeline_created_at", "case_timeline", ["created_at"])
    op.create_index(
        "ix_case_timeline_case_created",
        "case_timeline",
        ["case_id", "created_at"],
    )


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------

def downgrade() -> None:
    # Drop child tables first.
    op.drop_table("case_timeline")
    op.drop_table("case_notes")

    # Remove columns added to cases.
    op.drop_index("ix_cases_template_id", table_name="cases")
    op.drop_index("ix_cases_resolution_type", table_name="cases")
    op.drop_index("ix_cases_resolved_at", table_name="cases")
    op.drop_index("ix_cases_sla_deadline", table_name="cases")
    op.drop_index("ix_cases_priority", table_name="cases")
    op.drop_index("ix_cases_assigned_to", table_name="cases")

    op.drop_column("cases", "template_id")
    op.drop_column("cases", "resolution_type")
    op.drop_column("cases", "resolved_at")
    op.drop_column("cases", "sla_deadline")
    op.drop_column("cases", "priority")
    op.drop_column("cases", "assigned_to")
