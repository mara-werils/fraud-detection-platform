"""Add model registry — versioning, experiments, metrics, and A/B assignments.

Revision ID: 004
Revises:     003
Created:     2026-05-25 00:00:00 UTC

Description
-----------
Adds four tables that form a lightweight MLflow-style model registry inside
the platform's own database.  These tables let the platform:

  * Track every trained model artifact with a canonical version string and
    rich metadata (framework, hyperparameters, training dataset provenance).
  * Record experiment runs so that model candidates can be compared before
    promotion to production.
  * Store per-cohort evaluation metrics (precision, recall, AUC, KS, PSI, …)
    at multiple points in time for drift monitoring.
  * Persist A/B test group assignments per (org, user) so that the scoring
    service delivers consistent model experiences without re-computing the
    hash bucket on every request.

model_versions
    Canonical registry of trained model artifacts.  Each row represents a
    unique, immutable snapshot of a model.  The ``stage`` column drives
    promotion workflow: development → staging → production → archived.
    Only one version per (org_id, model_name, stage='production') should be
    active at a time — enforced by a partial unique index.

model_experiments
    A parent experiment groups multiple model_versions trained for the same
    objective (e.g. "Q2-2026 card-not-present XGBoost tuning sprint").
    Experiment tracking enables team-level visibility and rollback decisions.

model_metrics
    Time-series metric snapshots linked to a model_version.  Written by:
      - The ML pipeline after offline evaluation on a hold-out set.
      - The drift detector service during live traffic monitoring.
      - A/B testing analysis jobs that compute online metrics per cohort.

    The ``evaluation_set`` column distinguishes metric origins:
    train | validation | test | production | ab_test_control | ab_test_treatment

ab_test_assignments
    Persists which A/B test variant each (org_id, user_id) was assigned to.
    Storing assignments prevents the hash bucket from drifting if user IDs
    change or the experiment traffic split is adjusted mid-test.

    The ``override_variant`` column allows analysts to force a specific user
    into a variant for debugging or customer-service purposes.

Rollback Notes
--------------
Drops all four tables in reverse FK order.  model_metrics and
ab_test_assignments reference model_versions, so they are dropped first.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

# ---------------------------------------------------------------------------
# Revision identifiers
# ---------------------------------------------------------------------------

revision: str = "004"
down_revision: str | Sequence[str] | None = "003"
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
    # ── model_experiments ─────────────────────────────────────────────────────
    # Create experiments before model_versions (FK target).
    op.create_table(
        "model_experiments",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,  # NULL = platform-managed experiment
        ),
        # Human-readable experiment name: "q2-xgboost-card-not-present-tuning"
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        # Experiment objective: fraud_detection | chargeback_prediction | etc.
        sa.Column("objective", sa.String(100), nullable=False, server_default="fraud_detection"),
        # active | completed | archived | failed
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        # Primary metric used for model selection (e.g. "auc_roc", "f1_score").
        sa.Column("primary_metric", sa.String(100), nullable=True),
        # Direction of optimisation for primary_metric: maximize | minimize
        sa.Column("metric_direction", sa.String(20), nullable=False, server_default="maximize"),
        # Version string of the winning model_version (populated on completion).
        sa.Column("best_version", sa.String(100), nullable=True),
        # Structured experiment config: dataset splits, search space, seeds.
        sa.Column("config", _JSONB, nullable=False, server_default="{}"),
        # Tags for filtering: ["xgboost", "card-not-present", "q2-2026"]
        sa.Column("tags", _JSONB, nullable=False, server_default="[]"),
        # Who owns this experiment.
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("completed_at", _TSTZ, nullable=True),
    )
    op.create_unique_constraint(
        "uq_model_experiments_org_name",
        "model_experiments",
        ["org_id", "name"],
    )
    op.create_index("ix_model_experiments_org_id", "model_experiments", ["org_id"])
    op.create_index("ix_model_experiments_status", "model_experiments", ["status"])
    op.create_index("ix_model_experiments_objective", "model_experiments", ["objective"])
    op.create_index(
        "ix_model_experiments_tags_gin",
        "model_experiments",
        ["tags"],
        postgresql_using="gin",
    )

    # ── model_versions ────────────────────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,  # NULL = platform-managed (shared across orgs)
        ),
        sa.Column(
            "experiment_id",
            _UUID,
            sa.ForeignKey("model_experiments.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # Logical model family name: "fraud-xgboost" | "fraud-rule-engine" | "fraud-gnn"
        sa.Column("model_name", sa.String(255), nullable=False),
        # SemVer or date-stamped string: "1.4.2" | "2026-05-25-abc1234"
        sa.Column("version", sa.String(100), nullable=False),
        # Canonical model URI: s3://bucket/models/fraud-xgboost/v1.4.2/model.pkl
        # or mlflow://mlflow-server/models/fraud-xgboost/versions/12
        sa.Column("artifact_uri", sa.String(1024), nullable=True),
        # SHA-256 digest of the serialized artifact for integrity verification.
        sa.Column("artifact_hash", sa.String(64), nullable=True),
        # ML framework: xgboost | sklearn | pytorch | lightgbm | rule-engine | ensemble
        sa.Column("framework", sa.String(100), nullable=False, server_default="xgboost"),
        # Promotion stage: development | staging | production | archived | shadow
        # shadow = receives live traffic for logging but decisions are not used
        sa.Column("stage", sa.String(50), nullable=False, server_default="development"),
        # description of what changed versus the previous version.
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        # Full hyperparameter dict: {n_estimators, max_depth, learning_rate, …}
        sa.Column("hyperparameters", _JSONB, nullable=False, server_default="{}"),
        # Feature names in the order expected by the model's predict() call.
        sa.Column("feature_names", _JSONB, nullable=False, server_default="[]"),
        # Training dataset provenance: {name, version, row_count, date_range, hash}
        sa.Column("training_data_ref", _JSONB, nullable=False, server_default="{}"),
        # Validation dataset provenance.
        sa.Column("validation_data_ref", _JSONB, nullable=False, server_default="{}"),
        # Summary of offline evaluation metrics at promotion time.
        sa.Column("eval_metrics", _JSONB, nullable=False, server_default="{}"),
        # Threshold configuration: {block: 0.8, review: 0.5}
        sa.Column("thresholds", _JSONB, nullable=False, server_default="{}"),
        # Arbitrary key/value metadata: git_sha, training_duration_s, etc.
        sa.Column("metadata", _JSONB, nullable=False, server_default="{}"),
        # Tags for search: ["production-candidate", "card-not-present"]
        sa.Column("tags", _JSONB, nullable=False, server_default="[]"),
        # Lineage: which version was this trained from (for incremental fine-tuning).
        sa.Column("parent_version_id", _UUID, nullable=True),
        sa.Column("trained_by", sa.String(255), nullable=True),
        sa.Column("reviewed_by", sa.String(255), nullable=True),
        sa.Column("deployed_by", sa.String(255), nullable=True),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("deployed_at", _TSTZ, nullable=True),
        sa.Column("archived_at", _TSTZ, nullable=True),
        # Deprecation warning shown in UI / API responses when stage=archived.
        sa.Column("deprecation_reason", sa.Text, nullable=True),
    )
    # Uniqueness across (org, name, version) — same version string cannot appear twice.
    op.create_unique_constraint(
        "uq_model_versions_name_version",
        "model_versions",
        ["org_id", "model_name", "version"],
    )
    # Only ONE production model per (org, model_name) at any time.
    op.create_index(
        "ix_model_versions_one_production",
        "model_versions",
        ["org_id", "model_name"],
        unique=True,
        postgresql_where=sa.text("stage = 'production'"),
    )
    op.create_index("ix_model_versions_org_id", "model_versions", ["org_id"])
    op.create_index("ix_model_versions_model_name", "model_versions", ["model_name"])
    op.create_index("ix_model_versions_stage", "model_versions", ["stage"])
    op.create_index("ix_model_versions_experiment", "model_versions", ["experiment_id"])
    op.create_index(
        "ix_model_versions_artifact_hash",
        "model_versions",
        ["artifact_hash"],
        postgresql_where=sa.text("artifact_hash IS NOT NULL"),
    )
    op.create_index(
        "ix_model_versions_tags_gin",
        "model_versions",
        ["tags"],
        postgresql_using="gin",
    )

    # ── model_metrics ─────────────────────────────────────────────────────────
    op.create_table(
        "model_metrics",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "model_version_id",
            _UUID,
            sa.ForeignKey("model_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
        # Evaluation set origin: train | validation | test | production |
        # ab_test_control | ab_test_treatment | shadow
        sa.Column("evaluation_set", sa.String(100), nullable=False),
        # ── Classification metrics ─────────────────────────────────────────────
        sa.Column("auc_roc", sa.Float, nullable=True),
        sa.Column("auc_pr", sa.Float, nullable=True),
        sa.Column("f1_score", sa.Float, nullable=True),
        sa.Column("precision", sa.Float, nullable=True),
        sa.Column("recall", sa.Float, nullable=True),
        sa.Column("specificity", sa.Float, nullable=True),
        # Area under the KS (Kolmogorov-Smirnov) curve — standard risk metric.
        sa.Column("ks_statistic", sa.Float, nullable=True),
        sa.Column("log_loss", sa.Float, nullable=True),
        sa.Column("brier_score", sa.Float, nullable=True),
        # ── Threshold-dependent metrics ────────────────────────────────────────
        # Computed at the model's configured block/review thresholds.
        sa.Column("block_precision", sa.Float, nullable=True),
        sa.Column("block_recall", sa.Float, nullable=True),
        sa.Column("review_precision", sa.Float, nullable=True),
        sa.Column("review_recall", sa.Float, nullable=True),
        sa.Column("false_positive_rate", sa.Float, nullable=True),
        sa.Column("false_negative_rate", sa.Float, nullable=True),
        # ── Business metrics ───────────────────────────────────────────────────
        # Dollar value of fraud caught vs. dollar value of false positives blocked.
        sa.Column("fraud_caught_amount_usd", sa.Numeric(18, 4), nullable=True),
        sa.Column("false_positive_amount_usd", sa.Numeric(18, 4), nullable=True),
        # Ratio of fraud_caught_amount / total_fraud_amount (coverage).
        sa.Column("fraud_coverage_ratio", sa.Float, nullable=True),
        # ── Drift metrics ─────────────────────────────────────────────────────
        # Population Stability Index — measures score distribution shift.
        sa.Column("psi", sa.Float, nullable=True),
        # Jensen-Shannon divergence of score distributions (current vs. baseline).
        sa.Column("js_divergence", sa.Float, nullable=True),
        # Covariate drift metric per feature (JSONB dict: {feature: psi_value}).
        sa.Column("feature_drift", _JSONB, nullable=False, server_default="{}"),
        # ── Throughput metrics ─────────────────────────────────────────────────
        # Sample size used to compute these metrics.
        sa.Column("sample_count", sa.BigInteger, nullable=True),
        sa.Column("positive_count", sa.BigInteger, nullable=True),   # actual fraud
        sa.Column("negative_count", sa.BigInteger, nullable=True),   # actual legit
        # Date range of the evaluation data.
        sa.Column("eval_start_date", _TSTZ, nullable=True),
        sa.Column("eval_end_date", _TSTZ, nullable=True),
        # ── Extra metrics ──────────────────────────────────────────────────────
        # Open-ended JSONB for metrics not captured by the fixed columns above.
        sa.Column("extra_metrics", _JSONB, nullable=False, server_default="{}"),
        sa.Column("computed_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        # Service that produced this row: ml_pipeline | drift_detector | ab_analyzer
        sa.Column("computed_by", sa.String(100), nullable=False, server_default="ml_pipeline"),
    )
    op.create_index(
        "ix_model_metrics_version_id",
        "model_metrics",
        ["model_version_id"],
    )
    op.create_index(
        "ix_model_metrics_version_eval",
        "model_metrics",
        ["model_version_id", "evaluation_set", "computed_at"],
    )
    op.create_index(
        "ix_model_metrics_computed_at",
        "model_metrics",
        ["computed_at"],
    )
    op.create_index(
        "ix_model_metrics_auc_roc",
        "model_metrics",
        ["auc_roc"],
        postgresql_where=sa.text("auc_roc IS NOT NULL"),
    )
    op.create_index(
        "ix_model_metrics_psi",
        "model_metrics",
        ["model_version_id", "psi", "computed_at"],
        postgresql_where=sa.text("psi IS NOT NULL"),
    )
    op.create_index(
        "ix_model_metrics_drift_gin",
        "model_metrics",
        ["feature_drift"],
        postgresql_using="gin",
    )

    # ── ab_test_assignments ───────────────────────────────────────────────────
    op.create_table(
        "ab_test_assignments",
        sa.Column("id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "org_id",
            _UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Experiment name — matches the scoring engine's ABTestConfig.name.
        sa.Column("experiment_name", sa.String(255), nullable=False),
        sa.Column(
            "experiment_id",
            _UUID,
            sa.ForeignKey("model_experiments.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # Merchant-supplied user_id used as the hash input for bucket assignment.
        sa.Column("user_id", sa.String(100), nullable=False),
        # Variant name as configured in the ABTestConfig groups: "control" | "treatment_a"
        sa.Column("variant", sa.String(100), nullable=False),
        # Raw hash bucket value [0, 100) for auditability.
        sa.Column("bucket", sa.SmallInteger, nullable=False),
        # Optional analyst override — bypasses hash assignment.
        sa.Column("override_variant", sa.String(100), nullable=True),
        sa.Column("override_reason", sa.Text, nullable=True),
        sa.Column("override_by", sa.String(255), nullable=True),
        sa.Column("override_at", _TSTZ, nullable=True),
        # Whether this assignment is still active for the running experiment.
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("TRUE")),
        # model_version_id being tested in this variant (for metric attribution).
        sa.Column(
            "model_version_id",
            _UUID,
            sa.ForeignKey("model_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("assigned_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("expires_at", _TSTZ, nullable=True),
        sa.Column("created_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", _TSTZ, nullable=False, server_default=sa.text("NOW()")),
    )
    # One active assignment per (org, experiment, user).
    op.create_unique_constraint(
        "uq_ab_assignments_active",
        "ab_test_assignments",
        ["org_id", "experiment_name", "user_id"],
        # NOTE: uniqueness applies across all rows regardless of is_active.
        # If the experiment is reset, old rows must be deactivated (is_active=FALSE)
        # before a new assignment for the same user is inserted.
    )
    op.create_index(
        "ix_ab_assignments_org_experiment",
        "ab_test_assignments",
        ["org_id", "experiment_name"],
    )
    op.create_index(
        "ix_ab_assignments_user",
        "ab_test_assignments",
        ["org_id", "user_id"],
    )
    op.create_index(
        "ix_ab_assignments_variant",
        "ab_test_assignments",
        ["experiment_name", "variant"],
    )
    op.create_index(
        "ix_ab_assignments_active",
        "ab_test_assignments",
        ["org_id", "experiment_name", "is_active"],
        postgresql_where=sa.text("is_active = TRUE"),
    )
    op.create_index(
        "ix_ab_assignments_model_version",
        "ab_test_assignments",
        ["model_version_id"],
        postgresql_where=sa.text("model_version_id IS NOT NULL"),
    )
    op.create_index(
        "ix_ab_assignments_expires",
        "ab_test_assignments",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------

def downgrade() -> None:
    # Drop in reverse FK dependency order.
    op.drop_table("ab_test_assignments")
    op.drop_table("model_metrics")
    op.drop_table("model_versions")
    op.drop_table("model_experiments")
