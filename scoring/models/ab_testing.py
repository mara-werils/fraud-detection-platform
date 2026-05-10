"""A/B testing framework for fraud scoring models.

Provides hash-based traffic splitting, per-group metric tracking, and
periodic flushing to ClickHouse for long-term analysis.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GroupConfig:
    """Configuration for an A/B test group.

    Args:
        name: Human-readable group name.
        range_start: Lower bound (inclusive) of the hash bucket [0, 100).
        range_end: Upper bound (exclusive) of the hash bucket [0, 100).
        model_weights: Optional override for ensemble weights.
    """

    name: str
    range_start: int
    range_end: int
    model_weights: dict[str, float] | None = None


@dataclass
class GroupMetrics:
    """Accumulated metrics for a single A/B test group."""

    total_scored: int = 0
    total_flagged: int = 0
    total_blocked: int = 0
    total_fraud_caught: int = 0
    total_false_positives: int = 0
    total_latency_ms: float = 0.0
    score_sum: float = 0.0

    @property
    def fraud_catch_rate(self) -> float:
        """Proportion of actual frauds caught (recall)."""
        if self.total_fraud_caught + self.total_false_positives == 0:
            return 0.0
        return self.total_fraud_caught / (self.total_fraud_caught + self.total_false_positives)

    @property
    def false_positive_rate(self) -> float:
        """Proportion of flagged transactions that were not fraud."""
        if self.total_flagged == 0:
            return 0.0
        return self.total_false_positives / self.total_flagged

    @property
    def avg_latency_ms(self) -> float:
        """Average scoring latency in milliseconds."""
        if self.total_scored == 0:
            return 0.0
        return self.total_latency_ms / self.total_scored

    @property
    def avg_score(self) -> float:
        if self.total_scored == 0:
            return 0.0
        return self.score_sum / self.total_scored

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_scored": self.total_scored,
            "total_flagged": self.total_flagged,
            "total_blocked": self.total_blocked,
            "total_fraud_caught": self.total_fraud_caught,
            "total_false_positives": self.total_false_positives,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_score": round(self.avg_score, 4),
            "fraud_catch_rate": round(self.fraud_catch_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
        }


# Default groups: 50/50 split
DEFAULT_GROUPS: tuple[GroupConfig, ...] = (
    GroupConfig(name="control", range_start=0, range_end=50),
    GroupConfig(name="challenger", range_start=50, range_end=100),
)


class ABTestManager:
    """Hash-based A/B testing framework for fraud scoring experiments.

    Users are deterministically assigned to groups via
    ``hash(user_id) % 100``.  Metrics are tracked in-memory and
    periodically flushed to ClickHouse.

    Args:
        experiment_name: Name of the A/B experiment.
        groups: Group configurations. Defaults to a 50/50
            control/challenger split.
        clickhouse_client: Optional async ClickHouse client for metric
            persistence.
        flush_interval_seconds: How often to flush metrics to ClickHouse.
    """

    def __init__(
        self,
        experiment_name: str = "default",
        groups: tuple[GroupConfig, ...] | list[GroupConfig] | None = None,
        clickhouse_client: Any | None = None,
        flush_interval_seconds: int = 60,
    ) -> None:
        self._experiment_name = experiment_name
        self._groups = tuple(groups) if groups else DEFAULT_GROUPS
        self._clickhouse = clickhouse_client
        self._flush_interval = flush_interval_seconds

        # Validate that groups cover the full range without overlap
        self._validate_groups()

        # Per-group metrics
        self._metrics: dict[str, GroupMetrics] = {g.name: GroupMetrics() for g in self._groups}

        # Background flush task handle
        self._flush_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_groups(self) -> None:
        """Ensure groups cover [0, 100) without gaps or overlaps."""
        sorted_groups = sorted(self._groups, key=lambda g: g.range_start)
        for i, group in enumerate(sorted_groups):
            if group.range_start < 0 or group.range_end > 100:
                raise ValueError(
                    f"Group '{group.name}' range [{group.range_start}, {group.range_end}) "
                    "is out of bounds [0, 100)"
                )
            if group.range_start >= group.range_end:
                raise ValueError(
                    f"Group '{group.name}' has invalid range: "
                    f"[{group.range_start}, {group.range_end})"
                )
            if i > 0 and group.range_start != sorted_groups[i - 1].range_end:
                raise ValueError(
                    f"Gap or overlap between group '{sorted_groups[i - 1].name}' and '{group.name}'"
                )

    # ------------------------------------------------------------------
    # Group assignment
    # ------------------------------------------------------------------

    def assign_group(self, user_id: UUID) -> GroupConfig:
        """Deterministically assign a user to an A/B test group.

        Uses ``MD5(user_id) % 100`` for uniform distribution.

        Args:
            user_id: The user to assign.

        Returns:
            The GroupConfig the user belongs to.
        """
        bucket = self._hash_bucket(user_id)
        for group in self._groups:
            if group.range_start <= bucket < group.range_end:
                return group

        # Should never happen if groups are validated
        logger.error("ab_test_no_group_matched", bucket=bucket)
        return self._groups[0]

    @staticmethod
    def _hash_bucket(user_id: UUID) -> int:
        """Map a user_id to a bucket in [0, 100)."""
        digest = hashlib.md5(str(user_id).encode()).hexdigest()
        return int(digest, 16) % 100

    # ------------------------------------------------------------------
    # Metric recording
    # ------------------------------------------------------------------

    def record_scoring(
        self,
        group_name: str,
        fraud_score: float,
        is_flagged: bool,
        is_blocked: bool,
        latency_ms: float,
    ) -> None:
        """Record the result of scoring a transaction for a group.

        Args:
            group_name: The group this transaction belongs to.
            fraud_score: The ensemble fraud score.
            is_flagged: Whether the transaction was flagged (review or block).
            is_blocked: Whether the transaction was blocked.
            latency_ms: Scoring latency in milliseconds.
        """
        metrics = self._metrics.get(group_name)
        if metrics is None:
            logger.warning("ab_test_unknown_group", group=group_name)
            return

        metrics.total_scored += 1
        metrics.score_sum += fraud_score
        metrics.total_latency_ms += latency_ms
        if is_flagged:
            metrics.total_flagged += 1
        if is_blocked:
            metrics.total_blocked += 1

    def record_feedback(
        self,
        group_name: str,
        was_fraud: bool,
        was_flagged: bool,
    ) -> None:
        """Record ground-truth feedback for a scored transaction.

        Args:
            group_name: The group this transaction belongs to.
            was_fraud: Whether the transaction was actually fraudulent.
            was_flagged: Whether the transaction was flagged by scoring.
        """
        metrics = self._metrics.get(group_name)
        if metrics is None:
            logger.warning("ab_test_unknown_group", group=group_name)
            return

        if was_fraud and was_flagged:
            metrics.total_fraud_caught += 1
        elif not was_fraud and was_flagged:
            metrics.total_false_positives += 1

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> dict[str, Any]:
        """Return current A/B test results for all groups.

        Returns:
            Dict with experiment metadata and per-group metrics.
        """
        return {
            "experiment": self._experiment_name,
            "groups": {name: metrics.to_dict() for name, metrics in self._metrics.items()},
        }

    # ------------------------------------------------------------------
    # ClickHouse flush
    # ------------------------------------------------------------------

    async def start_flush_loop(self) -> None:
        """Start a background task to periodically flush metrics to ClickHouse."""
        if self._flush_task is not None:
            return
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(
            "ab_test_flush_loop_started",
            experiment=self._experiment_name,
            interval=self._flush_interval,
        )

    async def stop_flush_loop(self) -> None:
        """Cancel the background flush task."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
            logger.info("ab_test_flush_loop_stopped", experiment=self._experiment_name)

    async def _flush_loop(self) -> None:
        """Periodically flush metrics to ClickHouse."""
        while True:
            await asyncio.sleep(self._flush_interval)
            await self._flush_to_clickhouse()

    async def _flush_to_clickhouse(self) -> None:
        """Write current metrics snapshot to ClickHouse.

        Silently logs errors if ClickHouse is unavailable.
        """
        if self._clickhouse is None:
            logger.debug("ab_test_flush_skipped", reason="no_clickhouse_client")
            return

        try:
            results = self.get_results()
            timestamp = int(time.time())

            rows: list[dict[str, Any]] = []
            for group_name, metrics in results["groups"].items():
                row = {
                    "experiment": self._experiment_name,
                    "group_name": group_name,
                    "timestamp": timestamp,
                    **metrics,
                }
                rows.append(row)

            await self._clickhouse.execute(
                "INSERT INTO ab_test_metrics VALUES",
                rows,
            )
            logger.info(
                "ab_test_metrics_flushed",
                experiment=self._experiment_name,
                groups=len(rows),
            )
        except Exception as exc:
            logger.error(
                "ab_test_flush_error",
                experiment=self._experiment_name,
                error=str(exc),
            )
