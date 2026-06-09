"""Velocity check engine for multi-window transaction monitoring.

Tracks transaction velocity across multiple dimensions (user, card, device, IP)
and time windows (1min, 5min, 1hr, 24hr) using Redis sorted sets for efficient
windowed counting.

Supports three velocity types:
  - Count velocity: number of transactions in a window
  - Amount velocity: cumulative transaction amount in a window
  - Unique merchant velocity: distinct merchants seen in a window

Redis key format: vel:{dimension}:{entity_id}:{velocity_type}
Sorted set scores are Unix timestamps; members encode transaction details.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from prometheus_client import Counter, Histogram
from redis.asyncio import Redis

__all__ = [
    "Dimension",
    "VelocityType",
    "TimeWindow",
    "VelocityThreshold",
    "VelocityResult",
    "VelocityReport",
    "VelocityEngine",
    "DEFAULT_THRESHOLDS",
]

logger = structlog.get_logger(__name__)

VELOCITY_CHECKS = Counter(
    "velocity_checks_total",
    "Velocity checks performed",
    ["dimension", "window", "velocity_type"],
)

VELOCITY_BREACHES = Counter(
    "velocity_breaches_total",
    "Velocity threshold breaches detected",
    ["dimension", "window", "velocity_type"],
)

VELOCITY_LATENCY = Histogram(
    "velocity_check_latency_seconds",
    "Latency of velocity check operations",
    ["operation"],
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)


class Dimension(str, Enum):
    """Entity dimension for velocity tracking."""

    USER = "user"
    CARD = "card"
    DEVICE = "device"
    IP = "ip"


class VelocityType(str, Enum):
    """Type of velocity measurement."""

    COUNT = "count"
    AMOUNT = "amount"
    UNIQUE_MERCHANT = "unique_merchant"


class TimeWindow(Enum):
    """Pre-defined time windows for velocity checks."""

    ONE_MIN = 60
    FIVE_MIN = 300
    ONE_HOUR = 3600
    TWENTY_FOUR_HOUR = 86400

    @property
    def label(self) -> str:
        labels = {60: "1m", 300: "5m", 3600: "1h", 86400: "24h"}
        return labels[self.value]


@dataclass(frozen=True)
class VelocityThreshold:
    """A single velocity threshold configuration.

    Attributes:
        dimension: Which entity dimension to track.
        velocity_type: Type of velocity measurement.
        window: Time window for the check.
        limit: The threshold value that triggers a breach.
    """

    dimension: Dimension
    velocity_type: VelocityType
    window: TimeWindow
    limit: float


@dataclass
class VelocityResult:
    """Result of a single velocity check.

    Attributes:
        dimension: Dimension checked.
        velocity_type: Type of velocity measured.
        window: Time window used.
        current_value: The current velocity value.
        limit: The configured threshold.
        breached: Whether the threshold was exceeded.
    """

    dimension: Dimension
    velocity_type: VelocityType
    window: TimeWindow
    current_value: float
    limit: float
    breached: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "velocity_type": self.velocity_type.value,
            "window": self.window.label,
            "current_value": self.current_value,
            "limit": self.limit,
            "breached": self.breached,
        }


@dataclass
class VelocityReport:
    """Aggregated velocity check report for a transaction.

    Attributes:
        entity_id: The entity that was checked.
        results: Individual velocity check results.
        score: Normalized velocity risk score in [0, 1].
        any_breach: True if any threshold was breached.
    """

    entity_id: str
    results: list[VelocityResult] = field(default_factory=list)
    score: float = 0.0
    any_breach: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "score": round(self.score, 4),
            "any_breach": self.any_breach,
            "results": [r.to_dict() for r in self.results],
        }


# Default thresholds — production deployments should override via config.
DEFAULT_THRESHOLDS: list[VelocityThreshold] = [
    # Count velocity
    VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_MIN, 3),
    VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.FIVE_MIN, 8),
    VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.ONE_HOUR, 20),
    VelocityThreshold(Dimension.USER, VelocityType.COUNT, TimeWindow.TWENTY_FOUR_HOUR, 50),
    VelocityThreshold(Dimension.CARD, VelocityType.COUNT, TimeWindow.ONE_MIN, 2),
    VelocityThreshold(Dimension.CARD, VelocityType.COUNT, TimeWindow.FIVE_MIN, 5),
    # Amount velocity
    VelocityThreshold(Dimension.USER, VelocityType.AMOUNT, TimeWindow.ONE_HOUR, 5000.0),
    VelocityThreshold(Dimension.USER, VelocityType.AMOUNT, TimeWindow.TWENTY_FOUR_HOUR, 20000.0),
    VelocityThreshold(Dimension.CARD, VelocityType.AMOUNT, TimeWindow.ONE_HOUR, 3000.0),
    # Unique merchant velocity
    VelocityThreshold(Dimension.USER, VelocityType.UNIQUE_MERCHANT, TimeWindow.ONE_HOUR, 5),
    VelocityThreshold(Dimension.USER, VelocityType.UNIQUE_MERCHANT, TimeWindow.TWENTY_FOUR_HOUR, 15),
    VelocityThreshold(Dimension.DEVICE, VelocityType.UNIQUE_MERCHANT, TimeWindow.ONE_HOUR, 4),
    # IP velocity
    VelocityThreshold(Dimension.IP, VelocityType.COUNT, TimeWindow.FIVE_MIN, 10),
    VelocityThreshold(Dimension.IP, VelocityType.COUNT, TimeWindow.ONE_HOUR, 30),
]


class VelocityEngine:
    """Velocity check engine backed by Redis sorted sets.

    Each tracked event is stored in a Redis sorted set keyed by
    ``vel:{dimension}:{entity_id}:{velocity_type}``.  The score is
    the Unix timestamp and the member encodes transaction metadata
    (amount, merchant_id, unique nonce).

    Args:
        redis: An async Redis client (``redis.asyncio.Redis``).
        thresholds: Velocity thresholds to evaluate. Defaults to
            ``DEFAULT_THRESHOLDS``.
        key_prefix: Redis key namespace prefix.
        ttl_buffer: Extra seconds added to the largest window for key expiry.
    """

    def __init__(
        self,
        redis: Redis,
        thresholds: list[VelocityThreshold] | None = None,
        key_prefix: str = "vel:",
        ttl_buffer: int = 3600,
    ) -> None:
        self._redis = redis
        self._thresholds = thresholds or DEFAULT_THRESHOLDS
        self._key_prefix = key_prefix
        self._ttl_buffer = ttl_buffer

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _key(self, dimension: Dimension, entity_id: str, velocity_type: VelocityType) -> str:
        return f"{self._key_prefix}{dimension.value}:{entity_id}:{velocity_type.value}"

    @staticmethod
    def _member(
        transaction_id: str,
        amount: float,
        merchant_id: str | None = None,
    ) -> str:
        """Encode a sorted-set member containing transaction metadata."""
        return json.dumps(
            {"txn": transaction_id, "amt": amount, "mer": merchant_id or ""},
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_member(raw: str | bytes) -> dict[str, Any]:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Record a transaction
    # ------------------------------------------------------------------

    async def record(
        self,
        entities: dict[Dimension, str],
        transaction_id: str | None = None,
        amount: float = 0.0,
        merchant_id: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Record a transaction event across all provided dimensions.

        Args:
            entities: Mapping of dimension to entity ID (e.g. ``{Dimension.USER: "u123"}``).
            transaction_id: Unique transaction identifier; auto-generated if omitted.
            amount: Transaction amount.
            merchant_id: Merchant identifier (used for unique merchant velocity).
            timestamp: Unix timestamp of the event; defaults to ``time.time()``.
        """
        ts = timestamp or time.time()
        txn_id = transaction_id or uuid.uuid4().hex
        member = self._member(txn_id, amount, merchant_id)
        max_window = max(w.value for w in TimeWindow)
        ttl = max_window + self._ttl_buffer

        start = time.monotonic()
        try:
            pipe = self._redis.pipeline()
            for dimension, entity_id in entities.items():
                for vtype in VelocityType:
                    key = self._key(dimension, entity_id, vtype)
                    pipe.zadd(key, {member: ts})
                    pipe.expire(key, ttl)
            await pipe.execute()
        except Exception as exc:
            await logger.awarning("velocity_record_error", error=str(exc))
            raise
        finally:
            VELOCITY_LATENCY.labels(operation="record").observe(time.monotonic() - start)

    # ------------------------------------------------------------------
    # Prune expired entries
    # ------------------------------------------------------------------

    async def _prune(self, key: str, cutoff: float) -> None:
        """Remove sorted-set members older than *cutoff*."""
        await self._redis.zremrangebyscore(key, "-inf", cutoff)

    # ------------------------------------------------------------------
    # Query velocity values
    # ------------------------------------------------------------------

    async def _count_velocity(self, key: str, window_start: float) -> float:
        """Return the number of events in the window."""
        return float(await self._redis.zcount(key, window_start, "+inf"))

    async def _amount_velocity(self, key: str, window_start: float) -> float:
        """Return the sum of amounts for events in the window."""
        members = await self._redis.zrangebyscore(key, window_start, "+inf")
        total = 0.0
        for m in members:
            data = self._decode_member(m)
            total += float(data.get("amt", 0))
        return total

    async def _unique_merchant_velocity(self, key: str, window_start: float) -> float:
        """Return the count of distinct merchants in the window."""
        members = await self._redis.zrangebyscore(key, window_start, "+inf")
        merchants: set[str] = set()
        for m in members:
            data = self._decode_member(m)
            mer = data.get("mer", "")
            if mer:
                merchants.add(mer)
        return float(len(merchants))

    async def _get_velocity_value(
        self,
        key: str,
        velocity_type: VelocityType,
        window_start: float,
    ) -> float:
        """Dispatch to the correct velocity query method."""
        if velocity_type == VelocityType.COUNT:
            return await self._count_velocity(key, window_start)
        if velocity_type == VelocityType.AMOUNT:
            return await self._amount_velocity(key, window_start)
        if velocity_type == VelocityType.UNIQUE_MERCHANT:
            return await self._unique_merchant_velocity(key, window_start)
        return 0.0  # pragma: no cover

    # ------------------------------------------------------------------
    # Evaluate thresholds
    # ------------------------------------------------------------------

    async def check(
        self,
        entities: dict[Dimension, str],
        now: float | None = None,
    ) -> VelocityReport:
        """Evaluate all configured thresholds for the given entities.

        Args:
            entities: Mapping of dimension to entity ID.
            now: Current Unix timestamp; defaults to ``time.time()``.

        Returns:
            A ``VelocityReport`` containing per-threshold results and an
            aggregated velocity score.
        """
        ts = now or time.time()
        # Use the first entity id for the report label.
        primary_id = next(iter(entities.values()), "unknown")
        report = VelocityReport(entity_id=primary_id)

        start = time.monotonic()
        try:
            # Pre-compute the largest window per key so pruning doesn't
            # remove entries still needed by a wider window.
            max_window_per_key: dict[str, float] = {}
            for threshold in self._thresholds:
                entity_id = entities.get(threshold.dimension)
                if entity_id is None:
                    continue
                key = self._key(threshold.dimension, entity_id, threshold.velocity_type)
                prev = max_window_per_key.get(key, 0.0)
                max_window_per_key[key] = max(prev, float(threshold.window.value))

            pruned_keys: set[str] = set()
            for threshold in self._thresholds:
                entity_id = entities.get(threshold.dimension)
                if entity_id is None:
                    continue

                key = self._key(threshold.dimension, entity_id, threshold.velocity_type)
                window_start = ts - threshold.window.value

                # Prune old entries once per key using the widest window.
                if key not in pruned_keys:
                    max_cutoff = ts - max_window_per_key[key]
                    await self._prune(key, max_cutoff - 1)
                    pruned_keys.add(key)

                value = await self._get_velocity_value(key, threshold.velocity_type, window_start)
                breached = value >= threshold.limit

                VELOCITY_CHECKS.labels(
                    dimension=threshold.dimension.value,
                    window=threshold.window.label,
                    velocity_type=threshold.velocity_type.value,
                ).inc()

                if breached:
                    VELOCITY_BREACHES.labels(
                        dimension=threshold.dimension.value,
                        window=threshold.window.label,
                        velocity_type=threshold.velocity_type.value,
                    ).inc()

                report.results.append(
                    VelocityResult(
                        dimension=threshold.dimension,
                        velocity_type=threshold.velocity_type,
                        window=threshold.window,
                        current_value=value,
                        limit=threshold.limit,
                        breached=breached,
                    )
                )
        except Exception as exc:
            await logger.awarning("velocity_check_error", error=str(exc))
            raise
        finally:
            VELOCITY_LATENCY.labels(operation="check").observe(time.monotonic() - start)

        report.any_breach = any(r.breached for r in report.results)
        report.score = self._compute_score(report.results)
        return report

    @staticmethod
    def _compute_score(results: list[VelocityResult]) -> float:
        """Compute a normalized velocity risk score in [0, 1].

        The score is the mean of per-threshold saturation ratios
        (current_value / limit), capped at 1.0 each.
        """
        if not results:
            return 0.0
        ratios = [min(r.current_value / r.limit, 1.0) if r.limit > 0 else 0.0 for r in results]
        return min(sum(ratios) / len(ratios), 1.0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def reset(self, dimension: Dimension, entity_id: str) -> None:
        """Delete all velocity data for an entity (e.g. after account reset)."""
        pipe = self._redis.pipeline()
        for vtype in VelocityType:
            key = self._key(dimension, entity_id, vtype)
            pipe.delete(key)
        await pipe.execute()

    def stats(self) -> dict[str, Any]:
        """Return a summary of the engine's current configuration."""
        return {
            "thresholds_configured": len(self._thresholds),
            "key_prefix": self._key_prefix,
            "dimensions": [d.value for d in Dimension],
            "windows": {w.label: w.value for w in TimeWindow},
        }
