"""Event sourcing store for a full, immutable audit trail.

Every state change in the fraud detection platform is represented as an
append-only Event.  Downstream projections rebuild read models from the
ordered event stream, which supports:

  - Full audit trail with actor attribution
  - Point-in-time replay for debugging and compliance
  - Optimistic-concurrency conflict detection
  - Fan-out to multiple real-time subscribers
  - Compaction / archival of aged events
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, AsyncIterator, Callable, Awaitable

import structlog
from prometheus_client import Counter, Gauge
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────

_events_total = Counter(
    "event_store_events_total",
    "Total events appended to the store",
)
_events_by_type = Counter(
    "event_store_events_by_type_total",
    "Events appended, broken down by event type",
    ["event_type"],
)
_projection_lag = Gauge(
    "event_store_projection_lag_events",
    "Number of events not yet processed by a projection",
    ["projection_name"],
)
_subscriber_fanout_total = Counter(
    "event_store_subscriber_fanout_total",
    "Total events dispatched to subscribers",
    ["event_type"],
)
_compact_archived_total = Counter(
    "event_store_compact_archived_total",
    "Total events archived during compaction runs",
)


# ── Enumerations ───────────────────────────────────────────────────────────────

class EventType(StrEnum):
    TRANSACTION_SCORED = "transaction_scored"
    CASE_CREATED = "case_created"
    CASE_UPDATED = "case_updated"
    CASE_RESOLVED = "case_resolved"
    FEEDBACK_SUBMITTED = "feedback_submitted"
    ALERT_TRIGGERED = "alert_triggered"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"
    MODEL_DEPLOYED = "model_deployed"
    RULE_UPDATED = "rule_updated"
    ENTITY_BLOCKED = "entity_blocked"
    ENTITY_UNBLOCKED = "entity_unblocked"
    USER_ACTION = "user_action"
    CONFIG_CHANGED = "config_changed"
    WEBHOOK_DELIVERED = "webhook_delivered"
    DATA_EXPORTED = "data_exported"
    DATA_ERASED = "data_erased"


class ActorType(StrEnum):
    USER = "user"
    SYSTEM = "system"
    API = "api"


# ── Pydantic models ────────────────────────────────────────────────────────────

class Event(BaseModel):
    """An immutable fact recorded in the event store.

    ``version`` is the sequence number within the aggregate's event stream.
    Callers set ``version = <expected_current_version> + 1``; the store
    rejects appends where the version has already been taken (optimistic
    concurrency control).
    """

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_type: EventType
    aggregate_id: str
    aggregate_type: str

    # Payload and context
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Attribution
    actor_id: str | None = None
    actor_type: ActorType = ActorType.SYSTEM

    # Ordering / concurrency
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    version: int = 1  # monotonically increasing per aggregate

    model_config = {"frozen": True}


class EventQuery(BaseModel):
    """Filter criteria for querying the event store."""

    aggregate_id: str | None = None
    aggregate_type: str | None = None
    event_types: list[EventType] | None = None
    actor_id: str | None = None
    from_timestamp: datetime | None = None
    to_timestamp: datetime | None = None
    limit: int = Field(default=100, ge=1, le=10_000)
    offset: int = Field(default=0, ge=0)


class EventStoreStats(BaseModel):
    """Runtime statistics for the event store."""

    total_events: int
    archived_events: int
    unique_aggregates: int
    events_by_type: dict[str, int]
    active_subscriptions: int
    oldest_event_ts: datetime | None
    newest_event_ts: datetime | None


# ── Subscriber type alias ──────────────────────────────────────────────────────

SubscriberCallback = Callable[[Event], Awaitable[None]]


class _Subscription:
    __slots__ = ("subscription_id", "event_types", "callback")

    def __init__(
        self,
        subscription_id: str,
        event_types: list[EventType],
        callback: SubscriberCallback,
    ) -> None:
        self.subscription_id = subscription_id
        self.event_types = frozenset(event_types)
        self.callback = callback


# ── Core event store ───────────────────────────────────────────────────────────

class EventStore:
    """Append-only, in-memory event store with replay and fan-out.

    Production deployments should swap the in-memory lists for a durable
    backend (e.g. PostgreSQL ``event_store`` table with a SEQUENCE, or
    EventStoreDB).  The public interface is backend-agnostic so a subclass
    can override ``_persist`` and ``_load`` without changing call sites.

    Thread / concurrency safety:
        All mutations go through ``_lock`` (asyncio.Lock).  Read paths
        iterate over snapshots taken under the lock to avoid holding it
        during I/O-heavy work.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Primary storage: ordered list of all live events
        self._events: list[Event] = []
        # Archive: events moved out by compaction
        self._archive: list[Event] = []
        # Per-aggregate version tracker for optimistic concurrency
        self._aggregate_versions: dict[str, int] = defaultdict(int)
        # Subscriptions keyed by subscription_id
        self._subscriptions: dict[str, _Subscription] = {}
        # Snapshot cache: aggregate_id -> reduced state dict
        self._snapshots: dict[str, dict[str, Any]] = {}

    # ── Write path ─────────────────────────────────────────────────────────────

    async def append(self, event: Event) -> Event:
        """Persist a single event.

        Raises:
            ValueError: If the event's version conflicts with the current
                aggregate version (optimistic concurrency violation).
        """
        async with self._lock:
            stored = await self._check_and_store(event)

        await self._fanout(stored)
        return stored

    async def append_batch(self, events: list[Event]) -> list[Event]:
        """Persist multiple events atomically (all-or-nothing).

        All events in the batch must target the same aggregate so that
        version sequencing can be validated contiguously.

        Raises:
            ValueError: On version conflict or empty batch.
        """
        if not events:
            raise ValueError("append_batch requires at least one event")

        stored_events: list[Event] = []
        async with self._lock:
            for event in events:
                stored = await self._check_and_store(event)
                stored_events.append(stored)

        for stored in stored_events:
            await self._fanout(stored)

        return stored_events

    async def _check_and_store(self, event: Event) -> Event:
        """Validate version and append; must be called under ``_lock``."""
        current_version = self._aggregate_versions[event.aggregate_id]
        expected_version = current_version + 1

        if event.version != expected_version:
            raise ValueError(
                f"Optimistic concurrency conflict for aggregate "
                f"'{event.aggregate_id}': expected version {expected_version}, "
                f"got {event.version}."
            )

        self._aggregate_versions[event.aggregate_id] = event.version
        self._events.append(event)

        # Metrics
        _events_total.inc()
        _events_by_type.labels(event_type=event.event_type.value).inc()

        await logger.ainfo(
            "event_appended",
            event_id=str(event.event_id),
            event_type=event.event_type,
            aggregate_id=event.aggregate_id,
            aggregate_type=event.aggregate_type,
            version=event.version,
            actor_id=event.actor_id,
        )

        return event

    # ── Read path ──────────────────────────────────────────────────────────────

    async def get_events(self, query: EventQuery) -> list[Event]:
        """Return events matching the given query, newest-last (insertion order)."""
        async with self._lock:
            snapshot = list(self._events)

        results: list[Event] = []
        for ev in snapshot:
            if query.aggregate_id and ev.aggregate_id != query.aggregate_id:
                continue
            if query.aggregate_type and ev.aggregate_type != query.aggregate_type:
                continue
            if query.event_types and ev.event_type not in query.event_types:
                continue
            if query.actor_id and ev.actor_id != query.actor_id:
                continue
            if query.from_timestamp and ev.timestamp < query.from_timestamp:
                continue
            if query.to_timestamp and ev.timestamp > query.to_timestamp:
                continue
            results.append(ev)

        return results[query.offset : query.offset + query.limit]

    async def get_aggregate_events(
        self,
        aggregate_id: str,
        aggregate_type: str | None = None,
    ) -> list[Event]:
        """Return all events for a specific aggregate, in version order."""
        query = EventQuery(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            limit=10_000,
        )
        events = await self.get_events(query)
        return sorted(events, key=lambda e: e.version)

    async def get_latest_event(self, aggregate_id: str) -> Event | None:
        """Return the most recent event for an aggregate, or None."""
        events = await self.get_aggregate_events(aggregate_id)
        return events[-1] if events else None

    # ── Replay ────────────────────────────────────────────────────────────────

    async def replay(
        self,
        from_event_id: uuid.UUID | None = None,
    ) -> AsyncIterator[Event]:
        """Yield all events in insertion order, optionally starting after ``from_event_id``.

        Useful for rebuilding projections after a restart.
        """
        async with self._lock:
            snapshot = list(self._events)

        skip = from_event_id is not None
        for ev in snapshot:
            if skip:
                if ev.event_id == from_event_id:
                    skip = False
                continue
            yield ev
            # Yield control to the event loop between chunks
            await asyncio.sleep(0)

    # ── Subscriptions ─────────────────────────────────────────────────────────

    def subscribe(
        self,
        event_types: list[EventType],
        callback: SubscriberCallback,
    ) -> str:
        """Register a callback for one or more event types.

        Args:
            event_types: Event types to receive.  Pass all values to receive
                every event.
            callback: An async callable that accepts a single ``Event``.

        Returns:
            A subscription ID that can be passed to ``unsubscribe``.
        """
        sub_id = str(uuid.uuid4())
        self._subscriptions[sub_id] = _Subscription(sub_id, event_types, callback)
        logger.info(
            "event_store_subscribed",
            subscription_id=sub_id,
            event_types=[et.value for et in event_types],
        )
        return sub_id

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription by ID.  No-op if the ID is unknown."""
        removed = self._subscriptions.pop(subscription_id, None)
        if removed:
            logger.info("event_store_unsubscribed", subscription_id=subscription_id)

    async def _fanout(self, event: Event) -> None:
        """Dispatch an event to all matching subscribers (fire-and-forget)."""
        for sub in list(self._subscriptions.values()):
            if event.event_type not in sub.event_types:
                continue
            _subscriber_fanout_total.labels(event_type=event.event_type.value).inc()
            asyncio.create_task(
                self._safe_dispatch(sub, event),
                name=f"event-dispatch-{sub.subscription_id[:8]}",
            )

    @staticmethod
    async def _safe_dispatch(sub: _Subscription, event: Event) -> None:
        try:
            await sub.callback(event)
        except Exception as exc:
            await logger.awarning(
                "event_subscriber_error",
                subscription_id=sub.subscription_id,
                event_id=str(event.event_id),
                error=str(exc),
            )

    # ── Snapshots ────────────────────────────────────────────────────────────

    async def get_snapshot(self, aggregate_id: str) -> dict[str, Any]:
        """Return a cached reduced-state dict for an aggregate.

        The snapshot is rebuilt on demand from all events when missing.
        Call this instead of re-replaying the full history for hot paths.
        """
        if aggregate_id in self._snapshots:
            return dict(self._snapshots[aggregate_id])

        events = await self.get_aggregate_events(aggregate_id)
        state: dict[str, Any] = {
            "aggregate_id": aggregate_id,
            "version": 0,
            "event_count": 0,
            "last_event_type": None,
            "last_updated": None,
            "history": [],
        }
        for ev in events:
            state["version"] = ev.version
            state["event_count"] += 1
            state["last_event_type"] = ev.event_type.value
            state["last_updated"] = ev.timestamp.isoformat()
            state["history"].append(
                {
                    "event_type": ev.event_type.value,
                    "timestamp": ev.timestamp.isoformat(),
                    "actor_id": ev.actor_id,
                    "version": ev.version,
                }
            )

        async with self._lock:
            self._snapshots[aggregate_id] = dict(state)

        return state

    def _invalidate_snapshot(self, aggregate_id: str) -> None:
        self._snapshots.pop(aggregate_id, None)

    # ── Compaction ────────────────────────────────────────────────────────────

    async def compact(self, before_timestamp: datetime) -> int:
        """Archive events older than ``before_timestamp``.

        Archived events are moved to a separate list (in a real system this
        would be a cold-storage table or object-storage blob).  Returns the
        number of events archived.
        """
        async with self._lock:
            live: list[Event] = []
            archived: list[Event] = []
            for ev in self._events:
                if ev.timestamp < before_timestamp:
                    archived.append(ev)
                else:
                    live.append(ev)

            self._events = live
            self._archive.extend(archived)
            # Snapshot cache is now stale
            self._snapshots.clear()

        count = len(archived)
        _compact_archived_total.inc(count)
        await logger.ainfo(
            "event_store_compacted",
            archived_count=count,
            before_timestamp=before_timestamp.isoformat(),
        )
        return count

    # ── Stats ────────────────────────────────────────────────────────────────

    async def get_stats(self) -> EventStoreStats:
        """Return current event store statistics."""
        async with self._lock:
            snapshot = list(self._events)
            archived_count = len(self._archive)
            active_subs = len(self._subscriptions)

        by_type: dict[str, int] = defaultdict(int)
        aggregates: set[str] = set()
        oldest: datetime | None = None
        newest: datetime | None = None

        for ev in snapshot:
            by_type[ev.event_type.value] += 1
            aggregates.add(ev.aggregate_id)
            if oldest is None or ev.timestamp < oldest:
                oldest = ev.timestamp
            if newest is None or ev.timestamp > newest:
                newest = ev.timestamp

        return EventStoreStats(
            total_events=len(snapshot),
            archived_events=archived_count,
            unique_aggregates=len(aggregates),
            events_by_type=dict(by_type),
            active_subscriptions=active_subs,
            oldest_event_ts=oldest,
            newest_event_ts=newest,
        )


# ── Projection base ───────────────────────────────────────────────────────────

class EventProjection:
    """Build a read model by consuming a stream of events.

    Subclasses override ``project`` to handle specific event types and
    accumulate state into ``_state``.  Call ``rebuild`` to re-process the
    full event history from the store.

    Example::

        class CaseProjection(EventProjection):
            name = "cases"

            def project(self, event: Event) -> None:
                if event.event_type == EventType.CASE_CREATED:
                    self._state[event.aggregate_id] = event.data
                elif event.event_type == EventType.CASE_RESOLVED:
                    case = self._state.get(event.aggregate_id, {})
                    case["resolved_at"] = event.timestamp.isoformat()
    """

    #: Unique human-readable name used in metric labels
    name: str = "base"

    def __init__(self, store: EventStore) -> None:
        self._store = store
        self._state: dict[str, Any] = {}
        self._last_event_id: uuid.UUID | None = None
        self._processed_count: int = 0
        self._lock = asyncio.Lock()

    # ── To be overridden ──────────────────────────────────────────────────────

    def project(self, event: Event) -> None:
        """Apply a single event to the projection's state.

        Override this in subclasses.  The base implementation is a no-op.
        """

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def rebuild(self) -> None:
        """Replay the full event history and rebuild state from scratch."""
        async with self._lock:
            self._state.clear()
            self._processed_count = 0
            self._last_event_id = None

        await logger.ainfo("projection_rebuild_started", projection=self.name)

        async for event in self._store.replay():
            async with self._lock:
                try:
                    self.project(event)
                    self._last_event_id = event.event_id
                    self._processed_count += 1
                except Exception as exc:
                    await logger.awarning(
                        "projection_event_error",
                        projection=self.name,
                        event_id=str(event.event_id),
                        error=str(exc),
                    )

        _projection_lag.labels(projection_name=self.name).set(0)
        await logger.ainfo(
            "projection_rebuild_complete",
            projection=self.name,
            events_processed=self._processed_count,
        )

    async def apply(self, event: Event) -> None:
        """Apply a single new event incrementally (called by a subscriber)."""
        async with self._lock:
            try:
                self.project(event)
                self._last_event_id = event.event_id
                self._processed_count += 1
            except Exception as exc:
                await logger.awarning(
                    "projection_apply_error",
                    projection=self.name,
                    event_id=str(event.event_id),
                    error=str(exc),
                )

    def get_state(self) -> dict[str, Any]:
        """Return a shallow copy of the current projection state."""
        return dict(self._state)

    def register(self, event_types: list[EventType] | None = None) -> str:
        """Subscribe this projection to the event store.

        Args:
            event_types: Specific types to handle, or ``None`` for all.

        Returns:
            The subscription ID so the caller can unsubscribe later.
        """
        types = event_types or list(EventType)
        return self._store.subscribe(types, self.apply)

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def last_event_id(self) -> uuid.UUID | None:
        return self._last_event_id
