"""Blocklist/allowlist/watchlist management service.

Manages entity-level access control lists for fraud detection, supporting
blocklist, allowlist, and watchlist entries across multiple entity types
(users, devices, IPs, merchants, card hashes, emails).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from enum import StrEnum
from threading import Lock
from typing import Any
from uuid import uuid4

import structlog
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

LIST_ENTRIES_TOTAL = Gauge(
    "entity_list_entries_total",
    "Total entries in entity lists",
    ["list_type", "entity_type"],
)

LIST_CHECK_DURATION = Histogram(
    "entity_list_check_duration_seconds",
    "Time to check an entity against all lists",
)

LIST_MATCH_COUNTER = Counter(
    "entity_list_matches_total",
    "Total matches found during entity checks",
    ["list_type", "entity_type"],
)

LIST_OPERATIONS = Counter(
    "entity_list_operations_total",
    "Total add/remove operations on entity lists",
    ["operation", "list_type"],
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EntityType(StrEnum):
    USER = "user"
    DEVICE = "device"
    IP = "ip"
    MERCHANT = "merchant"
    CARD_HASH = "card_hash"
    EMAIL = "email"


class ListType(StrEnum):
    BLOCKLIST = "blocklist"
    ALLOWLIST = "allowlist"
    WATCHLIST = "watchlist"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EntityEntry(BaseModel):
    """A single entry in an entity list."""

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    entity_type: EntityType
    entity_value: str = Field(..., min_length=1, description="Canonical entity value")
    list_type: ListType
    reason: str = Field(..., min_length=1, description="Human-readable reason for listing")
    added_by: str = Field(..., description="User or system that added the entry")
    added_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(None, description="Optional expiry; None = permanent")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_expired(self) -> bool:
        """Return True if this entry has passed its expiry timestamp."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_active(self) -> bool:
        """Return True when the entry is not expired."""
        return not self.is_expired()


class ListMatchResult(BaseModel):
    """Result of checking a single entity against the lists."""

    matched: bool
    list_type: ListType | None = None
    entity_type: EntityType | None = None
    entity_value: str | None = None
    reason: str | None = None
    entry: EntityEntry | None = None

    @classmethod
    def no_match(cls) -> ListMatchResult:
        return cls(matched=False)


class ImportResult(BaseModel):
    """Summary of a bulk-import operation."""

    imported: int
    skipped: int
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class EntityListService:
    """In-memory entity list service with thread-safe operations.

    In production this should be backed by a Redis hash or a database table;
    the interface is kept stable so the storage layer can be swapped without
    touching the rest of the scoring pipeline.
    """

    def __init__(self) -> None:
        # Primary index: (list_type, entity_type, entity_value) -> EntityEntry
        self._entries: dict[tuple[str, str, str], EntityEntry] = {}
        # Reverse index for fast lookup: (entity_type, entity_value) -> list of entries
        self._by_entity: dict[tuple[str, str], list[EntityEntry]] = defaultdict(list)
        self._lock = Lock()
        logger.info("entity_list_service.initialized")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_entry(self, entry: EntityEntry) -> EntityEntry:
        """Add or replace an entry.  Returns the stored entry."""
        key = (entry.list_type, entry.entity_type, entry.entity_value)
        entity_key = (entry.entity_type, entry.entity_value)

        with self._lock:
            # Remove stale reference from reverse index if replacing
            if key in self._entries:
                old = self._entries[key]
                self._by_entity[entity_key] = [
                    e for e in self._by_entity[entity_key] if e.entry_id != old.entry_id
                ]

            self._entries[key] = entry
            self._by_entity[entity_key].append(entry)

        LIST_OPERATIONS.labels(operation="add", list_type=entry.list_type).inc()
        LIST_ENTRIES_TOTAL.labels(
            list_type=entry.list_type, entity_type=entry.entity_type
        ).inc()
        logger.info(
            "entity_list.entry_added",
            list_type=entry.list_type,
            entity_type=entry.entity_type,
            entity_value=entry.entity_value,
            added_by=entry.added_by,
        )
        return entry

    def remove_entry(
        self,
        entity_type: EntityType | str,
        entity_value: str,
        list_type: ListType | str,
    ) -> bool:
        """Remove an entry.  Returns True if something was removed."""
        key = (str(list_type), str(entity_type), entity_value)
        entity_key = (str(entity_type), entity_value)

        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return False
            self._by_entity[entity_key] = [
                e for e in self._by_entity[entity_key] if e.entry_id != entry.entry_id
            ]
            if not self._by_entity[entity_key]:
                del self._by_entity[entity_key]

        LIST_OPERATIONS.labels(operation="remove", list_type=list_type).inc()
        LIST_ENTRIES_TOTAL.labels(list_type=list_type, entity_type=entity_type).dec()
        logger.info(
            "entity_list.entry_removed",
            list_type=list_type,
            entity_type=entity_type,
            entity_value=entity_value,
        )
        return True

    # ------------------------------------------------------------------
    # Read / check operations
    # ------------------------------------------------------------------

    def check(
        self,
        entity_type: EntityType | str,
        entity_value: str,
    ) -> ListMatchResult:
        """Check a single entity across all lists (blocklist first)."""
        entity_key = (str(entity_type), entity_value)

        with LIST_CHECK_DURATION.time():
            with self._lock:
                candidates = list(self._by_entity.get(entity_key, []))

        # Prioritise: BLOCKLIST > ALLOWLIST > WATCHLIST
        priority = {ListType.BLOCKLIST: 0, ListType.ALLOWLIST: 1, ListType.WATCHLIST: 2}
        active = [e for e in candidates if e.is_active()]
        active.sort(key=lambda e: priority.get(e.list_type, 99))

        if not active:
            return ListMatchResult.no_match()

        match = active[0]
        LIST_MATCH_COUNTER.labels(
            list_type=match.list_type, entity_type=match.entity_type
        ).inc()
        return ListMatchResult(
            matched=True,
            list_type=match.list_type,
            entity_type=match.entity_type,
            entity_value=match.entity_value,
            reason=match.reason,
            entry=match,
        )

    def check_transaction(self, transaction: Any) -> list[ListMatchResult]:
        """Check all relevant entity fields of a transaction object.

        The transaction is expected to have attributes (or dict keys) for:
        user_id, device_id, ip_address, merchant_id, card_hash, email.
        Missing fields are silently skipped.
        """
        fields: list[tuple[EntityType, str]] = [
            (EntityType.USER, "user_id"),
            (EntityType.DEVICE, "device_id"),
            (EntityType.IP, "ip_address"),
            (EntityType.MERCHANT, "merchant_id"),
            (EntityType.CARD_HASH, "card_hash"),
            (EntityType.EMAIL, "email"),
        ]

        results: list[ListMatchResult] = []
        for entity_type, attr in fields:
            value = (
                transaction.get(attr)
                if isinstance(transaction, dict)
                else getattr(transaction, attr, None)
            )
            if not value:
                continue
            result = self.check(entity_type, str(value))
            if result.matched:
                results.append(result)

        return results

    # ------------------------------------------------------------------
    # Bulk / management operations
    # ------------------------------------------------------------------

    def get_entries(
        self,
        list_type: ListType | str,
        entity_type: EntityType | str | None = None,
        limit: int = 100,
    ) -> list[EntityEntry]:
        """Return active entries optionally filtered by entity type."""
        with self._lock:
            candidates = list(self._entries.values())

        results = [
            e
            for e in candidates
            if e.list_type == str(list_type)
            and (entity_type is None or e.entity_type == str(entity_type))
            and e.is_active()
        ]
        results.sort(key=lambda e: e.added_at, reverse=True)
        return results[:limit]

    def import_entries(self, entries: list[EntityEntry]) -> ImportResult:
        """Bulk-import a list of entries.  Duplicate keys are overwritten."""
        imported = 0
        skipped = 0
        errors: list[str] = []

        for entry in entries:
            try:
                self.add_entry(entry)
                imported += 1
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                errors.append(f"{entry.entity_type}:{entry.entity_value} — {exc}")

        logger.info("entity_list.import_complete", imported=imported, skipped=skipped)
        return ImportResult(imported=imported, skipped=skipped, errors=errors)

    def export_entries(self, list_type: ListType | str) -> list[EntityEntry]:
        """Export all active entries for a given list type."""
        with self._lock:
            snapshot = list(self._entries.values())
        return [e for e in snapshot if e.list_type == str(list_type) and e.is_active()]

    def cleanup_expired(self) -> int:
        """Remove all expired entries and return the count removed."""
        removed = 0
        with self._lock:
            expired_keys = [k for k, e in self._entries.items() if e.is_expired()]
            for key in expired_keys:
                entry = self._entries.pop(key)
                entity_key = (entry.entity_type, entry.entity_value)
                self._by_entity[entity_key] = [
                    e
                    for e in self._by_entity[entity_key]
                    if e.entry_id != entry.entry_id
                ]
                if not self._by_entity[entity_key]:
                    del self._by_entity[entity_key]
                removed += 1

        if removed:
            logger.info("entity_list.cleanup", expired_removed=removed)
        return removed

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics broken down by list type and entity type."""
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        expired = 0

        with self._lock:
            entries = list(self._entries.values())

        for entry in entries:
            if entry.is_expired():
                expired += 1
                continue
            counts[entry.list_type][entry.entity_type] += 1

        return {
            "total_active": sum(
                v for lt in counts.values() for v in lt.values()
            ),
            "total_expired_pending_cleanup": expired,
            "by_list_type": {lt: dict(et) for lt, et in counts.items()},
        }
