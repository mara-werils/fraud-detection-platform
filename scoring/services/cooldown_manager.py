"""Entity cooldown manager for post-detection action throttling.

After a fraud flag is raised on an entity (card, account, device, IP),
downstream actions such as blocking or step-up authentication should be
enforced for a configurable cooldown period.  This module manages those
cooldown windows using Redis TTL keys.

Cooldown semantics:
  - ``start_cooldown(entity_type, entity_id)`` sets a Redis key with a
    TTL equal to the configured duration.  If a cooldown already exists
    it is *extended* to whichever expiry is later.
  - ``is_cooling_down(entity_type, entity_id)`` checks key existence.
  - ``remaining(entity_type, entity_id)`` returns seconds left, or 0.

Falls back to an in-memory store when Redis is unavailable (e.g. tests).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import structlog
from prometheus_client import Counter, Gauge

logger = structlog.get_logger(__name__)

COOLDOWNS_STARTED = Counter(
    "cooldown_started_total",
    "Number of cooldown periods initiated",
    ["entity_type"],
)
COOLDOWNS_ACTIVE = Gauge(
    "cooldowns_active",
    "Number of currently active cooldowns",
    ["entity_type"],
)


class EntityType(StrEnum):
    CARD = "card"
    ACCOUNT = "account"
    DEVICE = "device"
    IP = "ip"


# ---------------------------------------------------------------------------
# Store protocol + implementations
# ---------------------------------------------------------------------------

class CooldownStore(Protocol):
    """Minimal store interface for cooldown keys."""

    def get_ttl(self, key: str) -> int:
        """Return remaining TTL in seconds, or -2 if key does not exist."""
        ...

    def set_with_ttl(self, key: str, value: str, ttl_seconds: int) -> None: ...

    def exists(self, key: str) -> bool: ...


class RedisCooldownStore:
    """Redis-backed cooldown store."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    def get_ttl(self, key: str) -> int:
        return self._redis.ttl(key)

    def set_with_ttl(self, key: str, value: str, ttl_seconds: int) -> None:
        self._redis.setex(key, ttl_seconds, value)

    def exists(self, key: str) -> bool:
        return bool(self._redis.exists(key))


class InMemoryCooldownStore:
    """In-memory cooldown store for testing."""

    def __init__(self) -> None:
        self._store: dict[str, float] = {}  # key -> expiry monotonic time

    def get_ttl(self, key: str) -> int:
        exp = self._store.get(key)
        if exp is None:
            return -2
        remaining = exp - time.monotonic()
        if remaining <= 0:
            del self._store[key]
            return -2
        return int(remaining)

    def set_with_ttl(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = time.monotonic() + ttl_seconds

    def exists(self, key: str) -> bool:
        return self.get_ttl(key) > 0


# ---------------------------------------------------------------------------
# Default cooldown durations (seconds)
# ---------------------------------------------------------------------------

DEFAULT_DURATIONS: dict[EntityType, int] = {
    EntityType.CARD: 1800,      # 30 minutes
    EntityType.ACCOUNT: 3600,   # 1 hour
    EntityType.DEVICE: 900,     # 15 minutes
    EntityType.IP: 600,         # 10 minutes
}

KEY_PREFIX = "cooldown"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

@dataclass
class CooldownStatus:
    entity_type: EntityType
    entity_id: str
    active: bool
    remaining_seconds: int


class CooldownManager:
    """Manages per-entity cooldown periods after fraud detection events."""

    def __init__(
        self,
        store: CooldownStore | None = None,
        durations: dict[EntityType, int] | None = None,
    ) -> None:
        self._store: CooldownStore = store or InMemoryCooldownStore()
        self._durations = durations or dict(DEFAULT_DURATIONS)

    def _key(self, entity_type: EntityType, entity_id: str) -> str:
        return f"{KEY_PREFIX}:{entity_type.value}:{entity_id}"

    def start_cooldown(
        self,
        entity_type: EntityType,
        entity_id: str,
        duration_override: int | None = None,
    ) -> CooldownStatus:
        """Initiate or extend a cooldown for the given entity."""
        key = self._key(entity_type, entity_id)
        duration = duration_override or self._durations.get(entity_type, 1800)

        current_ttl = self._store.get_ttl(key)
        if current_ttl > 0 and current_ttl >= duration:
            logger.debug("cooldown_already_longer", key=key, current_ttl=current_ttl)
            return CooldownStatus(entity_type, entity_id, True, current_ttl)

        self._store.set_with_ttl(key, "active", duration)
        COOLDOWNS_STARTED.labels(entity_type=entity_type.value).inc()
        logger.info("cooldown_started", entity_type=entity_type.value,
                     entity_id=entity_id, duration=duration)
        return CooldownStatus(entity_type, entity_id, True, duration)

    def is_cooling_down(self, entity_type: EntityType, entity_id: str) -> bool:
        return self._store.exists(self._key(entity_type, entity_id))

    def remaining(self, entity_type: EntityType, entity_id: str) -> int:
        ttl = self._store.get_ttl(self._key(entity_type, entity_id))
        return max(ttl, 0)

    def status(self, entity_type: EntityType, entity_id: str) -> CooldownStatus:
        ttl = self._store.get_ttl(self._key(entity_type, entity_id))
        active = ttl > 0
        return CooldownStatus(entity_type, entity_id, active, max(ttl, 0))
