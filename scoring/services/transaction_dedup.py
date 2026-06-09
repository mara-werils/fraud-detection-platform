"""Transaction deduplication service with content hashing.

Prevents the same transaction from being scored or processed more than once
by computing a deterministic content hash over key fields (amount, merchant,
timestamp, card hash) and checking against a short-lived cache.

The cache can be backed by either an in-memory TTL dict (for tests and
single-process deployments) or a Redis instance (for distributed setups).

Key design choices:
  - SHA-256 content hash for collision resistance.
  - Configurable TTL window (default 5 minutes).
  - Returns the original transaction ID on duplicate detection so callers
    can link the duplicate back to the first occurrence.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

__all__ = [
    "DedupResult",
    "CacheBackend",
    "InMemoryTTLCache",
    "RedisCacheBackend",
    "TransactionDedup",
    "DEFAULT_TTL_SECONDS",
]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DedupResult:
    """Result of a deduplication check."""

    is_duplicate: bool
    content_hash: str
    original_txn_id: str | None = None


class CacheBackend(Protocol):
    """Minimal interface for a TTL-aware cache backend."""

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_seconds: int) -> None: ...


# ---------------------------------------------------------------------------
# In-memory backend (tests / single-node)
# ---------------------------------------------------------------------------

class InMemoryTTLCache:
    """Simple in-process TTL cache using a dict and lazy expiry."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = (value, time.monotonic() + ttl_seconds)

    def purge_expired(self) -> int:
        """Remove all expired entries.  Returns count of purged items."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
        return len(expired)


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

class RedisCacheBackend:
    """Redis-backed dedup cache using SETEX for automatic TTL expiry."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    def get(self, key: str) -> str | None:
        value = self._redis.get(key)
        if value is None:
            return None
        return value.decode() if isinstance(value, bytes) else value

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._redis.setex(key, ttl_seconds, value)


# ---------------------------------------------------------------------------
# Dedup service
# ---------------------------------------------------------------------------

DEFAULT_TTL_SECONDS = 300  # 5 minutes


class TransactionDedup:
    """Detects duplicate transactions via content hashing.

    Parameters
    ----------
    cache:
        Backend implementing ``CacheBackend`` protocol.
    ttl_seconds:
        How long a content hash is retained (dedup window).
    key_prefix:
        Redis/cache key namespace prefix.
    """

    def __init__(
        self,
        cache: CacheBackend | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        key_prefix: str = "txdedup",
    ) -> None:
        self._cache: CacheBackend = cache or InMemoryTTLCache()
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        txn_id: str,
        amount: float,
        merchant: str,
        timestamp: str,
        card_hash: str,
    ) -> DedupResult:
        """Check whether a transaction is a duplicate.

        If the content hash already exists in the cache the transaction is
        considered a duplicate and the original ``txn_id`` is returned.
        Otherwise the hash is stored and the transaction is treated as new.
        """
        content_hash = self._compute_hash(amount, merchant, timestamp, card_hash)
        cache_key = f"{self._prefix}:{content_hash}"

        existing_id = self._cache.get(cache_key)
        if existing_id is not None:
            logger.info(
                "duplicate_transaction_detected",
                txn_id=txn_id,
                original_txn_id=existing_id,
                content_hash=content_hash,
            )
            return DedupResult(
                is_duplicate=True,
                content_hash=content_hash,
                original_txn_id=existing_id,
            )

        self._cache.set(cache_key, txn_id, self._ttl)
        return DedupResult(is_duplicate=False, content_hash=content_hash)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(
        amount: float, merchant: str, timestamp: str, card_hash: str
    ) -> str:
        """Deterministic SHA-256 hash of transaction content fields."""
        raw = f"{amount:.2f}|{merchant.strip().lower()}|{timestamp}|{card_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()
