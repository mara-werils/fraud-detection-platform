"""Alert deduplication with TTL-based expiry."""

from __future__ import annotations

import asyncio
import time
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)

# Metrics
_dedup_hit_count: int = 0
_dedup_miss_count: int = 0


def get_dedup_metrics() -> dict[str, int]:
    """Return current dedup metrics."""
    return {
        "dedup_hit_count": _dedup_hit_count,
        "dedup_miss_count": _dedup_miss_count,
    }


class AlertDeduplicator:
    """In-memory alert deduplication with TTL.

    Prevents duplicate alerts for the same user_id + alert_type
    within a configurable time window (default 10 minutes).
    """

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._ttl_seconds = ttl_seconds
        self._seen: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        await logger.ainfo(
            "dedup_started",
            ttl_seconds=self._ttl_seconds,
        )

    async def stop(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        await logger.ainfo("dedup_stopped")

    async def is_duplicate(self, user_id: UUID, alert_type: str) -> bool:
        """Check if an alert is a duplicate within the TTL window.

        Args:
            user_id: The user ID to check.
            alert_type: The alert type (e.g., "BLOCK", "REVIEW").

        Returns:
            True if the alert is a duplicate and should be suppressed.
        """
        global _dedup_hit_count, _dedup_miss_count

        key = f"{user_id}:{alert_type}"
        now = time.monotonic()

        async with self._lock:
            last_seen = self._seen.get(key)
            if last_seen is not None and (now - last_seen) < self._ttl_seconds:
                _dedup_hit_count += 1
                await logger.ainfo(
                    "alert_deduplicated",
                    user_id=str(user_id),
                    alert_type=alert_type,
                    seconds_since_last=round(now - last_seen, 1),
                )
                return True

            self._seen[key] = now
            _dedup_miss_count += 1
            return False

    async def _periodic_cleanup(self) -> None:
        """Periodically remove expired entries to prevent memory leaks."""
        while True:
            try:
                await asyncio.sleep(self._ttl_seconds)
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except Exception:
                await logger.aexception("dedup_cleanup_error")

    async def _cleanup(self) -> None:
        """Remove expired entries from the dedup map."""
        now = time.monotonic()
        async with self._lock:
            expired_keys = [
                k for k, ts in self._seen.items()
                if (now - ts) >= self._ttl_seconds
            ]
            for key in expired_keys:
                del self._seen[key]
            if expired_keys:
                await logger.ainfo(
                    "dedup_cleanup",
                    expired_count=len(expired_keys),
                    remaining_count=len(self._seen),
                )
