"""Async retry decorator with exponential backoff and jitter.

Usage::

    @retry(max_attempts=3, base_delay=0.5, retryable=(ConnectionError, TimeoutError))
    async def call_external_api():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

__all__ = ["retry"]

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable: tuple[type[Exception], ...] = (Exception,),
    jitter: bool = True,
) -> Callable:
    """Decorator that retries an async function on failure.

    Args:
        max_attempts: Maximum number of attempts (including the first).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Cap on the delay between retries.
        backoff_factor: Multiplier applied to the delay after each retry.
        retryable: Tuple of exception types that should trigger a retry.
        jitter: Add random jitter to delay to prevent thundering herd.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "retry_exhausted",
                            func=func.__qualname__,
                            attempts=max_attempts,
                            error=str(exc),
                        )
                        raise
                    actual_delay = min(delay, max_delay)
                    if jitter:
                        actual_delay *= 0.5 + random.random()
                    logger.warning(
                        "retry_attempt",
                        func=func.__qualname__,
                        attempt=attempt,
                        next_delay_seconds=round(actual_delay, 3),
                        error=str(exc),
                    )
                    await asyncio.sleep(actual_delay)
                    delay *= backoff_factor
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
