"""Circuit breaker pattern for resilient service communication.

Implements the circuit breaker pattern with three states:
- CLOSED: Requests flow normally.
- OPEN: Requests fail immediately (circuit tripped after too many failures).
- HALF_OPEN: A limited number of test requests are allowed through.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

import structlog

__all__ = [
    "CircuitState",
    "CircuitBreakerError",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "circuit_registry",
]

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit is open and the call is rejected."""

    def __init__(self, name: str, state: CircuitState) -> None:
        super().__init__(f"Circuit breaker '{name}' is {state.value}")
        self.circuit_name = name
        self.state = state


class CircuitBreaker:
    """Circuit breaker for protecting downstream service calls.

    Args:
        name: Human-readable name for this circuit.
        failure_threshold: Number of failures before opening the circuit.
        recovery_timeout: Seconds to wait before transitioning to half-open.
        half_open_max_calls: Max test calls allowed in half-open state.
        success_threshold: Successes needed in half-open to close the circuit.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: float = 0.0
        self._last_state_change: float = time.monotonic()

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state, with automatic transition from OPEN to HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)
        return self._state

    @property
    def metrics(self) -> dict[str, Any]:
        """Return circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_rejections": self._total_rejections,
        }

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        fallback: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a function through the circuit breaker.

        Args:
            func: The async function to call.
            *args: Arguments for the function.
            fallback: Optional fallback function if the circuit is open.
            **kwargs: Keyword arguments for the function.

        Returns:
            The function's return value.

        Raises:
            CircuitBreakerError: If circuit is open and no fallback provided.
        """
        current_state = self.state
        self._total_calls += 1

        # OPEN: reject or use fallback
        if current_state == CircuitState.OPEN:
            self._total_rejections += 1
            if fallback is not None:
                logger.info("circuit_breaker_fallback", name=self.name)
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                return fallback(*args, **kwargs)
            raise CircuitBreakerError(self.name, current_state)

        # HALF_OPEN: limit test calls
        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self._half_open_max_calls:
                self._total_rejections += 1
                if fallback is not None:
                    return fallback(*args, **kwargs) if not asyncio.iscoroutinefunction(fallback) else await fallback(*args, **kwargs)
                raise CircuitBreakerError(self.name, current_state)
            self._half_open_calls += 1

        # Execute the call
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._success_threshold:
                self._transition(CircuitState.CLOSED)
        else:
            # Reset failure count on success in CLOSED state
            self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle a failed call."""
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens the circuit
            self._transition(CircuitState.OPEN)
        elif self._failure_count >= self._failure_threshold:
            self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.monotonic()

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0

        logger.info(
            "circuit_breaker_state_change",
            name=self.name,
            old_state=old_state.value,
            new_state=new_state.value,
        )

    def __repr__(self) -> str:
        return f"CircuitBreaker(name={self.name!r}, state={self._state.value!r})"

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED."""
        self._transition(CircuitState.CLOSED)


class CircuitBreakerRegistry:
    """Registry of circuit breakers for different services."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> CircuitBreaker:
        """Get an existing circuit breaker or create a new one."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return self._breakers[name]

    def all_metrics(self) -> list[dict[str, Any]]:
        """Return metrics for all circuit breakers."""
        return [cb.metrics for cb in self._breakers.values()]


# Global registry
circuit_registry = CircuitBreakerRegistry()
