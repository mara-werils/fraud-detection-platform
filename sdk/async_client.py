"""Async Python SDK client for the Fraud Detection Platform."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from sdk.models import BatchResult, CaseInfo, ScoringResult, SearchResult

__all__ = [
    "AsyncFraudClientError",
    "RateLimitError",
    "AsyncFraudClient",
]

logger = logging.getLogger("fraud_sdk.async_client")

# Default retry / backoff settings
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 0.5
_DEFAULT_BACKOFF_MAX = 30.0
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class AsyncFraudClientError(Exception):
    """Base exception for async SDK errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(AsyncFraudClientError):
    """Raised when the API returns HTTP 429."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class AsyncFraudClient:
    """Async Python SDK for the Fraud Detection Platform.

    Features:
    - Connection pooling and keep-alive via ``httpx.AsyncClient``
    - Automatic retry with exponential backoff for transient failures
    - Rate-limit handling (HTTP 429) with ``Retry-After`` support
    - Request/response logging
    - Type-safe API methods
    - Batch scoring support
    - Webhook registration
    - Real-time event streaming via WebSocket
    - Async context-manager support (``async with``)

    Example::

        async with AsyncFraudClient("http://localhost:8000", api_key="fdp_...") as client:
            result = await client.score_transaction(
                user_id="user-001",
                amount=45000.00,
                merchant_id="merch-001",
            )
            print(f"Score: {result.fraud_score}, Decision: {result.decision}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
        backoff_max: float = _DEFAULT_BACKOFF_MAX,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        pool_limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            limits=pool_limits,
        )
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncFraudClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def score_transaction(
        self,
        user_id: str,
        amount: float,
        currency: str = "USD",
        merchant_id: str | None = None,
        category: str | None = None,
        channel: str | None = None,
        transaction_type: str = "purchase",
        ip_address: str | None = None,
        device_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ScoringResult:
        """Score a single transaction.

        Args:
            user_id: ID of the user making the transaction.
            amount: Transaction amount.
            currency: ISO 4217 currency code.
            merchant_id: Merchant identifier.
            category: Merchant category.
            channel: Transaction channel (e.g., mobile_app, web).
            transaction_type: Type (purchase, transfer, withdrawal).
            ip_address: Source IP address.
            device_id: Device fingerprint.
            latitude: Transaction latitude.
            longitude: Transaction longitude.
            metadata: Additional metadata.

        Returns:
            ScoringResult with fraud score and decision.
        """
        payload: dict[str, Any] = {
            "transaction_id": str(uuid4()),
            "user_id": user_id,
            "amount": amount,
            "currency": currency,
            "transaction_type": transaction_type,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if merchant_id:
            payload["merchant_id"] = merchant_id
        if category:
            payload["merchant_category"] = category
        if channel:
            payload["metadata"] = {"channel": channel, **(metadata or {})}
        elif metadata:
            payload["metadata"] = metadata
        if ip_address:
            payload["ip_address"] = ip_address
        if device_id:
            payload["device_id"] = device_id
        if latitude is not None:
            payload["latitude"] = latitude
        if longitude is not None:
            payload["longitude"] = longitude

        resp = await self._request("POST", "/score", json=payload)
        return ScoringResult.from_response(resp.json())

    async def batch_score(
        self,
        transactions: list[dict[str, Any]],
    ) -> BatchResult:
        """Score multiple transactions in a single request.

        Args:
            transactions: List of transaction dicts matching score_transaction() params.

        Returns:
            BatchResult with all scored transactions.
        """
        resp = await self._request(
            "POST",
            "/api/v1/batch/score",
            json={"transactions": transactions},
        )
        return BatchResult.from_response(resp.json())

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    async def list_transactions(
        self,
        user_id: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        is_flagged: bool | None = None,
        decision: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SearchResult:
        """Search and list scored transactions.

        Args:
            user_id: Filter by user ID.
            min_score: Minimum fraud score.
            max_score: Maximum fraud score.
            is_flagged: Filter flagged transactions only.
            decision: Filter by decision (BLOCK/REVIEW/ALLOW).
            limit: Results per page.
            offset: Pagination offset.

        Returns:
            SearchResult with matching transactions.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if user_id:
            params["user_id"] = user_id
        if min_score is not None:
            params["min_score"] = min_score
        if max_score is not None:
            params["max_score"] = max_score
        if is_flagged is not None:
            params["is_flagged"] = is_flagged
        if decision:
            params["decision"] = decision

        resp = await self._request("GET", "/api/v1/transactions", params=params)
        return SearchResult.from_response(resp.json())

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------

    async def get_case(self, case_id: str) -> CaseInfo:
        """Get a single fraud case by ID.

        Args:
            case_id: The case identifier.

        Returns:
            CaseInfo with case details.
        """
        resp = await self._request("GET", f"/api/v1/cases/{case_id}")
        return CaseInfo.from_response(resp.json())

    async def list_cases(
        self,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 50,
    ) -> list[CaseInfo]:
        """List fraud cases.

        Args:
            status: Filter by status.
            priority: Filter by priority.
            limit: Maximum results.

        Returns:
            List of CaseInfo objects.
        """
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority

        resp = await self._request("GET", "/api/v1/cases", params=params)
        data = resp.json()
        return [CaseInfo.from_response(c) for c in data.get("cases", [])]

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    async def create_rule(
        self,
        name: str,
        condition: str,
        action: str = "flag",
        priority: int = 0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a fraud detection rule.

        Args:
            name: Human-readable rule name.
            condition: Rule condition expression.
            action: Action when rule triggers (flag, block, review).
            priority: Rule evaluation priority (higher = first).
            enabled: Whether the rule is active.
            metadata: Additional rule metadata.

        Returns:
            Created rule as a dictionary.
        """
        payload: dict[str, Any] = {
            "name": name,
            "condition": condition,
            "action": action,
            "priority": priority,
            "enabled": enabled,
        }
        if metadata:
            payload["metadata"] = metadata

        resp = await self._request("POST", "/api/v1/rules", json=payload)
        return resp.json()

    # ------------------------------------------------------------------
    # Webhooks
    # ------------------------------------------------------------------

    async def register_webhook(
        self,
        url: str,
        events: list[str],
        secret: str | None = None,
        active: bool = True,
    ) -> dict[str, Any]:
        """Register a webhook endpoint for event notifications.

        Args:
            url: The webhook URL to receive events.
            events: List of event types to subscribe to
                (e.g., ``["transaction.flagged", "case.created"]``).
            secret: Shared secret for webhook signature verification.
            active: Whether the webhook is active.

        Returns:
            Registration response with webhook ID.
        """
        payload: dict[str, Any] = {
            "url": url,
            "events": events,
            "active": active,
        }
        if secret:
            payload["secret"] = secret

        resp = await self._request("POST", "/api/v1/webhooks", json=payload)
        return resp.json()

    async def list_webhooks(self) -> list[dict[str, Any]]:
        """List all registered webhooks.

        Returns:
            List of webhook configurations.
        """
        resp = await self._request("GET", "/api/v1/webhooks")
        data = resp.json()
        return data.get("webhooks", data if isinstance(data, list) else [])

    async def delete_webhook(self, webhook_id: str) -> None:
        """Delete a webhook registration.

        Args:
            webhook_id: The webhook identifier to delete.
        """
        await self._request("DELETE", f"/api/v1/webhooks/{webhook_id}")

    # ------------------------------------------------------------------
    # Real-time event streaming (WebSocket)
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def stream_events(
        self,
        event_types: list[str] | None = None,
    ) -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
        """Stream real-time fraud events via WebSocket.

        Usage::

            async with client.stream_events(["transaction.scored"]) as stream:
                async for event in stream:
                    print(event)

        Args:
            event_types: Optional list of event types to filter.

        Yields:
            An async iterator of event dictionaries.
        """
        import json as _json

        ws_url = str(self._client.base_url).replace("http", "ws", 1)
        params = ""
        if event_types:
            params = "?events=" + ",".join(event_types)

        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "websockets package is required for streaming. "
                "Install it with: pip install websockets"
            ) from exc

        extra_headers: dict[str, str] = {}
        if "X-API-Key" in (self._client.headers or {}):
            extra_headers["X-API-Key"] = str(self._client.headers["X-API-Key"])

        async with websockets.connect(
            f"{ws_url}/ws/events{params}",
            additional_headers=extra_headers,
        ) as ws:

            async def _event_iter() -> AsyncIterator[dict[str, Any]]:
                async for message in ws:
                    try:
                        yield _json.loads(message)
                    except _json.JSONDecodeError:
                        logger.warning("Received non-JSON WebSocket message: %s", message)
                        continue

            yield _event_iter()

    # ------------------------------------------------------------------
    # Health & Info
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Check service health."""
        resp = await self._request("GET", "/health")
        return resp.json()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Execute an HTTP request with retry, backoff, and logging.

        Retries on transient errors (5xx, 429) with exponential backoff.
        Rate-limit responses honour the ``Retry-After`` header when present.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            start = time.monotonic()
            logger.debug(
                "Request attempt %d/%d: %s %s",
                attempt,
                self._max_retries,
                method,
                path,
            )

            try:
                resp = await self._client.request(
                    method, path, json=json, params=params
                )
            except httpx.TransportError as exc:
                logger.warning(
                    "Transport error on attempt %d: %s", attempt, exc
                )
                last_exc = exc
                await self._sleep_backoff(attempt)
                continue

            elapsed = (time.monotonic() - start) * 1000
            logger.debug(
                "Response %d from %s %s (%.1fms)",
                resp.status_code,
                method,
                path,
                elapsed,
            )

            if resp.status_code not in _RETRYABLE_STATUS_CODES:
                self._check_response(resp)
                return resp

            # Rate-limit handling
            if resp.status_code == 429:
                retry_after = self._parse_retry_after(resp)
                wait = retry_after if retry_after else self._backoff_delay(attempt)
                logger.warning(
                    "Rate limited (429). Retrying in %.2fs (attempt %d/%d)",
                    wait,
                    attempt,
                    self._max_retries,
                )
                last_exc = RateLimitError(
                    "Rate limited by API",
                    retry_after=retry_after,
                )
            else:
                wait = self._backoff_delay(attempt)
                logger.warning(
                    "Retryable status %d. Retrying in %.2fs (attempt %d/%d)",
                    resp.status_code,
                    wait,
                    attempt,
                    self._max_retries,
                )
                last_exc = AsyncFraudClientError(
                    f"Server error {resp.status_code}",
                    status_code=resp.status_code,
                )

            if attempt < self._max_retries:
                await asyncio.sleep(wait)

        # All retries exhausted
        if last_exc:
            raise last_exc
        raise AsyncFraudClientError("Request failed after all retries")

    def _backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay capped at backoff_max."""
        return min(self._backoff_base * (2 ** (attempt - 1)), self._backoff_max)

    async def _sleep_backoff(self, attempt: int) -> None:
        if attempt < self._max_retries:
            await asyncio.sleep(self._backoff_delay(attempt))

    @staticmethod
    def _parse_retry_after(resp: httpx.Response) -> float | None:
        """Parse the Retry-After header if present."""
        value = resp.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _check_response(resp: httpx.Response) -> None:
        """Raise AsyncFraudClientError for non-2xx responses."""
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise AsyncFraudClientError(
                f"API error {resp.status_code}: {detail}",
                status_code=resp.status_code,
            )
