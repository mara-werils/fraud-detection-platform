"""Tests for the async fraud detection SDK client."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from sdk.async_client import (
    AsyncFraudClient,
    AsyncFraudClientError,
    RateLimitError,
    _RETRYABLE_STATUS_CODES,
)
from sdk.models import BatchResult, CaseInfo, ScoringResult, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_response(data: dict[str, Any], status_code: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    """Build a synthetic httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        headers=headers or {},
        request=httpx.Request("GET", "http://test"),
    )


def _transport_error() -> httpx.TransportError:
    return httpx.ConnectError("connection refused")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    @pytest.mark.asyncio
    async def test_async_with(self) -> None:
        async with AsyncFraudClient("http://localhost:8000") as client:
            assert client is not None
            assert isinstance(client, AsyncFraudClient)

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        client = AsyncFraudClient("http://localhost:8000")
        await client.close()


# ---------------------------------------------------------------------------
# Connection pool configuration
# ---------------------------------------------------------------------------

class TestConnectionPooling:
    def test_default_pool_limits(self) -> None:
        client = AsyncFraudClient()
        assert client._client is not None

    def test_custom_pool_limits(self) -> None:
        client = AsyncFraudClient(max_connections=50, max_keepalive_connections=10)
        assert client._client is not None

    def test_api_key_header(self) -> None:
        client = AsyncFraudClient(api_key="fdp_test_key")
        assert client._client.headers["X-API-Key"] == "fdp_test_key"

    def test_no_api_key(self) -> None:
        client = AsyncFraudClient()
        assert "X-API-Key" not in client._client.headers


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------

class TestRetryBackoff:
    def test_backoff_delay_exponential(self) -> None:
        client = AsyncFraudClient(backoff_base=1.0, backoff_max=30.0)
        assert client._backoff_delay(1) == 1.0
        assert client._backoff_delay(2) == 2.0
        assert client._backoff_delay(3) == 4.0
        assert client._backoff_delay(4) == 8.0

    def test_backoff_delay_capped(self) -> None:
        client = AsyncFraudClient(backoff_base=1.0, backoff_max=5.0)
        assert client._backoff_delay(10) == 5.0

    @pytest.mark.asyncio
    async def test_retry_on_500(self) -> None:
        client = AsyncFraudClient(max_retries=3, backoff_base=0.0)
        call_count = 0

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return _json_response({"error": "oops"}, status_code=500)
            return _json_response({"status": "ok"})

        client._client.request = mock_request  # type: ignore[assignment]

        resp = await client._request("GET", "/health")
        assert resp.status_code == 200
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self) -> None:
        client = AsyncFraudClient(max_retries=2, backoff_base=0.0)

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            return _json_response({"error": "fail"}, status_code=503)

        client._client.request = mock_request  # type: ignore[assignment]

        with pytest.raises(AsyncFraudClientError) as exc_info:
            await client._request("GET", "/health")
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_retry_on_transport_error(self) -> None:
        client = AsyncFraudClient(max_retries=3, backoff_base=0.0)
        call_count = 0

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _transport_error()
            return _json_response({"status": "ok"})

        client._client.request = mock_request  # type: ignore[assignment]

        resp = await client._request("GET", "/health")
        assert resp.status_code == 200
        assert call_count == 3


# ---------------------------------------------------------------------------
# Rate-limit handling
# ---------------------------------------------------------------------------

class TestRateLimitHandling:
    @pytest.mark.asyncio
    async def test_rate_limit_retry_after(self) -> None:
        client = AsyncFraudClient(max_retries=3, backoff_base=0.0)
        call_count = 0

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _json_response(
                    {"error": "rate limited"},
                    status_code=429,
                    headers={"Retry-After": "0"},
                )
            return _json_response({"status": "ok"})

        client._client.request = mock_request  # type: ignore[assignment]

        resp = await client._request("GET", "/test")
        assert resp.status_code == 200
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_exhausted(self) -> None:
        client = AsyncFraudClient(max_retries=2, backoff_base=0.0)

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            return _json_response(
                {"error": "rate limited"},
                status_code=429,
                headers={"Retry-After": "0"},
            )

        client._client.request = mock_request  # type: ignore[assignment]

        with pytest.raises(RateLimitError):
            await client._request("GET", "/test")

    def test_parse_retry_after_float(self) -> None:
        resp = _json_response({}, headers={"Retry-After": "1.5"})
        assert AsyncFraudClient._parse_retry_after(resp) == 1.5

    def test_parse_retry_after_missing(self) -> None:
        resp = _json_response({})
        assert AsyncFraudClient._parse_retry_after(resp) is None

    def test_parse_retry_after_invalid(self) -> None:
        resp = _json_response({}, headers={"Retry-After": "not-a-number"})
        assert AsyncFraudClient._parse_retry_after(resp) is None


# ---------------------------------------------------------------------------
# score_transaction()
# ---------------------------------------------------------------------------

class TestScoreTransaction:
    @pytest.mark.asyncio
    async def test_score_transaction_basic(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            return _json_response({
                "transaction_id": "txn-001",
                "user_id": "user-001",
                "fraud_score": 0.15,
                "is_flagged": False,
                "model_version": "1.0",
                "scoring_latency_ms": 12.3,
            })

        client._client.request = mock_request  # type: ignore[assignment]

        result = await client.score_transaction(user_id="user-001", amount=100.0)
        assert isinstance(result, ScoringResult)
        assert result.user_id == "user-001"
        assert result.fraud_score == 0.15
        assert result.decision == "ALLOW"

    @pytest.mark.asyncio
    async def test_score_transaction_with_all_params(self) -> None:
        client = AsyncFraudClient(max_retries=1)
        captured: dict[str, Any] = {}

        async def mock_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
            captured["json"] = kwargs.get("json")
            return _json_response({
                "transaction_id": "txn-002",
                "user_id": "user-002",
                "fraud_score": 0.85,
                "is_flagged": True,
                "flag_reason": "high amount",
                "model_version": "1.0",
                "scoring_latency_ms": 15.0,
            })

        client._client.request = mock_request  # type: ignore[assignment]

        result = await client.score_transaction(
            user_id="user-002",
            amount=99999.0,
            currency="EUR",
            merchant_id="merch-001",
            category="electronics",
            channel="web",
            transaction_type="purchase",
            ip_address="1.2.3.4",
            device_id="dev-001",
            latitude=40.7,
            longitude=-74.0,
            metadata={"extra": "data"},
        )
        assert result.decision == "BLOCK"
        assert result.is_flagged is True
        assert captured["json"]["merchant_id"] == "merch-001"
        assert captured["json"]["ip_address"] == "1.2.3.4"
        assert captured["json"]["metadata"]["channel"] == "web"
        assert captured["json"]["metadata"]["extra"] == "data"


# ---------------------------------------------------------------------------
# batch_score()
# ---------------------------------------------------------------------------

class TestBatchScore:
    @pytest.mark.asyncio
    async def test_batch_score(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            return _json_response({
                "results": [
                    {
                        "transaction_id": "txn-a",
                        "user_id": "u1",
                        "fraud_score": 0.1,
                        "is_flagged": False,
                        "model_version": "1.0",
                        "scoring_latency_ms": 5.0,
                    },
                    {
                        "transaction_id": "txn-b",
                        "user_id": "u2",
                        "fraud_score": 0.9,
                        "is_flagged": True,
                        "model_version": "1.0",
                        "scoring_latency_ms": 6.0,
                    },
                ],
                "total": 2,
                "failed": 0,
                "latency_ms": 11.0,
            })

        client._client.request = mock_request  # type: ignore[assignment]

        result = await client.batch_score([
            {"user_id": "u1", "amount": 10.0},
            {"user_id": "u2", "amount": 50000.0},
        ])
        assert isinstance(result, BatchResult)
        assert result.total == 2
        assert result.failed == 0
        assert len(result.results) == 2
        assert result.results[1].fraud_score == 0.9


# ---------------------------------------------------------------------------
# get_case()
# ---------------------------------------------------------------------------

class TestGetCase:
    @pytest.mark.asyncio
    async def test_get_case(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
            assert "/api/v1/cases/case-123" in url
            return _json_response({
                "case_id": "case-123",
                "transaction_id": "txn-001",
                "user_id": "user-001",
                "status": "open",
                "priority": "high",
                "assigned_to": "analyst-1",
                "created_at": "2025-01-01T00:00:00Z",
                "notes": ["suspicious pattern"],
            })

        client._client.request = mock_request  # type: ignore[assignment]

        case = await client.get_case("case-123")
        assert isinstance(case, CaseInfo)
        assert case.case_id == "case-123"
        assert case.status == "open"
        assert case.priority == "high"

    @pytest.mark.asyncio
    async def test_get_case_not_found(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            return _json_response({"detail": "not found"}, status_code=404)

        client._client.request = mock_request  # type: ignore[assignment]

        with pytest.raises(AsyncFraudClientError) as exc_info:
            await client.get_case("nonexistent")
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# list_transactions()
# ---------------------------------------------------------------------------

class TestListTransactions:
    @pytest.mark.asyncio
    async def test_list_transactions(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
            return _json_response({
                "transactions": [
                    {
                        "transaction_id": "txn-1",
                        "user_id": "u1",
                        "fraud_score": 0.3,
                        "is_flagged": False,
                        "model_version": "1.0",
                        "scoring_latency_ms": 5.0,
                    }
                ],
                "total": 1,
                "has_more": False,
            })

        client._client.request = mock_request  # type: ignore[assignment]

        result = await client.list_transactions(user_id="u1", min_score=0.1)
        assert isinstance(result, SearchResult)
        assert result.total == 1
        assert len(result.transactions) == 1

    @pytest.mark.asyncio
    async def test_list_transactions_passes_params(self) -> None:
        client = AsyncFraudClient(max_retries=1)
        captured: dict[str, Any] = {}

        async def mock_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
            captured["params"] = kwargs.get("params")
            return _json_response({"transactions": [], "total": 0, "has_more": False})

        client._client.request = mock_request  # type: ignore[assignment]

        await client.list_transactions(
            user_id="u1",
            min_score=0.5,
            max_score=0.9,
            is_flagged=True,
            decision="BLOCK",
            limit=10,
            offset=5,
        )
        p = captured["params"]
        assert p["user_id"] == "u1"
        assert p["min_score"] == 0.5
        assert p["limit"] == 10
        assert p["offset"] == 5


# ---------------------------------------------------------------------------
# create_rule()
# ---------------------------------------------------------------------------

class TestCreateRule:
    @pytest.mark.asyncio
    async def test_create_rule(self) -> None:
        client = AsyncFraudClient(max_retries=1)
        captured: dict[str, Any] = {}

        async def mock_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
            captured["json"] = kwargs.get("json")
            return _json_response({
                "rule_id": "rule-001",
                "name": "high-amount",
                "condition": "amount > 10000",
                "action": "block",
                "priority": 10,
                "enabled": True,
            })

        client._client.request = mock_request  # type: ignore[assignment]

        result = await client.create_rule(
            name="high-amount",
            condition="amount > 10000",
            action="block",
            priority=10,
        )
        assert result["rule_id"] == "rule-001"
        assert captured["json"]["name"] == "high-amount"
        assert captured["json"]["condition"] == "amount > 10000"
        assert captured["json"]["action"] == "block"
        assert captured["json"]["priority"] == 10


# ---------------------------------------------------------------------------
# Webhook registration
# ---------------------------------------------------------------------------

class TestWebhooks:
    @pytest.mark.asyncio
    async def test_register_webhook(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
            return _json_response({
                "webhook_id": "wh-001",
                "url": "https://example.com/hook",
                "events": ["transaction.flagged"],
                "active": True,
            })

        client._client.request = mock_request  # type: ignore[assignment]

        result = await client.register_webhook(
            url="https://example.com/hook",
            events=["transaction.flagged"],
            secret="s3cret",
        )
        assert result["webhook_id"] == "wh-001"

    @pytest.mark.asyncio
    async def test_list_webhooks(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            return _json_response({
                "webhooks": [{"webhook_id": "wh-001", "url": "https://example.com/hook"}]
            })

        client._client.request = mock_request  # type: ignore[assignment]

        hooks = await client.list_webhooks()
        assert len(hooks) == 1
        assert hooks[0]["webhook_id"] == "wh-001"

    @pytest.mark.asyncio
    async def test_delete_webhook(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            return _json_response({}, status_code=204)

        client._client.request = mock_request  # type: ignore[assignment]

        # Should not raise
        await client.delete_webhook("wh-001")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            return _json_response({"status": "healthy", "version": "1.1.0"})

        client._client.request = mock_request  # type: ignore[assignment]

        result = await client.health()
        assert result["status"] == "healthy"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_4xx_not_retried(self) -> None:
        client = AsyncFraudClient(max_retries=3, backoff_base=0.0)
        call_count = 0

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"detail": "bad request"}, status_code=400)

        client._client.request = mock_request  # type: ignore[assignment]

        with pytest.raises(AsyncFraudClientError) as exc_info:
            await client._request("POST", "/test")
        assert exc_info.value.status_code == 400
        assert call_count == 1  # Not retried

    @pytest.mark.asyncio
    async def test_non_json_error_body(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                status_code=400,
                text="plain error",
                request=httpx.Request("GET", "http://test"),
            )

        client._client.request = mock_request  # type: ignore[assignment]

        with pytest.raises(AsyncFraudClientError):
            await client._request("GET", "/test")


# ---------------------------------------------------------------------------
# WebSocket stream_events (import check)
# ---------------------------------------------------------------------------

class TestStreamEvents:
    @pytest.mark.asyncio
    async def test_stream_events_requires_websockets(self) -> None:
        """Verify that stream_events raises ImportError when websockets is missing."""
        client = AsyncFraudClient(max_retries=1)

        with (
            patch.dict("sys.modules", {"websockets": None}),
            pytest.raises(ImportError, match="websockets"),
        ):
            async with client.stream_events(["transaction.scored"]):
                pass


# ---------------------------------------------------------------------------
# list_cases()
# ---------------------------------------------------------------------------

class TestListCases:
    @pytest.mark.asyncio
    async def test_list_cases(self) -> None:
        client = AsyncFraudClient(max_retries=1)

        async def mock_request(*args: Any, **kwargs: Any) -> httpx.Response:
            return _json_response({
                "cases": [
                    {
                        "case_id": "c-1",
                        "transaction_id": "t-1",
                        "user_id": "u-1",
                        "status": "open",
                        "priority": "high",
                        "assigned_to": None,
                        "created_at": "2025-01-01T00:00:00Z",
                    },
                ]
            })

        client._client.request = mock_request  # type: ignore[assignment]

        cases = await client.list_cases(status="open", priority="high")
        assert len(cases) == 1
        assert isinstance(cases[0], CaseInfo)
        assert cases[0].case_id == "c-1"
