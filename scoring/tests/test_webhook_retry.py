"""Tests for the webhook retry service."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from scoring.services.webhook_retry import (
    AttemptRecord,
    CircuitBreaker,
    CircuitState,
    DeliveryRecord,
    DeliveryStatus,
    RetryPolicy,
    WebhookEndpoint,
    WebhookEvent,
    WebhookRetryService,
    WebhookSignature,
    _compute_delay,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> WebhookRetryService:
    """Service with fast retry policy for tests."""
    return WebhookRetryService(
        default_policy=RetryPolicy(
            max_retries=3,
            initial_delay_seconds=0.01,
            max_delay_seconds=0.1,
            backoff_multiplier=2.0,
            jitter=False,
            timeout_seconds=5.0,
        ),
        circuit_failure_threshold=3,
        circuit_recovery_timeout=0.1,
    )


@pytest.fixture
def endpoint(service: WebhookRetryService) -> WebhookEndpoint:
    return service.register_endpoint(
        url="https://example.com/webhook",
        signing_secret="test_secret_123",
    )


@pytest.fixture
def event(endpoint: WebhookEndpoint) -> WebhookEvent:
    return WebhookEvent(
        endpoint_id=endpoint.endpoint_id,
        event_type="transaction.flagged",
        payload={"transaction_id": "tx_123", "score": 0.95},
    )


def _make_response(status_code: int = 200, text: str = "OK") -> httpx.Response:
    """Build a minimal httpx.Response for mocking."""
    return httpx.Response(
        status_code=status_code,
        text=text,
        request=httpx.Request("POST", "https://example.com/webhook"),
    )


# ---------------------------------------------------------------------------
# WebhookSignature
# ---------------------------------------------------------------------------


class TestWebhookSignature:
    def test_sign_and_verify(self) -> None:
        body = b'{"event": "test"}'
        secret = "my_secret"
        sig = WebhookSignature.sign(body, secret)
        header = f"sha256={sig}"
        assert WebhookSignature.verify(body, secret, header) is True

    def test_verify_rejects_wrong_secret(self) -> None:
        body = b'{"event": "test"}'
        sig = WebhookSignature.sign(body, "correct_secret")
        header = f"sha256={sig}"
        assert WebhookSignature.verify(body, "wrong_secret", header) is False

    def test_verify_rejects_bad_prefix(self) -> None:
        body = b"hello"
        assert WebhookSignature.verify(body, "secret", "md5=abc") is False

    def test_sign_with_timestamp(self) -> None:
        body = b'{"event": "test"}'
        secret = "ts_secret"
        ts = str(int(time.time()))
        sig = WebhookSignature.sign(body, secret, timestamp=ts)
        header = f"sha256={sig}"
        assert WebhookSignature.verify(body, secret, header, timestamp=ts) is True

    def test_verify_rejects_stale_timestamp(self) -> None:
        body = b'{"event": "test"}'
        secret = "ts_secret"
        stale_ts = "1000000000"  # very old
        sig = WebhookSignature.sign(body, secret, timestamp=stale_ts)
        header = f"sha256={sig}"
        assert (
            WebhookSignature.verify(
                body, secret, header, timestamp=stale_ts, tolerance_seconds=60.0
            )
            is False
        )

    def test_verify_rejects_invalid_timestamp(self) -> None:
        body = b"data"
        secret = "sec"
        sig = WebhookSignature.sign(body, secret, timestamp="not_a_number")
        header = f"sha256={sig}"
        assert (
            WebhookSignature.verify(body, secret, header, timestamp="not_a_number")
            is False
        )

    def test_sign_deterministic(self) -> None:
        body = b"same"
        secret = "key"
        assert WebhookSignature.sign(body, secret) == WebhookSignature.sign(
            body, secret
        )

    def test_different_bodies_different_signatures(self) -> None:
        secret = "key"
        sig1 = WebhookSignature.sign(b"body1", secret)
        sig2 = WebhookSignature.sign(b"body2", secret)
        assert sig1 != sig2


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_stays_closed_below_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_recovery_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Simulate passage of time
        cb.last_state_change = time.monotonic() - 0.1
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        cb.last_state_change = time.monotonic() - 0.1
        cb.allow_request()  # transitions to HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        cb.last_state_change = time.monotonic() - 0.1
        cb.allow_request()  # transitions to HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_reset(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# _compute_delay
# ---------------------------------------------------------------------------


class TestComputeDelay:
    def test_exponential_without_jitter(self) -> None:
        policy = RetryPolicy(
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_delay_seconds=60.0,
            jitter=False,
        )
        assert _compute_delay(policy, 1) == 1.0
        assert _compute_delay(policy, 2) == 2.0
        assert _compute_delay(policy, 3) == 4.0
        assert _compute_delay(policy, 4) == 8.0

    def test_clamped_to_max(self) -> None:
        policy = RetryPolicy(
            initial_delay_seconds=10.0,
            backoff_multiplier=10.0,
            max_delay_seconds=50.0,
            jitter=False,
        )
        assert _compute_delay(policy, 3) == 50.0

    def test_jitter_within_bounds(self) -> None:
        policy = RetryPolicy(
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_delay_seconds=60.0,
            jitter=True,
        )
        for _ in range(100):
            delay = _compute_delay(policy, 2)
            assert 0 <= delay <= 2.0


# ---------------------------------------------------------------------------
# AttemptRecord
# ---------------------------------------------------------------------------


class TestAttemptRecord:
    def test_succeeded_2xx(self) -> None:
        assert AttemptRecord(attempt_number=1, status_code=200).succeeded is True
        assert AttemptRecord(attempt_number=1, status_code=201).succeeded is True
        assert AttemptRecord(attempt_number=1, status_code=204).succeeded is True

    def test_failed_non_2xx(self) -> None:
        assert AttemptRecord(attempt_number=1, status_code=500).succeeded is False
        assert AttemptRecord(attempt_number=1, status_code=400).succeeded is False

    def test_failed_no_status(self) -> None:
        assert AttemptRecord(attempt_number=1, error="timeout").succeeded is False


# ---------------------------------------------------------------------------
# DeliveryRecord
# ---------------------------------------------------------------------------


class TestDeliveryRecord:
    def test_total_attempts(self) -> None:
        rec = DeliveryRecord(
            event_id="e1",
            endpoint_id="ep1",
            event_type="test",
            attempts=[
                AttemptRecord(attempt_number=1, status_code=500),
                AttemptRecord(attempt_number=2, status_code=200),
            ],
        )
        assert rec.total_attempts == 2


# ---------------------------------------------------------------------------
# WebhookRetryService — endpoint management
# ---------------------------------------------------------------------------


class TestEndpointManagement:
    def test_register_and_get(self, service: WebhookRetryService) -> None:
        ep = service.register_endpoint("https://a.com/hook", signing_secret="s")
        assert service.get_endpoint(ep.endpoint_id) is ep
        assert ep.url == "https://a.com/hook"
        assert ep.signing_secret == "s"

    def test_list_endpoints(self, service: WebhookRetryService) -> None:
        service.register_endpoint("https://a.com")
        service.register_endpoint("https://b.com")
        assert len(service.list_endpoints()) == 2

    def test_unregister(self, service: WebhookRetryService) -> None:
        ep = service.register_endpoint("https://a.com")
        assert service.unregister_endpoint(ep.endpoint_id) is True
        assert service.get_endpoint(ep.endpoint_id) is None

    def test_unregister_nonexistent(self, service: WebhookRetryService) -> None:
        assert service.unregister_endpoint("nope") is False

    def test_custom_retry_policy(self, service: WebhookRetryService) -> None:
        custom = RetryPolicy(max_retries=10, initial_delay_seconds=5.0)
        ep = service.register_endpoint("https://a.com", retry_policy=custom)
        assert ep.retry_policy.max_retries == 10
        assert ep.retry_policy.initial_delay_seconds == 5.0


# ---------------------------------------------------------------------------
# WebhookRetryService — circuit breaker access
# ---------------------------------------------------------------------------


class TestCircuitBreakerAccess:
    def test_get_circuit_state(
        self, service: WebhookRetryService, endpoint: WebhookEndpoint
    ) -> None:
        assert service.get_circuit_state(endpoint.endpoint_id) == CircuitState.CLOSED

    def test_get_circuit_state_unknown(self, service: WebhookRetryService) -> None:
        assert service.get_circuit_state("unknown") is None

    def test_reset_circuit(
        self, service: WebhookRetryService, endpoint: WebhookEndpoint
    ) -> None:
        assert service.reset_circuit(endpoint.endpoint_id) is True

    def test_reset_circuit_unknown(self, service: WebhookRetryService) -> None:
        assert service.reset_circuit("unknown") is False


# ---------------------------------------------------------------------------
# WebhookRetryService — delivery
# ---------------------------------------------------------------------------


class TestDelivery:
    @pytest.mark.asyncio
    async def test_successful_delivery(
        self,
        service: WebhookRetryService,
        event: WebhookEvent,
    ) -> None:
        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(200)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            record = await service.deliver(event)

        assert record.status == DeliveryStatus.DELIVERED
        assert record.total_attempts == 1
        assert record.attempts[0].status_code == 200

    @pytest.mark.asyncio
    async def test_retries_on_500_then_succeeds(
        self,
        service: WebhookRetryService,
        event: WebhookEvent,
    ) -> None:
        responses = [_make_response(500, "error"), _make_response(200, "ok")]

        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = responses
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            record = await service.deliver(event)

        assert record.status == DeliveryStatus.DELIVERED
        assert record.total_attempts == 2

    @pytest.mark.asyncio
    async def test_non_retryable_status_stops_early(
        self,
        service: WebhookRetryService,
        event: WebhookEvent,
    ) -> None:
        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(404, "not found")
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            record = await service.deliver(event)

        # 404 is not retryable, so only one attempt
        assert record.total_attempts == 1
        assert record.status == DeliveryStatus.DEAD_LETTERED

    @pytest.mark.asyncio
    async def test_exhausts_retries_then_dead_letters(
        self,
        service: WebhookRetryService,
        event: WebhookEvent,
    ) -> None:
        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(502, "bad gw")
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            record = await service.deliver(event)

        # Circuit breaker (threshold=3) opens after 3 consecutive failures,
        # halting further attempts even though max_retries would allow a 4th.
        assert record.total_attempts == 3
        assert record.status == DeliveryStatus.DEAD_LETTERED
        assert service.dead_letter_queue_size() == 1

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(
        self,
        service: WebhookRetryService,
        event: WebhookEvent,
    ) -> None:
        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [
                httpx.ReadTimeout("timeout"),
                _make_response(200),
            ]
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            record = await service.deliver(event)

        assert record.status == DeliveryStatus.DELIVERED
        assert record.total_attempts == 2
        assert "timeout" in record.attempts[0].error

    @pytest.mark.asyncio
    async def test_request_error_triggers_retry(
        self,
        service: WebhookRetryService,
        event: WebhookEvent,
    ) -> None:
        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [
                httpx.ConnectError("connection refused"),
                _make_response(200),
            ]
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            record = await service.deliver(event)

        assert record.status == DeliveryStatus.DELIVERED
        assert record.total_attempts == 2

    @pytest.mark.asyncio
    async def test_unknown_endpoint_raises(
        self, service: WebhookRetryService
    ) -> None:
        event = WebhookEvent(
            endpoint_id="nonexistent",
            event_type="test",
            payload={},
        )
        with pytest.raises(ValueError, match="Unknown endpoint"):
            await service.deliver(event)

    @pytest.mark.asyncio
    async def test_dlq_disabled(self) -> None:
        svc = WebhookRetryService(
            default_policy=RetryPolicy(
                max_retries=0,
                initial_delay_seconds=0.01,
                jitter=False,
                timeout_seconds=1.0,
            ),
            dlq_enabled=False,
        )
        ep = svc.register_endpoint("https://fail.com")
        event = WebhookEvent(
            endpoint_id=ep.endpoint_id, event_type="test", payload={}
        )

        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(500)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            record = await svc.deliver(event)

        assert record.status == DeliveryStatus.FAILED
        assert svc.dead_letter_queue_size() == 0


# ---------------------------------------------------------------------------
# WebhookRetryService — circuit breaker integration
# ---------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(
        self,
        service: WebhookRetryService,
        endpoint: WebhookEndpoint,
    ) -> None:
        """Deliver enough failing events to trip the circuit breaker."""
        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(500)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            # Each delivery will make multiple attempts, all failing 500
            for _ in range(2):
                event = WebhookEvent(
                    endpoint_id=endpoint.endpoint_id,
                    event_type="test",
                    payload={},
                )
                await service.deliver(event)

        # After many failures the circuit should be open
        state = service.get_circuit_state(endpoint.endpoint_id)
        assert state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_open_skips_delivery(
        self,
        service: WebhookRetryService,
        endpoint: WebhookEndpoint,
    ) -> None:
        """When the circuit is open, events are immediately dead-lettered."""
        # Force the circuit open
        cb = service._circuit_breakers[endpoint.endpoint_id]
        cb.state = CircuitState.OPEN
        cb.last_state_change = time.monotonic()

        event = WebhookEvent(
            endpoint_id=endpoint.endpoint_id,
            event_type="test",
            payload={},
        )
        record = await service.deliver(event)
        assert record.status == DeliveryStatus.DEAD_LETTERED
        assert record.total_attempts == 0


# ---------------------------------------------------------------------------
# WebhookRetryService — batch delivery
# ---------------------------------------------------------------------------


class TestBatchDelivery:
    @pytest.mark.asyncio
    async def test_deliver_batch_empty(
        self, service: WebhookRetryService
    ) -> None:
        results = await service.deliver_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_deliver_batch_multiple(
        self,
        service: WebhookRetryService,
        endpoint: WebhookEndpoint,
    ) -> None:
        events = [
            WebhookEvent(
                endpoint_id=endpoint.endpoint_id,
                event_type="test",
                payload={"i": i},
            )
            for i in range(3)
        ]

        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(200)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            results = await service.deliver_batch(events)

        assert len(results) == 3
        assert all(r.status == DeliveryStatus.DELIVERED for r in results)


# ---------------------------------------------------------------------------
# WebhookRetryService — dead-letter queue
# ---------------------------------------------------------------------------


class TestDeadLetterQueue:
    @pytest.mark.asyncio
    async def test_retry_dead_letters_recovers(
        self,
        service: WebhookRetryService,
        endpoint: WebhookEndpoint,
    ) -> None:
        event = WebhookEvent(
            endpoint_id=endpoint.endpoint_id,
            event_type="test",
            payload={},
        )

        # First delivery fails
        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(502)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await service.deliver(event)

        assert service.dead_letter_queue_size() == 1

        # Reset circuit so retry can proceed
        service.reset_circuit(endpoint.endpoint_id)

        # Retry succeeds
        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(200)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            results = await service.retry_dead_letters()

        assert len(results) == 1
        assert results[0].status == DeliveryStatus.DELIVERED
        assert service.dead_letter_queue_size() == 0

    @pytest.mark.asyncio
    async def test_retry_dead_letters_empty(
        self, service: WebhookRetryService
    ) -> None:
        results = await service.retry_dead_letters()
        assert results == []

    def test_purge_dlq(self, service: WebhookRetryService, endpoint: WebhookEndpoint) -> None:
        # Manually add items to DLQ
        for i in range(5):
            service._dlq.append(
                WebhookEvent(
                    endpoint_id=endpoint.endpoint_id,
                    event_type="test",
                    payload={"i": i},
                )
            )
        assert service.dead_letter_queue_size() == 5
        count = service.purge_dead_letter_queue()
        assert count == 5
        assert service.dead_letter_queue_size() == 0

    def test_get_dead_letter_queue_returns_snapshot(
        self, service: WebhookRetryService, endpoint: WebhookEndpoint
    ) -> None:
        event = WebhookEvent(
            endpoint_id=endpoint.endpoint_id,
            event_type="test",
            payload={},
        )
        service._dlq.append(event)
        snapshot = service.get_dead_letter_queue()
        assert len(snapshot) == 1
        assert snapshot[0].event_id == event.event_id
        # Verify it is a copy
        snapshot.clear()
        assert service.dead_letter_queue_size() == 1


# ---------------------------------------------------------------------------
# WebhookRetryService — delivery history & stats
# ---------------------------------------------------------------------------


class TestDeliveryHistory:
    @pytest.mark.asyncio
    async def test_history_populated(
        self,
        service: WebhookRetryService,
        endpoint: WebhookEndpoint,
    ) -> None:
        event = WebhookEvent(
            endpoint_id=endpoint.endpoint_id,
            event_type="test",
            payload={},
        )

        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(200)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await service.deliver(event)

        history = service.get_delivery_history()
        assert len(history) == 1
        assert history[0].event_id == event.event_id

    @pytest.mark.asyncio
    async def test_history_filtered_by_endpoint(
        self,
        service: WebhookRetryService,
    ) -> None:
        ep1 = service.register_endpoint("https://a.com")
        ep2 = service.register_endpoint("https://b.com")

        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(200)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            for ep in [ep1, ep2, ep1]:
                ev = WebhookEvent(
                    endpoint_id=ep.endpoint_id, event_type="t", payload={}
                )
                await service.deliver(ev)

        assert len(service.get_delivery_history(endpoint_id=ep1.endpoint_id)) == 2
        assert len(service.get_delivery_history(endpoint_id=ep2.endpoint_id)) == 1

    @pytest.mark.asyncio
    async def test_delivery_stats(
        self,
        service: WebhookRetryService,
        endpoint: WebhookEndpoint,
    ) -> None:
        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_response(200)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            event = WebhookEvent(
                endpoint_id=endpoint.endpoint_id, event_type="t", payload={}
            )
            await service.deliver(event)

        stats = service.get_delivery_stats()
        assert stats["total"] == 1
        assert stats["delivered"] == 1
        assert stats["failed"] == 0
        assert stats["endpoints"] >= 1


# ---------------------------------------------------------------------------
# WebhookRetryService — signature in delivery
# ---------------------------------------------------------------------------


class TestDeliverySignature:
    @pytest.mark.asyncio
    async def test_signed_delivery_includes_signature_header(
        self,
        service: WebhookRetryService,
        endpoint: WebhookEndpoint,
        event: WebhookEvent,
    ) -> None:
        """Verify that delivery to a signed endpoint includes the signature header."""
        captured_headers: dict[str, str] = {}

        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()

            async def capture_post(url: str, content: bytes, headers: dict) -> httpx.Response:
                captured_headers.update(headers)
                return _make_response(200)

            mock_client.post.side_effect = capture_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await service.deliver(event)

        assert "X-Webhook-Signature" in captured_headers
        assert captured_headers["X-Webhook-Signature"].startswith("sha256=")
        assert "X-Webhook-Timestamp" in captured_headers

    @pytest.mark.asyncio
    async def test_unsigned_delivery_omits_signature(
        self,
        service: WebhookRetryService,
    ) -> None:
        ep = service.register_endpoint("https://nosig.com", signing_secret=None)
        event = WebhookEvent(
            endpoint_id=ep.endpoint_id, event_type="test", payload={}
        )
        captured_headers: dict[str, str] = {}

        with patch("scoring.services.webhook_retry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()

            async def capture_post(url: str, content: bytes, headers: dict) -> httpx.Response:
                captured_headers.update(headers)
                return _make_response(200)

            mock_client.post.side_effect = capture_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await service.deliver(event)

        assert "X-Webhook-Signature" not in captured_headers
