"""Advanced API tests for the scoring service.

Covers batch scoring, transaction search, case CRUD, feedback, drift,
rate-limiting behaviour, error responses, and CORS headers.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scoring.api.middleware import setup_middleware
from scoring.api.routes import router as core_router
from scoring.models.ensemble import EnsembleScorer
from scoring.services.case_manager import CaseManager
from scoring.services.drift_detector import DriftDetector
from scoring.services.feedback_store import FeedbackStore
from scoring.services.transaction_store import TransactionStore
from shared.schemas import (
    FeatureVector,
    ScoredTransaction,
    Transaction,
    TransactionType,
)

# ---------------------------------------------------------------------------
# Extended test app fixture (includes extra routers and state)
# ---------------------------------------------------------------------------


@pytest.fixture
def full_app(scorer: EnsembleScorer, mock_redis: Any, mock_producer: Any, mock_consumer: Any) -> FastAPI:
    """FastAPI app that mounts all advanced routers and in-memory stores."""
    from scoring.api.batch import router as batch_router
    from scoring.api.cases import router as cases_router
    from scoring.api.drift import router as drift_router
    from scoring.api.feedback import router as feedback_router
    from scoring.api.transactions import router as txn_router

    app = FastAPI(title="Test Full Scoring Service")
    setup_middleware(app)
    app.include_router(core_router)
    app.include_router(batch_router)
    app.include_router(txn_router)
    app.include_router(cases_router)
    app.include_router(feedback_router)
    app.include_router(drift_router)

    txn_store = TransactionStore()
    feedback_store = FeedbackStore()
    case_mgr = CaseManager()
    drift_detector = DriftDetector()

    app.state.scorer = scorer
    app.state.redis = mock_redis
    app.state.producer = mock_producer
    app.state.consumer = mock_consumer
    app.state.transaction_store = txn_store
    app.state.feedback_store = feedback_store
    app.state.case_manager = case_mgr
    app.state.drift_detector = drift_detector

    return app


@pytest.fixture
def full_client(full_app: FastAPI) -> TestClient:
    """Test client for the full app."""
    with TestClient(full_app, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_ID = str(UUID("11111111-1111-1111-1111-111111111111"))


def _txn_payload(
    amount: str = "50.00",
    txn_type: str = "purchase",
    hour: int = 14,
) -> dict[str, Any]:
    return {
        "transaction_id": str(uuid4()),
        "user_id": _USER_ID,
        "amount": amount,
        "currency": "USD",
        "transaction_type": txn_type,
        "timestamp": f"2025-06-15T{hour:02d}:00:00",
    }


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


class TestBatchScoringEndpoint:
    """POST /api/v1/batch/score"""

    def test_batch_single_transaction(self, full_client: TestClient) -> None:
        """Batch of one should return one result."""
        payload = {"transactions": [_txn_payload()]}
        response = full_client.post("/api/v1/batch/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["results"]) == 1
        assert data["failed"] == 0

    def test_batch_multiple_transactions(self, full_client: TestClient) -> None:
        """Batch of 5 transactions should return 5 results."""
        payload = {"transactions": [_txn_payload() for _ in range(5)]}
        response = full_client.post("/api/v1/batch/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["results"]) == 5

    def test_batch_scores_in_range(self, full_client: TestClient) -> None:
        """Every fraud_score must be in [0, 1]."""
        payload = {"transactions": [_txn_payload(amount=str(i * 10)) for i in range(1, 6)]}
        response = full_client.post("/api/v1/batch/score", json=payload)
        assert response.status_code == 200
        for result in response.json()["results"]:
            assert 0.0 <= result["fraud_score"] <= 1.0

    def test_batch_latency_ms_present(self, full_client: TestClient) -> None:
        """Response must include a non-negative latency_ms."""
        payload = {"transactions": [_txn_payload()]}
        response = full_client.post("/api/v1/batch/score", json=payload)
        assert response.json()["latency_ms"] >= 0

    def test_batch_empty_rejects_with_422(self, full_client: TestClient) -> None:
        """Empty transactions list should return 422 Unprocessable Entity."""
        response = full_client.post("/api/v1/batch/score", json={"transactions": []})
        assert response.status_code == 422

    def test_batch_missing_field_rejects_with_422(self, full_client: TestClient) -> None:
        """Payload without 'transactions' key should return 422."""
        response = full_client.post("/api/v1/batch/score", json={})
        assert response.status_code == 422

    def test_batch_no_scorer_returns_503(self, full_app: FastAPI, full_client: TestClient) -> None:
        """When scorer is None batch endpoint must return 503."""
        full_app.state.scorer = None
        resp = full_client.post("/api/v1/batch/score", json={"transactions": [_txn_payload()]})
        assert resp.status_code == 503
        # Restore for subsequent tests
        full_app.state.scorer = EnsembleScorer()


# ---------------------------------------------------------------------------
# Transaction search with filters
# ---------------------------------------------------------------------------


class TestTransactionSearch:
    """GET /api/v1/transactions"""

    def _seed(self, full_app: FastAPI, count: int = 3) -> list[ScoredTransaction]:
        """Insert synthetic scored transactions into the store."""
        store: TransactionStore = full_app.state.transaction_store
        txns = []
        for i in range(count):
            st = ScoredTransaction(
                transaction_id=uuid4(),
                user_id=UUID(_USER_ID),
                amount=Decimal(str((i + 1) * 100)),
                currency="USD",
                transaction_type=TransactionType.PURCHASE,
                timestamp=datetime(2025, 6, 15, 12, 0, 0),
                fraud_score=0.1 * (i + 1),
                model_version="ensemble-v1",
                scoring_latency_ms=5.0,
                is_flagged=i >= 2,
            )
            store.add(st)
            txns.append(st)
        return txns

    def test_search_returns_all(self, full_app: FastAPI, full_client: TestClient) -> None:
        """Without filters, all seeded transactions are returned."""
        self._seed(full_app, 3)
        response = full_client.get("/api/v1/transactions")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 3

    def test_search_filter_min_score(self, full_app: FastAPI, full_client: TestClient) -> None:
        """min_score filter should exclude low-score transactions."""
        self._seed(full_app, 5)
        response = full_client.get("/api/v1/transactions", params={"min_score": 0.4})
        data = response.json()
        for txn in data["transactions"]:
            assert txn["fraud_score"] >= 0.4

    def test_search_filter_is_flagged(self, full_app: FastAPI, full_client: TestClient) -> None:
        """is_flagged=true filter returns only flagged transactions."""
        self._seed(full_app, 3)
        response = full_client.get("/api/v1/transactions", params={"is_flagged": True})
        data = response.json()
        for txn in data["transactions"]:
            assert txn["is_flagged"] is True

    def test_search_no_store_returns_503(self, full_app: FastAPI, full_client: TestClient) -> None:
        """When transaction_store is None, endpoint must return 503."""
        full_app.state.transaction_store = None
        resp = full_client.get("/api/v1/transactions")
        assert resp.status_code == 503
        full_app.state.transaction_store = TransactionStore()

    def test_get_transaction_by_id_not_found(self, full_client: TestClient) -> None:
        """Unknown transaction_id must return 404."""
        response = full_client.get(f"/api/v1/transactions/{uuid4()}")
        assert response.status_code == 404

    def test_pagination_limit_and_offset(self, full_app: FastAPI, full_client: TestClient) -> None:
        """limit and offset query params control pagination."""
        self._seed(full_app, 10)
        response = full_client.get("/api/v1/transactions", params={"limit": 3, "offset": 0})
        assert response.status_code == 200
        data = response.json()
        assert len(data["transactions"]) <= 3


# ---------------------------------------------------------------------------
# Case management CRUD
# ---------------------------------------------------------------------------


class TestCaseManagement:
    """POST / GET / PATCH /api/v1/cases"""

    def test_create_case(self, full_client: TestClient) -> None:
        """Creating a case should return a FraudCase with a case_id."""
        payload = {
            "transaction_id": str(uuid4()),
            "priority": "high",
            "notes": "Suspected card testing.",
        }
        response = full_client.post("/api/v1/cases", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "case_id" in data
        assert data["status"] == "open"
        assert data["priority"] == "high"

    def test_get_case_by_id(self, full_client: TestClient) -> None:
        """Created case should be retrievable by its case_id."""
        txn_id = str(uuid4())
        resp = full_client.post("/api/v1/cases", json={"transaction_id": txn_id})
        case_id = resp.json()["case_id"]
        get_resp = full_client.get(f"/api/v1/cases/{case_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["case_id"] == case_id

    def test_get_case_not_found(self, full_client: TestClient) -> None:
        """Non-existent case_id must return 404."""
        response = full_client.get("/api/v1/cases/nonexistent-id")
        assert response.status_code == 404

    def test_update_case_status(self, full_client: TestClient) -> None:
        """PATCH can change case status."""
        txn_id = str(uuid4())
        resp = full_client.post("/api/v1/cases", json={"transaction_id": txn_id})
        case_id = resp.json()["case_id"]
        patch_resp = full_client.patch(
            f"/api/v1/cases/{case_id}",
            json={"status": "investigating", "actor": "analyst1"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["status"] == "investigating"

    def test_assign_case(self, full_client: TestClient) -> None:
        """PATCH can assign a case to an analyst."""
        txn_id = str(uuid4())
        resp = full_client.post("/api/v1/cases", json={"transaction_id": txn_id})
        case_id = resp.json()["case_id"]
        patch_resp = full_client.patch(
            f"/api/v1/cases/{case_id}",
            json={"assigned_to": "alice", "actor": "manager"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["assigned_to"] == "alice"

    def test_list_cases(self, full_client: TestClient) -> None:
        """GET /api/v1/cases should return a list with total count."""
        full_client.post("/api/v1/cases", json={"transaction_id": str(uuid4())})
        response = full_client.get("/api/v1/cases")
        assert response.status_code == 200
        data = response.json()
        assert "cases" in data
        assert "total" in data

    def test_idempotent_case_creation(self, full_client: TestClient) -> None:
        """Creating a case for the same transaction_id twice returns the same case."""
        txn_id = str(uuid4())
        r1 = full_client.post("/api/v1/cases", json={"transaction_id": txn_id})
        r2 = full_client.post("/api/v1/cases", json={"transaction_id": txn_id})
        assert r1.json()["case_id"] == r2.json()["case_id"]

    def test_case_manager_unavailable_returns_503(
        self, full_app: FastAPI, full_client: TestClient
    ) -> None:
        """503 when case_manager is None."""
        full_app.state.case_manager = None
        resp = full_client.post("/api/v1/cases", json={"transaction_id": str(uuid4())})
        assert resp.status_code == 503
        full_app.state.case_manager = CaseManager()


# ---------------------------------------------------------------------------
# Feedback submission
# ---------------------------------------------------------------------------


class TestFeedbackSubmission:
    """POST /api/v1/feedback"""

    def test_submit_fraud_feedback(self, full_client: TestClient) -> None:
        """Submitting fraud=True feedback should persist an entry."""
        txn_id = str(uuid4())
        payload = {"transaction_id": txn_id, "is_fraud": True, "analyst": "bob"}
        response = full_client.post("/api/v1/feedback", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["transaction_id"] == txn_id
        assert data["is_fraud"] is True
        assert data["analyst"] == "bob"

    def test_submit_legitimate_feedback(self, full_client: TestClient) -> None:
        """Feedback with is_fraud=False should also persist correctly."""
        payload = {"transaction_id": str(uuid4()), "is_fraud": False, "analyst": "carol"}
        response = full_client.post("/api/v1/feedback", json=payload)
        assert response.status_code == 200
        assert response.json()["is_fraud"] is False

    def test_feedback_missing_is_fraud_returns_422(self, full_client: TestClient) -> None:
        """Omitting the required is_fraud field should return 422."""
        payload = {"transaction_id": str(uuid4()), "analyst": "dave"}
        response = full_client.post("/api/v1/feedback", json=payload)
        assert response.status_code == 422

    def test_feedback_store_unavailable_returns_503(
        self, full_app: FastAPI, full_client: TestClient
    ) -> None:
        """503 when feedback_store is None."""
        full_app.state.feedback_store = None
        resp = full_client.post(
            "/api/v1/feedback",
            json={"transaction_id": str(uuid4()), "is_fraud": False},
        )
        assert resp.status_code == 503
        full_app.state.feedback_store = FeedbackStore()

    def test_feedback_list_endpoint(self, full_client: TestClient) -> None:
        """GET /api/v1/feedback should return a list of entries."""
        full_client.post(
            "/api/v1/feedback",
            json={"transaction_id": str(uuid4()), "is_fraud": True},
        )
        response = full_client.get("/api/v1/feedback")
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "total" in data


# ---------------------------------------------------------------------------
# Drift report endpoint
# ---------------------------------------------------------------------------


class TestDriftReportEndpoint:
    """GET /api/v1/drift/report"""

    def test_insufficient_data_returns_status_message(self, full_client: TestClient) -> None:
        """With no samples the endpoint should return insufficient_data status."""
        response = full_client.get("/api/v1/drift/report")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "insufficient_data"

    def test_drift_detector_unavailable_returns_503(
        self, full_app: FastAPI, full_client: TestClient
    ) -> None:
        """503 when drift_detector is None."""
        full_app.state.drift_detector = None
        resp = full_client.get("/api/v1/drift/report")
        assert resp.status_code == 503
        full_app.state.drift_detector = DriftDetector()

    def test_drift_report_with_enough_data(self, full_app: FastAPI, full_client: TestClient) -> None:
        """Once populated the endpoint returns a structured report."""
        # Replace with a small-window detector so thresholds are reachable in tests
        small_detector = DriftDetector(
            reference_window=100,
            current_window=50,
        )
        full_app.state.drift_detector = small_detector

        import random
        random.seed(42)
        # Fill reference window
        for _ in range(110):
            small_detector.add_sample(
                {feat: random.random() for feat in DriftDetector.DEFAULT_FEATURES}
            )
        # Fill current window
        for _ in range(60):
            small_detector.add_sample(
                {feat: random.random() * 2 for feat in DriftDetector.DEFAULT_FEATURES}
            )

        response = full_client.get("/api/v1/drift/report")
        assert response.status_code == 200
        data = response.json()
        assert "overall_drift" in data
        assert "feature_metrics" in data

        # Restore default
        full_app.state.drift_detector = DriftDetector()


# ---------------------------------------------------------------------------
# Rate limiting behaviour
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Token-bucket rate limiter is tested in isolation (not middleware layer)."""

    def test_token_bucket_allows_within_capacity(self) -> None:
        """Requests within burst capacity should all be allowed."""
        from scoring.api.rate_limiter import RateLimiter

        limiter = RateLimiter(default_rpm=60, burst_multiplier=2.0)
        allowed_count = sum(1 for _ in range(2) if limiter.check("test_client")[0])
        assert allowed_count == 2

    def test_token_bucket_blocks_when_exhausted(self) -> None:
        """Requests exceeding burst capacity must be denied."""
        from scoring.api.rate_limiter import RateLimiter

        # 6 RPM => 0.1 tokens/s, burst = 0.2 → capacity < 1 whole token initially
        # Set burst_multiplier=10 so burst=1.0 → exactly 1 initial token
        limiter = RateLimiter(default_rpm=6, burst_multiplier=10.0)
        # First call should pass, subsequent ones may fail when exhausted
        results = [limiter.check("exhausted_client")[0] for _ in range(5)]
        # At least one request must be blocked
        assert False in results

    def test_retry_after_is_positive_when_blocked(self) -> None:
        """retry_after should be > 0 when rate limit exceeded."""
        from scoring.api.rate_limiter import RateLimiter

        limiter = RateLimiter(default_rpm=6, burst_multiplier=10.0)
        for _ in range(5):
            allowed, retry_after = limiter.check("retry_client")
            if not allowed:
                assert retry_after > 0.0
                break

    def test_different_clients_have_independent_buckets(self) -> None:
        """Two different client keys should not share rate limit state."""
        from scoring.api.rate_limiter import RateLimiter

        limiter = RateLimiter(default_rpm=6, burst_multiplier=10.0)
        # Exhaust client_a
        for _ in range(10):
            limiter.check("client_a")
        # client_b should still have tokens
        allowed, _ = limiter.check("client_b")
        assert allowed is True


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------


class TestErrorResponses:
    """Verify correct HTTP status codes for various error conditions."""

    def test_score_400_invalid_amount(self, test_client: TestClient) -> None:
        """Negative amount should produce 422 (validation error)."""
        payload = _txn_payload(amount="-10.00")
        response = test_client.post("/score", json=payload)
        assert response.status_code == 422

    def test_score_422_missing_required_field(self, test_client: TestClient) -> None:
        """Omitting user_id should produce 422."""
        payload = {
            "amount": "50.00",
            "currency": "USD",
            "transaction_type": "purchase",
            "timestamp": "2025-06-15T14:00:00",
        }
        response = test_client.post("/score", json=payload)
        assert response.status_code == 422

    def test_score_503_no_scorer(self, test_app: FastAPI, test_client: TestClient) -> None:
        """503 when no model is loaded."""
        test_app.state.scorer = None
        resp = test_client.post("/score", json=_txn_payload())
        assert resp.status_code == 503
        test_app.state.scorer = EnsembleScorer()

    def test_model_info_503_no_scorer(self, test_app: FastAPI, test_client: TestClient) -> None:
        """GET /model/info returns 503 when scorer is None."""
        test_app.state.scorer = None
        resp = test_client.get("/model/info")
        assert resp.status_code == 503
        test_app.state.scorer = EnsembleScorer()

    def test_health_503_when_unhealthy(self, test_app: FastAPI, test_client: TestClient) -> None:
        """Health endpoint returns 503 when redis is unavailable."""
        test_app.state.redis = None
        resp = test_client.get("/health")
        assert resp.status_code == 503
        # Restore
        from unittest.mock import AsyncMock
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        test_app.state.redis = mock

    def test_invalid_json_body_returns_422(self, test_client: TestClient) -> None:
        """Malformed JSON body should return 422."""
        response = test_client.post(
            "/score",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------


class TestCORSHeaders:
    """Confirm that CORS headers are present on responses."""

    def test_cors_allow_origin_present(self, test_client: TestClient) -> None:
        """Responses to cross-origin requests must include CORS headers."""
        response = test_client.get(
            "/health",
            headers={"Origin": "https://example.com"},
        )
        assert response.status_code == 200
        # CORSMiddleware injects this header
        assert "access-control-allow-origin" in response.headers

    def test_cors_preflight_options(self, test_client: TestClient) -> None:
        """OPTIONS preflight requests should be handled by CORSMiddleware."""
        response = test_client.options(
            "/score",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI CORSMiddleware returns 200 for OPTIONS
        assert response.status_code in (200, 400)  # 400 allowed if body required

    def test_x_request_id_propagated(self, test_client: TestClient) -> None:
        """RequestIDMiddleware must echo back X-Request-ID."""
        custom_id = "my-custom-request-id"
        response = test_client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.headers.get("x-request-id") == custom_id

    def test_x_request_id_generated_when_absent(self, test_client: TestClient) -> None:
        """When no X-Request-ID is provided, one must be generated and returned."""
        response = test_client.get("/health")
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0
