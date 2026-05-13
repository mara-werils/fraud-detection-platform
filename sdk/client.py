"""Fraud Detection Platform Python SDK client."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import httpx

from sdk.models import BatchResult, CaseInfo, ScoringResult, SearchResult


class FraudClientError(Exception):
    """Base exception for SDK errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FraudClient:
    """Python SDK for the Fraud Detection Platform.

    Provides a clean interface for scoring transactions, searching results,
    managing cases, and interacting with the platform API.

    Example:
        >>> client = FraudClient("http://localhost:8000", api_key="fdp_...")
        >>> result = client.score(
        ...     user_id="user-001",
        ...     amount=45000.00,
        ...     merchant_id="merch-001",
        ... )
        >>> print(f"Score: {result.fraud_score}, Decision: {result.decision}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> FraudClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
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
            "timestamp": datetime.utcnow().isoformat(),
        }

        if merchant_id:
            payload["merchant_id"] = merchant_id
        if category:
            payload["merchant_category"] = category
        if channel:
            payload["metadata"] = {"channel": channel, **(metadata or {})}
        if ip_address:
            payload["ip_address"] = ip_address
        if device_id:
            payload["device_id"] = device_id
        if latitude is not None:
            payload["latitude"] = latitude
        if longitude is not None:
            payload["longitude"] = longitude

        resp = self._client.post("/score", json=payload)
        self._check_response(resp)
        return ScoringResult.from_response(resp.json())

    def batch_score(
        self,
        transactions: list[dict[str, Any]],
    ) -> BatchResult:
        """Score multiple transactions in a single request.

        Args:
            transactions: List of transaction dicts with fields matching
                the score() parameters.

        Returns:
            BatchResult with all scored transactions.
        """
        resp = self._client.post(
            "/api/v1/batch/score",
            json={"transactions": transactions},
        )
        self._check_response(resp)
        return BatchResult.from_response(resp.json())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_transactions(
        self,
        user_id: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        is_flagged: bool | None = None,
        decision: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SearchResult:
        """Search scored transactions.

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

        resp = self._client.get("/api/v1/transactions", params=params)
        self._check_response(resp)
        return SearchResult.from_response(resp.json())

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------

    def create_case(
        self,
        transaction_id: str,
        priority: str = "medium",
        assigned_to: str | None = None,
        notes: str | None = None,
    ) -> CaseInfo:
        """Create a fraud investigation case.

        Args:
            transaction_id: ID of the suspicious transaction.
            priority: Case priority (low/medium/high/critical).
            assigned_to: Analyst to assign the case to.
            notes: Initial investigation notes.

        Returns:
            Created CaseInfo.
        """
        payload: dict[str, Any] = {
            "transaction_id": transaction_id,
            "priority": priority,
        }
        if assigned_to:
            payload["assigned_to"] = assigned_to
        if notes:
            payload["notes"] = notes

        resp = self._client.post("/api/v1/cases", json=payload)
        self._check_response(resp)
        return CaseInfo.from_response(resp.json())

    def list_cases(
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

        resp = self._client.get("/api/v1/cases", params=params)
        self._check_response(resp)
        data = resp.json()
        return [CaseInfo.from_response(c) for c in data.get("cases", [])]

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def submit_feedback(
        self,
        transaction_id: str,
        is_fraud: bool,
        analyst: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Submit analyst feedback on a scored transaction.

        Args:
            transaction_id: ID of the transaction.
            is_fraud: Whether the transaction was actually fraudulent.
            analyst: Name of the analyst submitting feedback.
            notes: Additional notes.

        Returns:
            Confirmation response.
        """
        payload: dict[str, Any] = {
            "transaction_id": transaction_id,
            "is_fraud": is_fraud,
        }
        if analyst:
            payload["analyst"] = analyst
        if notes:
            payload["notes"] = notes

        resp = self._client.post("/api/v1/feedback", json=payload)
        self._check_response(resp)
        return resp.json()

    # ------------------------------------------------------------------
    # Health & Info
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Check service health."""
        resp = self._client.get("/health")
        return resp.json()

    def model_info(self) -> dict[str, Any]:
        """Get current model information."""
        resp = self._client.get("/model/info")
        self._check_response(resp)
        return resp.json()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_response(self, resp: httpx.Response) -> None:
        """Raise FraudClientError for non-2xx responses."""
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise FraudClientError(
                f"API error {resp.status_code}: {detail}",
                status_code=resp.status_code,
            )
