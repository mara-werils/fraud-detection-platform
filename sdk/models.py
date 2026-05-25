"""Data models for the Fraud Detection SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoringResult:
    """Result from scoring a single transaction."""

    transaction_id: str
    user_id: str
    fraud_score: float
    is_flagged: bool
    decision: str
    flag_reason: str | None
    model_version: str
    scoring_latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> ScoringResult:
        decision = "ALLOW"
        if data.get("fraud_score", 0) >= 0.8:
            decision = "BLOCK"
        elif data.get("fraud_score", 0) >= 0.5:
            decision = "REVIEW"

        return cls(
            transaction_id=data.get("transaction_id", ""),
            user_id=data.get("user_id", ""),
            fraud_score=data.get("fraud_score", 0.0),
            is_flagged=data.get("is_flagged", False),
            decision=decision,
            flag_reason=data.get("flag_reason"),
            model_version=data.get("model_version", ""),
            scoring_latency_ms=data.get("scoring_latency_ms", 0.0),
            raw=data,
        )


@dataclass
class BatchResult:
    """Result from batch scoring."""

    results: list[ScoringResult]
    total: int
    failed: int
    latency_ms: float
    errors: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> BatchResult:
        return cls(
            results=[ScoringResult.from_response(r) for r in data.get("results", [])],
            total=data.get("total", 0),
            failed=data.get("failed", 0),
            latency_ms=data.get("latency_ms", 0.0),
            errors=data.get("errors", []),
        )


@dataclass
class SearchResult:
    """Result from searching transactions."""

    transactions: list[ScoringResult]
    total: int
    has_more: bool

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> SearchResult:
        return cls(
            transactions=[
                ScoringResult.from_response(t) for t in data.get("transactions", [])
            ],
            total=data.get("total", 0),
            has_more=data.get("has_more", False),
        )


@dataclass
class CaseInfo:
    """Fraud case information."""

    case_id: str
    transaction_id: str
    user_id: str
    status: str
    priority: str
    assigned_to: str | None
    created_at: str
    notes: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> CaseInfo:
        return cls(
            case_id=data.get("case_id", ""),
            transaction_id=data.get("transaction_id", ""),
            user_id=data.get("user_id", ""),
            status=data.get("status", ""),
            priority=data.get("priority", ""),
            assigned_to=data.get("assigned_to"),
            created_at=data.get("created_at", ""),
            notes=data.get("notes", []),
            raw=data,
        )
