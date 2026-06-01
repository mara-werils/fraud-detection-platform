"""Data export API for compliance reporting and analytics."""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/export", tags=["export"])


class ExportRequest(BaseModel):
    """Request parameters for data export."""

    format: str = Field(default="csv", description="Export format: csv or json")
    start_time: datetime | None = Field(default=None, description="Start of time range")
    end_time: datetime | None = Field(default=None, description="End of time range")
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_score: float | None = Field(default=None, ge=0.0, le=1.0)
    is_flagged: bool | None = Field(default=None)
    user_id: str | None = Field(default=None)
    include_features: bool = Field(default=False, description="Include feature vectors")
    limit: int = Field(default=10000, ge=1, le=100000)


@router.post("/transactions")
async def export_transactions(req: ExportRequest, request: Request) -> Any:
    """Export scored transactions for compliance reporting.

    Supports CSV and JSON formats with configurable filtering.
    CSV exports stream directly for large datasets.
    """
    store = getattr(request.app.state, "transaction_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Transaction store not available")

    transactions, total = store.search(
        user_id=req.user_id,
        min_score=req.min_score,
        max_score=req.max_score,
        is_flagged=req.is_flagged,
        start_time=req.start_time,
        end_time=req.end_time,
        limit=req.limit,
        offset=0,
    )

    if req.format == "csv":
        return _generate_csv_response(transactions, req.include_features)
    else:
        return _generate_json_response(transactions, total, req.include_features)


@router.post("/cases")
async def export_cases(request: Request, format: str = Query("csv")) -> Any:
    """Export fraud cases for compliance and audit reporting."""
    case_mgr = getattr(request.app.state, "case_manager", None)
    if case_mgr is None:
        raise HTTPException(status_code=503, detail="Case manager not available")

    cases, total = case_mgr.list_cases(limit=10000)

    if format == "csv":
        return _generate_cases_csv(cases)
    else:
        return {
            "total": total,
            "exported_at": datetime.now(UTC).isoformat(),
            "cases": [c.model_dump(mode="json") for c in cases],
        }


@router.post("/feedback")
async def export_feedback(request: Request, format: str = Query("csv")) -> Any:
    """Export analyst feedback for model retraining and audit."""
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Feedback store not available")

    entries, total = store.list_entries(limit=10000)

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "feedback_id", "transaction_id", "is_fraud", "analyst",
            "original_score", "original_decision", "notes", "created_at",
        ])

        for e in entries:
            writer.writerow([
                e.feedback_id, e.transaction_id, e.is_fraud, e.analyst,
                e.original_score, e.original_decision, e.notes or "",
                e.created_at.isoformat(),
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=feedback_export.csv"},
        )
    else:
        return {
            "total": total,
            "exported_at": datetime.now(UTC).isoformat(),
            "data": [e.model_dump(mode="json") for e in entries],
        }


def _generate_csv_response(
    transactions: list[Any],
    include_features: bool,
) -> StreamingResponse:
    """Generate a streaming CSV response."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    headers = [
        "transaction_id", "user_id", "amount", "currency",
        "transaction_type", "timestamp", "fraud_score",
        "is_flagged", "flag_reason", "model_version",
        "scoring_latency_ms", "scored_at",
    ]

    if include_features:
        headers.extend([
            "txn_count_1h", "txn_count_24h", "amount_zscore",
            "amount_to_avg_ratio", "distance_from_last_txn_km",
            "is_new_device", "is_new_merchant", "unique_countries_24h",
        ])

    writer.writerow(headers)

    for txn in transactions:
        row = [
            str(txn.transaction_id),
            str(txn.user_id),
            str(txn.amount),
            txn.currency,
            txn.transaction_type,
            txn.timestamp.isoformat(),
            txn.fraud_score,
            txn.is_flagged,
            txn.flag_reason or "",
            txn.model_version,
            txn.scoring_latency_ms,
            txn.scored_at.isoformat(),
        ]

        if include_features and txn.features:
            f = txn.features
            row.extend([
                f.txn_count_1h, f.txn_count_24h, f.amount_zscore,
                f.amount_to_avg_ratio, f.distance_from_last_txn_km,
                f.is_new_device, f.is_new_merchant, f.unique_countries_24h,
            ])
        elif include_features:
            row.extend([""] * 8)

        writer.writerow(row)

    output.seek(0)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=transactions_{timestamp}.csv"
        },
    )


def _generate_json_response(
    transactions: list[Any],
    total: int,
    include_features: bool,
) -> dict[str, Any]:
    """Generate a JSON export response."""
    data = []
    for txn in transactions:
        entry: dict[str, Any] = {
            "transaction_id": str(txn.transaction_id),
            "user_id": str(txn.user_id),
            "amount": str(txn.amount),
            "currency": txn.currency,
            "transaction_type": txn.transaction_type,
            "timestamp": txn.timestamp.isoformat(),
            "fraud_score": txn.fraud_score,
            "is_flagged": txn.is_flagged,
            "flag_reason": txn.flag_reason,
            "model_version": txn.model_version,
            "scoring_latency_ms": txn.scoring_latency_ms,
            "scored_at": txn.scored_at.isoformat(),
        }

        if include_features and txn.features:
            entry["features"] = txn.features.model_dump(mode="json")

        data.append(entry)

    return {
        "total": total,
        "exported": len(data),
        "exported_at": datetime.now(UTC).isoformat(),
        "data": data,
    }


def _generate_cases_csv(cases: list[Any]) -> StreamingResponse:
    """Generate CSV for cases export."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "case_id", "transaction_id", "user_id", "fraud_score",
        "amount", "currency", "status", "priority",
        "assigned_to", "is_fraud", "created_at", "resolved_at",
        "notes_count", "events_count", "tags",
    ])

    for case in cases:
        writer.writerow([
            case.case_id, case.transaction_id, case.user_id,
            case.fraud_score, case.amount, case.currency,
            case.status.value, case.priority.value,
            case.assigned_to or "", case.is_fraud,
            case.created_at.isoformat(),
            case.resolved_at.isoformat() if case.resolved_at else "",
            len(case.notes), len(case.events),
            ",".join(case.tags),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cases_export.csv"},
    )
