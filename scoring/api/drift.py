"""Drift detection API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/api/v1/drift", tags=["drift"])


@router.get("/report")
async def drift_report(request: Request) -> dict[str, Any]:
    """Get the latest drift detection report.

    Returns PSI and KS statistics for all monitored features,
    indicating whether model retraining may be needed.
    """
    detector = getattr(request.app.state, "drift_detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not available")

    report = detector.check_drift()
    if report is None:
        return {
            "status": "insufficient_data",
            "message": "Not enough data for drift analysis. Need at least 100 reference and 50 current samples.",
        }

    return report.to_dict()


@router.get("/history")
async def drift_history(
    request: Request,
    limit: int = Query(10, ge=1, le=100, description="Number of reports"),
) -> dict[str, Any]:
    """Get historical drift reports."""
    detector = getattr(request.app.state, "drift_detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not available")

    reports = detector.get_reports(limit=limit)
    return {
        "total": len(reports),
        "reports": [r.to_dict() for r in reports],
    }
