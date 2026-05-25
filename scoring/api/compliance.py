"""FastAPI router for compliance reporting — PCI-DSS, SOX, GDPR, AML/KYC.

Exposes endpoints for generating reports, retrieving scores, listing check
results, running on-demand checks, and fetching remediation plans.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from scoring.services.compliance import (
    AuditSummary,
    ComplianceCheck,
    ComplianceReport,
    ComplianceService,
    ComplianceStandard,
    RemediationItem,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])

# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------


class ScoreResponse(BaseModel):
    """Compliance score response for one or all standards."""

    scores: dict[str, float]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunCheckResponse(BaseModel):
    """Result of an on-demand compliance check run."""

    standard: str
    checks_run: int
    compliance_score: float
    overall_status: str
    checks: list[ComplianceCheck]
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScheduleRequest(BaseModel):
    """Request body to schedule automated compliance checks."""

    cron_expression: str = Field(
        ...,
        examples=["0 6 * * 1"],
        description="5-field cron expression (min hr dom mon dow)",
    )


# ---------------------------------------------------------------------------
# Dependency: resolve ComplianceService from app state
# ---------------------------------------------------------------------------


def _get_service(request: Request) -> ComplianceService:
    svc: ComplianceService | None = getattr(request.app.state, "compliance_service", None)
    if svc is None:
        # Fall back to a fresh instance — stateless for scoring/checks.
        svc = ComplianceService()
    return svc


def _parse_standard(standard: str) -> ComplianceStandard:
    try:
        return ComplianceStandard(standard.upper())
    except ValueError:
        valid = [s.value for s in ComplianceStandard]
        raise HTTPException(
            status_code=422,
            detail=f"Unknown compliance standard '{standard}'. Valid values: {valid}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/report/{standard}",
    response_model=ComplianceReport,
    summary="Generate a compliance report",
    description=(
        "Run all compliance checks for the given standard over the specified time window "
        "and return a full report including score, check details, and summary."
    ),
)
async def get_compliance_report(
    standard: str,
    request: Request,
    period_days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Report window in days ending now",
    ),
) -> ComplianceReport:
    """Generate and return a compliance report for the specified standard."""
    std = _parse_standard(standard)
    svc = _get_service(request)
    now = datetime.now(timezone.utc)
    period_start = now - timedelta(days=period_days)

    logger.info("api_compliance_report", standard=std.value, period_days=period_days)
    try:
        report = svc.generate_report(std, period_start=period_start, period_end=now)
    except Exception as exc:
        logger.exception("compliance_report_failed", standard=std.value, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to generate compliance report") from exc

    return report


@router.get(
    "/score",
    response_model=ScoreResponse,
    summary="Get compliance scores",
    description=(
        "Return the current weighted compliance score (0–100) for one or all standards. "
        "Pass ?standard=PCI_DSS to filter to a single standard."
    ),
)
async def get_compliance_scores(
    request: Request,
    standard: str | None = Query(
        default=None,
        description="Filter to a specific standard (PCI_DSS, SOX, GDPR, AML_KYC)",
    ),
) -> ScoreResponse:
    """Return compliance score(s). Omit ?standard to get all standards."""
    svc = _get_service(request)

    if standard:
        std = _parse_standard(standard)
        standards = [std]
    else:
        standards = list(ComplianceStandard)

    scores: dict[str, float] = {}
    for std in standards:
        try:
            scores[std.value] = svc.get_compliance_score(std)
        except Exception as exc:
            logger.warning("score_fetch_failed", standard=std.value, error=str(exc))
            scores[std.value] = -1.0  # sentinel for unavailable

    return ScoreResponse(scores=scores)


@router.get(
    "/checks/{standard}",
    response_model=list[ComplianceCheck],
    summary="List compliance checks for a standard",
    description=(
        "Execute and return the full list of individual compliance check results "
        "for the given standard. Optionally filter by status."
    ),
)
async def list_compliance_checks(
    standard: str,
    request: Request,
    status_filter: str | None = Query(
        default=None,
        alias="status",
        description="Filter results by status: COMPLIANT, NON_COMPLIANT, PARTIAL, UNKNOWN",
    ),
) -> list[ComplianceCheck]:
    """List all compliance check results for the requested standard."""
    std = _parse_standard(standard)
    svc = _get_service(request)

    logger.info("api_list_checks", standard=std.value, status_filter=status_filter)
    try:
        checks = svc._run_checks_for(std)  # noqa: SLF001
    except Exception as exc:
        logger.exception("checks_failed", standard=std.value, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to run compliance checks") from exc

    if status_filter:
        status_filter_upper = status_filter.upper()
        checks = [c for c in checks if c.status.value == status_filter_upper]

    return checks


@router.post(
    "/run/{standard}",
    response_model=RunCheckResponse,
    summary="Trigger on-demand compliance checks",
    description=(
        "Immediately execute all compliance checks for the specified standard "
        "and return results with an updated compliance score."
    ),
)
async def run_compliance_checks(
    standard: str,
    request: Request,
) -> RunCheckResponse:
    """Trigger an on-demand compliance check run for the given standard."""
    std = _parse_standard(standard)
    svc = _get_service(request)

    logger.info("api_run_checks", standard=std.value)
    try:
        checks = svc._run_checks_for(std)  # noqa: SLF001
        score = svc._score(checks)  # noqa: SLF001
        overall = svc._overall_status(score)  # noqa: SLF001
    except Exception as exc:
        logger.exception("run_checks_failed", standard=std.value, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to execute compliance checks") from exc

    return RunCheckResponse(
        standard=std.value,
        checks_run=len(checks),
        compliance_score=score,
        overall_status=overall.value,
        checks=checks,
    )


@router.get(
    "/remediation/{standard}",
    response_model=list[RemediationItem],
    summary="Get remediation plan for a standard",
    description=(
        "Return a prioritised remediation plan for all failing or partial compliance "
        "checks under the specified standard, including effort estimates and owners."
    ),
)
async def get_remediation_plan(
    standard: str,
    request: Request,
) -> list[RemediationItem]:
    """Return a prioritised remediation plan for non-compliant/partial checks."""
    std = _parse_standard(standard)
    svc = _get_service(request)

    logger.info("api_remediation_plan", standard=std.value)
    try:
        plan = svc.get_remediation_plan(std)
    except Exception as exc:
        logger.exception("remediation_plan_failed", standard=std.value, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to generate remediation plan") from exc

    return plan


@router.get(
    "/audit-summary",
    response_model=AuditSummary,
    summary="Get audit activity summary",
    description=(
        "Return an aggregated audit activity summary for the given time window, "
        "including event counts by category, policy violations, and high-risk events."
    ),
)
async def get_audit_summary(
    request: Request,
    period_days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Look-back window in days",
    ),
) -> AuditSummary:
    """Return an aggregated audit activity summary."""
    svc = _get_service(request)
    now = datetime.now(timezone.utc)
    period_start = now - timedelta(days=period_days)

    logger.info("api_audit_summary", period_days=period_days)
    try:
        return svc.get_audit_summary(period_start=period_start, period_end=now)
    except Exception as exc:
        logger.exception("audit_summary_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to generate audit summary") from exc


@router.post(
    "/schedule/{standard}",
    summary="Schedule automated compliance checks",
    description=(
        "Register a cron schedule for automated compliance checks on the given standard. "
        "Cron expression uses 5-field format: min hr dom mon dow."
    ),
)
async def schedule_compliance_checks(
    standard: str,
    body: ScheduleRequest,
    request: Request,
) -> dict[str, Any]:
    """Register a cron schedule for the specified compliance standard."""
    std = _parse_standard(standard)
    svc = _get_service(request)

    try:
        svc.schedule_check(std, body.cron_expression)
    except Exception as exc:
        logger.exception("schedule_failed", standard=std.value, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to schedule compliance checks") from exc

    return {
        "standard": std.value,
        "cron_expression": body.cron_expression,
        "scheduled_at": datetime.now(timezone.utc).isoformat(),
        "message": f"Compliance checks for {std.value} scheduled with cron '{body.cron_expression}'",
    }
