"""Sanctions screening API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/sanctions", tags=["sanctions"])


class ScreenRequest(BaseModel):
    """Request to screen a name against sanctions lists."""

    name: str = Field(..., min_length=2, description="Name to screen")
    check_types: list[str] = Field(
        default=["sanctions", "pep", "watchlist"],
        description="List types to check",
    )


@router.post("/check")
async def check_sanctions(req: ScreenRequest, request: Request) -> dict[str, Any]:
    """Screen a name against sanctions, PEP, and watchlist databases.

    Returns any matches found with match type (exact/fuzzy/partial),
    match score, source list, and risk level assessment.
    """
    screener = getattr(request.app.state, "sanctions_screener", None)
    if screener is None:
        raise HTTPException(status_code=503, detail="Sanctions screener not available")

    result = screener.screen(req.name, check_types=req.check_types)
    return result.to_dict()


@router.get("/stats")
async def sanctions_stats(request: Request) -> dict[str, Any]:
    """Get sanctions screening statistics."""
    screener = getattr(request.app.state, "sanctions_screener", None)
    if screener is None:
        raise HTTPException(status_code=503, detail="Sanctions screener not available")

    return screener.stats
