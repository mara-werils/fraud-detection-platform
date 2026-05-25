"""Blocklist/allowlist/watchlist management API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from scoring.services.entity_lists import (
    EntityEntry,
    EntityListService,
    EntityType,
    ImportResult,
    ListMatchResult,
    ListType,
)

router = APIRouter(prefix="/api/v1/lists", tags=["entity-lists"])


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------


class RemoveEntryRequest(BaseModel):
    """Body for the DELETE /entries endpoint."""

    entity_type: EntityType
    entity_value: str = Field(..., min_length=1)
    list_type: ListType


class CheckEntityRequest(BaseModel):
    """Body for the POST /check endpoint."""

    entity_type: EntityType
    entity_value: str = Field(..., min_length=1)


class ImportRequest(BaseModel):
    """Body for the POST /import endpoint."""

    entries: list[EntityEntry] = Field(..., min_length=1, max_length=10_000)


# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------


def _get_service(request: Request) -> EntityListService:
    service: EntityListService | None = getattr(
        request.app.state, "entity_list_service", None
    )
    if service is None:
        raise HTTPException(status_code=503, detail="Entity list service not available")
    return service


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/entries",
    response_model=EntityEntry,
    status_code=201,
    summary="Add an entity to a list",
)
async def add_entry(entry: EntityEntry, request: Request) -> EntityEntry:
    """Add or overwrite a single entity entry on a blocklist, allowlist, or watchlist.

    Supplying an entry with the same (list_type, entity_type, entity_value) as an
    existing entry will replace it atomically.
    """
    service = _get_service(request)
    stored = service.add_entry(entry)
    return stored


@router.delete(
    "/entries",
    summary="Remove an entity from a list",
)
async def remove_entry(body: RemoveEntryRequest, request: Request) -> dict[str, Any]:
    """Remove an entity entry from the specified list.

    Returns 404 when no matching entry is found.
    """
    service = _get_service(request)
    removed = service.remove_entry(
        entity_type=body.entity_type,
        entity_value=body.entity_value,
        list_type=body.list_type,
    )
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No {body.list_type} entry found for "
                f"{body.entity_type}:{body.entity_value}"
            ),
        )
    return {"removed": True, "entity_type": body.entity_type, "entity_value": body.entity_value}


@router.post(
    "/check",
    response_model=ListMatchResult,
    summary="Check whether an entity appears on any list",
)
async def check_entity(body: CheckEntityRequest, request: Request) -> ListMatchResult:
    """Check a single entity value against blocklist, allowlist, and watchlist.

    Blocklist hits are returned first.  Returns `matched: false` when no active
    entry is found.
    """
    service = _get_service(request)
    result = service.check(entity_type=body.entity_type, entity_value=body.entity_value)
    return result


@router.get(
    "/entries",
    response_model=list[EntityEntry],
    summary="List entries with optional filters",
)
async def list_entries(
    request: Request,
    list_type: ListType = Query(..., description="Which list to query"),
    entity_type: EntityType | None = Query(None, description="Filter by entity type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum entries to return"),
) -> list[EntityEntry]:
    """Return active entries for the given list type, newest first.

    Optionally filter by entity type.  Expired entries are excluded.
    """
    service = _get_service(request)
    return service.get_entries(list_type=list_type, entity_type=entity_type, limit=limit)


@router.post(
    "/import",
    response_model=ImportResult,
    summary="Bulk import entity entries",
)
async def import_entries(body: ImportRequest, request: Request) -> ImportResult:
    """Bulk-import up to 10 000 entity entries in a single request.

    Duplicate (list_type, entity_type, entity_value) keys are overwritten.
    The response reports how many entries were imported, skipped, and any
    error messages for individual failures.
    """
    service = _get_service(request)
    result = service.import_entries(body.entries)
    return result


@router.get(
    "/stats",
    summary="Entity list statistics",
)
async def list_stats(request: Request) -> dict[str, Any]:
    """Return a breakdown of active and expired entries by list type and entity type."""
    service = _get_service(request)
    return service.get_stats()
