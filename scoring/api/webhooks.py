"""Webhook management API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from scoring.services.webhook_manager import WebhookConfig

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


class RegisterWebhookRequest(BaseModel):
    """Request to register a new webhook."""

    url: str = Field(..., description="Webhook delivery URL")
    events: list[str] = Field(
        default=["transaction.flagged"],
        description="Events to subscribe to",
    )
    description: str = Field(default="", description="Webhook description")


class UpdateWebhookRequest(BaseModel):
    """Request to update a webhook."""

    url: str | None = None
    events: list[str] | None = None
    status: str | None = Field(None, description="active/paused")


@router.get("")
async def list_webhooks(request: Request) -> dict[str, Any]:
    """List all registered webhooks."""
    mgr = getattr(request.app.state, "webhook_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Webhook manager not available")

    webhooks = mgr.list_webhooks()
    return {
        "webhooks": [wh.model_dump() for wh in webhooks],
        "total": len(webhooks),
        "available_events": [
            "transaction.flagged",
            "transaction.blocked",
            "case.created",
            "case.resolved",
            "drift.detected",
            "model.updated",
        ],
    }


@router.post("", response_model=WebhookConfig)
async def register_webhook(req: RegisterWebhookRequest, request: Request) -> WebhookConfig:
    """Register a new webhook endpoint.

    Returns the webhook configuration including the signing secret.
    Store the secret securely — it's only shown once.
    """
    mgr = getattr(request.app.state, "webhook_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Webhook manager not available")

    webhook = mgr.register(
        url=req.url,
        events=req.events,
        description=req.description,
    )
    return webhook


@router.get("/{webhook_id}")
async def get_webhook(webhook_id: str, request: Request) -> dict[str, Any]:
    """Get a webhook by ID."""
    mgr = getattr(request.app.state, "webhook_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Webhook manager not available")

    webhook = mgr.get(webhook_id)
    if webhook is None:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return webhook.model_dump()


@router.patch("/{webhook_id}", response_model=WebhookConfig)
async def update_webhook(
    webhook_id: str, req: UpdateWebhookRequest, request: Request
) -> WebhookConfig:
    """Update a webhook configuration."""
    mgr = getattr(request.app.state, "webhook_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Webhook manager not available")

    webhook = mgr.update(
        webhook_id,
        url=req.url,
        events=req.events,
        status=req.status,
    )
    if webhook is None:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return webhook


@router.delete("/{webhook_id}")
async def delete_webhook(webhook_id: str, request: Request) -> dict[str, str]:
    """Delete a webhook registration."""
    mgr = getattr(request.app.state, "webhook_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Webhook manager not available")

    if not mgr.unregister(webhook_id):
        raise HTTPException(status_code=404, detail="Webhook not found")

    return {"status": "deleted", "webhook_id": webhook_id}


@router.get("/{webhook_id}/deliveries")
async def webhook_deliveries(
    webhook_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Get delivery history for a webhook."""
    mgr = getattr(request.app.state, "webhook_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Webhook manager not available")

    deliveries = mgr.get_deliveries(webhook_id=webhook_id, limit=limit)
    return {
        "deliveries": [d.model_dump() for d in deliveries],
        "total": len(deliveries),
    }


__all__ = ["RegisterWebhookRequest", "UpdateWebhookRequest", "router"]
