"""Audit logging service for compliance and security monitoring.

Writes immutable audit records to PostgreSQL for:
  - Authentication events (login, token refresh, API key creation/revocation)
  - Data access (transaction search, case view, export)
  - Administrative actions (org create/update, user role change)
  - Model decisions (BLOCK/REVIEW overrides by analysts)

Records are also published to the Kafka 'audit' topic for SIEM integration.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime

import structlog

from scoring.db.engine import get_db_session
from scoring.db.models import AuditLog

logger = structlog.get_logger(__name__)

# Action constants
A_LOGIN = "auth.login"
A_LOGOUT = "auth.logout"
A_TOKEN_REFRESH = "auth.token_refresh"
A_API_KEY_CREATED = "api_key.created"
A_API_KEY_REVOKED = "api_key.revoked"
A_ORG_CREATED = "org.created"
A_ORG_UPDATED = "org.updated"
A_USER_CREATED = "user.created"
A_USER_ROLE_CHANGED = "user.role_changed"
A_CASE_CREATED = "case.created"
A_CASE_STATUS_CHANGED = "case.status_changed"
A_CASE_ASSIGNED = "case.assigned"
A_FEEDBACK_PROVIDED = "feedback.provided"
A_DATA_EXPORTED = "data.exported"
A_TRANSACTION_VIEWED = "transaction.viewed"
A_SCORE_OVERRIDE = "score.override"
A_WEBHOOK_CREATED = "webhook.created"
A_WEBHOOK_DELETED = "webhook.deleted"


class AuditLogger:
    """Async audit logger with buffered PostgreSQL writes.

    Buffers events to avoid a DB write on every request.
    Critical events (auth, admin) are written immediately.
    """

    CRITICAL_ACTIONS = frozenset({
        A_LOGIN, A_API_KEY_REVOKED, A_USER_ROLE_CHANGED, A_ORG_UPDATED,
        A_SCORE_OVERRIDE, A_DATA_EXPORTED,
    })

    def __init__(
        self,
        kafka_producer=None,
        flush_interval: float = 5.0,
        buffer_size: int = 500,
    ) -> None:
        self._producer = kafka_producer
        self._flush_interval = flush_interval
        self._buffer_size = buffer_size
        self._buffer: list[dict] = []
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._flush_loop(), name="audit-logger-flush")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        await self._flush()

    async def log(
        self,
        action: str,
        resource_type: str,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        resource_id: str | None = None,
        details: dict | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        """Record an audit event."""
        event = {
            "id": str(uuid.uuid4()),
            "org_id": org_id,
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await logger.ainfo(
            "audit_event",
            action=action,
            resource_type=resource_type,
            org_id=org_id,
            user_id=user_id,
        )

        # Publish to Kafka for SIEM (fire-and-forget)
        if self._producer:
            try:
                await self._producer.send(
                    topic="audit",
                    key=org_id or "system",
                    value=json.dumps(event, default=str).encode(),
                )
            except Exception as exc:
                await logger.awarning("audit_kafka_publish_error", error=str(exc))

        if action in self.CRITICAL_ACTIONS:
            # Write immediately for critical events
            await self._write_single(event)
        else:
            async with self._lock:
                self._buffer.append(event)
                if len(self._buffer) >= self._buffer_size:
                    await self._flush()

    async def query_org_audit(
        self,
        org_id: str,
        action_filter: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Query audit log for an org."""
        from sqlalchemy import select
        async with get_db_session() as session:
            q = (
                select(AuditLog)
                .where(AuditLog.org_id == uuid.UUID(org_id))
                .order_by(AuditLog.timestamp.desc())
                .limit(limit)
                .offset(offset)
            )
            if action_filter:
                q = q.where(AuditLog.action == action_filter)

            result = await session.execute(q)
            records = result.scalars().all()

        return [
            {
                "id": str(r.id),
                "action": r.action,
                "resource_type": r.resource_type,
                "resource_id": r.resource_id,
                "user_id": str(r.user_id) if r.user_id else None,
                "details": r.details,
                "ip_address": r.ip_address,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in records
        ]

    async def _write_single(self, event: dict) -> None:
        try:
            record = self._dict_to_model(event)
            async with get_db_session() as session:
                session.add(record)
        except Exception as exc:
            await logger.awarning("audit_db_write_error", error=str(exc))

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            try:
                await self._flush()
            except Exception as exc:
                await logger.awarning("audit_flush_error", error=str(exc))

    async def _flush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()

        records = [self._dict_to_model(e) for e in batch]
        try:
            async with get_db_session() as session:
                session.add_all(records)
        except Exception as exc:
            await logger.awarning("audit_db_flush_error", error=str(exc), count=len(records))

    @staticmethod
    def _dict_to_model(event: dict) -> AuditLog:
        return AuditLog(
            id=uuid.UUID(event["id"]),
            org_id=uuid.UUID(event["org_id"]) if event.get("org_id") else None,
            user_id=uuid.UUID(event["user_id"]) if event.get("user_id") else None,
            action=event["action"],
            resource_type=event["resource_type"],
            resource_id=event.get("resource_id"),
            details=event.get("details", {}),
            ip_address=event.get("ip_address"),
            user_agent=event.get("user_agent"),
        )


# ── Audit API endpoint ─────────────────────────────────────────────────────────

from fastapi import APIRouter, Depends, HTTPException, Request  # noqa: E402

from scoring.api.jwt_auth import CurrentUser  # noqa: E402
from scoring.api.rbac import require_analyst  # noqa: E402

audit_router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


@audit_router.get("/logs")
async def get_audit_logs(
    request: Request,
    action: str | None = None,
    limit: int = 100,
    offset: int = 0,
    current_user: CurrentUser = Depends(require_analyst),
):
    """Return audit log entries for the current org."""
    audit_logger = getattr(request.app.state, "audit_logger", None)
    if not audit_logger:
        raise HTTPException(status_code=503, detail="Audit logger not available")

    records = await audit_logger.query_org_audit(
        current_user.org_id,
        action_filter=action,
        limit=limit,
        offset=offset,
    )
    return {"items": records, "limit": limit, "offset": offset}
