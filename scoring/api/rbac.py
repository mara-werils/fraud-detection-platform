"""Role-Based Access Control (RBAC) for the fraud detection platform.

Roles (from most to least privileged):
  admin    — full platform management
  analyst  — investigate cases, provide feedback, view transactions
  viewer   — read-only access to transactions and scores
  api_user — programmatic access to scoring endpoints only
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Callable

from fastapi import Depends, HTTPException

from scoring.api.jwt_auth import CurrentUser, require_user

# ── Role definitions ──────────────────────────────────────────────────────────


class Role(StrEnum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


# Permission constants
PERM_SCORE = "score"
PERM_BATCH_SCORE = "batch_score"
PERM_READ_TRANSACTIONS = "read_transactions"
PERM_WRITE_CASES = "write_cases"
PERM_READ_CASES = "read_cases"
PERM_PROVIDE_FEEDBACK = "provide_feedback"
PERM_MANAGE_USERS = "manage_users"
PERM_MANAGE_ORG = "manage_org"
PERM_MANAGE_API_KEYS = "manage_api_keys"
PERM_VIEW_METRICS = "view_metrics"
PERM_EXPORT_DATA = "export_data"
PERM_MANAGE_WEBHOOKS = "manage_webhooks"
PERM_ADMIN = "admin"

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    Role.ADMIN: frozenset({
        PERM_SCORE, PERM_BATCH_SCORE, PERM_READ_TRANSACTIONS,
        PERM_WRITE_CASES, PERM_READ_CASES, PERM_PROVIDE_FEEDBACK,
        PERM_MANAGE_USERS, PERM_MANAGE_ORG, PERM_MANAGE_API_KEYS,
        PERM_VIEW_METRICS, PERM_EXPORT_DATA, PERM_MANAGE_WEBHOOKS,
        PERM_ADMIN,
    }),
    Role.ANALYST: frozenset({
        PERM_SCORE, PERM_BATCH_SCORE, PERM_READ_TRANSACTIONS,
        PERM_WRITE_CASES, PERM_READ_CASES, PERM_PROVIDE_FEEDBACK,
        PERM_VIEW_METRICS, PERM_EXPORT_DATA, PERM_MANAGE_WEBHOOKS,
    }),
    Role.VIEWER: frozenset({
        PERM_READ_TRANSACTIONS, PERM_READ_CASES, PERM_VIEW_METRICS,
    }),
    Role.API_USER: frozenset({
        PERM_SCORE, PERM_BATCH_SCORE,
    }),
}


def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, frozenset())


# ── Dependency factories ──────────────────────────────────────────────────────


def require_permission(permission: str) -> Callable:
    """Return a FastAPI dependency that enforces a specific permission."""

    async def _check(current_user: CurrentUser = Depends(require_user)) -> CurrentUser:
        if not has_permission(current_user.role, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: '{permission}' required (your role: {current_user.role})",
            )
        return current_user

    _check.__name__ = f"require_{permission}"
    return _check


def require_role(role: Role) -> Callable:
    """Return a FastAPI dependency that enforces a minimum role level."""
    role_order = [Role.VIEWER, Role.API_USER, Role.ANALYST, Role.ADMIN]

    async def _check(current_user: CurrentUser = Depends(require_user)) -> CurrentUser:
        user_role = current_user.role
        try:
            user_idx = role_order.index(user_role)
            required_idx = role_order.index(role)
        except ValueError:
            raise HTTPException(status_code=403, detail="Unknown role")

        if user_idx < required_idx:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' or higher required (your role: {user_role})",
            )
        return current_user

    _check.__name__ = f"require_role_{role}"
    return _check


# Pre-built common dependencies
require_admin = require_permission(PERM_ADMIN)
require_analyst = require_role(Role.ANALYST)
require_scorer = require_permission(PERM_SCORE)
require_case_write = require_permission(PERM_WRITE_CASES)
require_export = require_permission(PERM_EXPORT_DATA)
