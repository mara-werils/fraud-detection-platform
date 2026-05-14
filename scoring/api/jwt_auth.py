"""JWT authentication and user management API."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from scoring.config import ScoringConfig
from scoring.db.engine import get_db_session
from scoring.db.models import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)

_config: ScoringConfig | None = None


def _get_config() -> ScoringConfig:
    global _config
    if _config is None:
        _config = ScoringConfig()
    return _config


# ── Password helpers ──────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT helpers ───────────────────────────────────────────────────────────────


def create_access_token(
    subject: str,
    org_id: str,
    role: str,
    config: ScoringConfig | None = None,
) -> str:
    cfg = config or _get_config()
    expire = datetime.now(timezone.utc) + timedelta(minutes=cfg.jwt_access_token_expire_minutes)
    payload = {
        "sub": subject,
        "org_id": org_id,
        "role": role,
        "type": "access",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def create_refresh_token(subject: str, org_id: str, config: ScoringConfig | None = None) -> str:
    cfg = config or _get_config()
    expire = datetime.now(timezone.utc) + timedelta(days=cfg.jwt_refresh_token_expire_days)
    payload = {
        "sub": subject,
        "org_id": org_id,
        "type": "refresh",
        "exp": expire,
    }
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def decode_token(token: str, config: ScoringConfig | None = None) -> dict:
    cfg = config or _get_config()
    try:
        return jwt.decode(token, cfg.jwt_secret_key, algorithms=[cfg.jwt_algorithm])
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e


# ── FastAPI dependency ────────────────────────────────────────────────────────


class CurrentUser(BaseModel):
    user_id: str
    email: str
    org_id: str
    role: str


async def get_current_user(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
) -> CurrentUser | None:
    """Resolve JWT token to a CurrentUser, attaches org_id to request.state."""
    if not token:
        return None

    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Expected access token")

    user = CurrentUser(
        user_id=payload["sub"],
        email=payload.get("email", ""),
        org_id=payload["org_id"],
        role=payload.get("role", "analyst"),
    )
    request.state.org_id = user.org_id
    request.state.user_id = user.user_id
    return user


async def require_user(
    current_user: CurrentUser | None = Depends(get_current_user),
) -> CurrentUser:
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user


# ── Schemas ───────────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)
    full_name: str = Field(default="")
    org_id: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    org_id: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(body: RegisterRequest):
    """Register a new user within an organization."""
    async with get_db_session() as session:
        existing = await session.execute(select(User).where(User.email == body.email))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Email already registered")

        user = User(
            id=uuid.uuid4(),
            org_id=uuid.UUID(body.org_id),
            email=body.email,
            hashed_password=hash_password(body.password),
            full_name=body.full_name,
            role="analyst",
        )
        session.add(user)

    await logger.ainfo("user_registered", email=body.email, org_id=body.org_id)
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        org_id=str(user.org_id),
        is_active=user.is_active,
        created_at=user.created_at or datetime.utcnow(),
    )


@router.post("/token", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Obtain JWT access + refresh tokens."""
    cfg = _get_config()

    async with get_db_session() as session:
        result = await session.execute(select(User).where(User.email == form.username))
        user = result.scalar_one_or_none()

    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    access_token = create_access_token(
        subject=str(user.id),
        org_id=str(user.org_id),
        role=user.role,
        config=cfg,
    )
    refresh_token = create_refresh_token(
        subject=str(user.id),
        org_id=str(user.org_id),
        config=cfg,
    )

    await logger.ainfo("user_login", user_id=str(user.id), org_id=str(user.org_id))
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=cfg.jwt_access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: Request):
    """Exchange a refresh token for a new access token."""
    body = await request.json()
    token = body.get("refresh_token")
    if not token:
        raise HTTPException(status_code=400, detail="refresh_token required")

    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="Expected refresh token")

    cfg = _get_config()
    async with get_db_session() as session:
        user = await session.get(User, uuid.UUID(payload["sub"]))

    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or disabled")

    access_token = create_access_token(
        subject=str(user.id),
        org_id=str(user.org_id),
        role=user.role,
        config=cfg,
    )
    new_refresh = create_refresh_token(
        subject=str(user.id),
        org_id=str(user.org_id),
        config=cfg,
    )
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        expires_in=cfg.jwt_access_token_expire_minutes * 60,
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: CurrentUser = Depends(require_user)):
    """Return current user profile."""
    async with get_db_session() as session:
        user = await session.get(User, uuid.UUID(current_user.user_id))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        org_id=str(user.org_id),
        is_active=user.is_active,
        created_at=user.created_at or datetime.utcnow(),
    )
