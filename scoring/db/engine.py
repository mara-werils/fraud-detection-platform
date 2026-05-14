"""Async PostgreSQL engine and session factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from scoring.config import ScoringConfig

_engine = None
_session_factory = None


def get_engine(config: ScoringConfig | None = None):
    global _engine
    if _engine is None:
        cfg = config or ScoringConfig()
        _engine = create_async_engine(
            cfg.database_url,
            pool_size=cfg.db_pool_size,
            max_overflow=cfg.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False,
        )
    return _engine


def get_session_factory(config: ScoringConfig | None = None) -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        engine = get_engine(config)
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def get_db_session(config: ScoringConfig | None = None) -> AsyncIterator[AsyncSession]:
    factory = get_session_factory(config)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(config: ScoringConfig | None = None) -> None:
    """Create all tables if they don't exist."""
    from scoring.db.models import Base
    engine = get_engine(config)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
