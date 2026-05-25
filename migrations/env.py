"""Alembic environment — supports both online (direct DB) and offline (SQL script) modes.

Key design decisions
--------------------
* asyncpg is the *runtime* driver (used by the application).  Alembic needs a
  *synchronous* driver for migrations; we use psycopg2 at migration time so we
  don't have to spin up an async event loop just for DDL.  The URL is rewritten
  automatically: ``postgresql+asyncpg://`` → ``postgresql+psycopg2://``.

* The database URL is resolved from (in priority order):
  1. ``ALEMBIC_DATABASE_URL`` environment variable  (ideal for CI/CD)
  2. ``DATABASE_URL`` environment variable           (generic fallback)
  3. ``sqlalchemy.url`` from alembic.ini             (local dev default)

* ``target_metadata`` is wired to the application's ``Base.metadata`` so Alembic
  can detect schema drift when you run ``alembic check`` or ``alembic revision --autogenerate``.

* ``compare_type=True`` ensures column type changes (e.g. VARCHAR length bumps)
  are detected by --autogenerate.

* ``compare_server_default=True`` catches changes to server-side default expressions.

* ``include_schemas=True`` / ``version_table_schema`` keep the alembic_version
  table in the same schema as the application tables (public by default).
"""

from __future__ import annotations

import os
import re
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Import application models so that Base.metadata is populated before Alembic
# inspects it.  Any model added to scoring/db/models.py will be automatically
# picked up by --autogenerate.
# ---------------------------------------------------------------------------
from scoring.db.models import Base  # noqa: F401  (side-effect: registers all mappers)

# ---------------------------------------------------------------------------
# Alembic Config object — gives access to values in alembic.ini.
# ---------------------------------------------------------------------------
config = context.config

# Honour the logging configuration in alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_database_url() -> str:
    """Resolve the synchronous database URL for Alembic DDL operations.

    Priority:
        1. ALEMBIC_DATABASE_URL   – explicit override for CI/CD pipelines
        2. DATABASE_URL           – generic platform env var
        3. alembic.ini            – local developer default
    """
    url = (
        os.environ.get("ALEMBIC_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or config.get_main_option("sqlalchemy.url")
        or "postgresql+psycopg2://fraud:fraud@localhost:5432/fraud"
    )
    # Rewrite asyncpg URLs to psycopg2 for synchronous Alembic operations.
    # The running application keeps using asyncpg; migrations use psycopg2.
    url = re.sub(r"postgresql\+asyncpg://", "postgresql+psycopg2://", url)
    # Normalise bare postgresql:// (no driver) — psycopg2 is the default.
    url = re.sub(r"^postgresql://", "postgresql+psycopg2://", url)
    return url


def _get_render_item(type_, obj, autogen_context):  # noqa: ANN001
    """Hook called by --autogenerate for each schema item.

    Return ``False`` to tell Alembic to skip rendering the item, or ``None``
    to use the default renderer.
    """
    return None


# ---------------------------------------------------------------------------
# Offline mode — emit raw SQL, no live DB connection required.
# Usage: alembic upgrade head --sql > schema.sql
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    url = _get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        render_as_batch=False,
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode — run migrations against a live database connection.
# Usage: alembic upgrade head
# ---------------------------------------------------------------------------

def run_migrations_online() -> None:
    url = _get_database_url()

    # Override the URL in the config object so engine_from_config picks it up.
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # Never pool connections during migrations.
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_schemas=True,
            render_item=_get_render_item,
            # Keep alembic_version in the public schema alongside app tables.
            version_table="alembic_version",
            version_table_schema=None,
        )
        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Entrypoint — choose mode based on Alembic's runtime context.
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
