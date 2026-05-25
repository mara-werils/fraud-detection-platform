"""Tenant-partitioned ClickHouse schema for multi-tenant isolation.

Adds org_id to all tables and partitions by (org_id, toYYYYMM(timestamp))
for efficient per-tenant data access and TTL management.

Data governance:
  - Each org's data is stored in its own partition shard
  - TTL per org (enterprise: 730 days, growth: 365, starter: 90)
  - Row-level access: queries always filter on org_id
"""

from __future__ import annotations

import asyncio

import structlog

logger = structlog.get_logger(__name__)

# TTL days per plan
PLAN_RETENTION_DAYS: dict[str, int] = {
    "starter": 90,
    "growth": 365,
    "enterprise": 730,
}

# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_TENANT_TRANSACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS {db}.tenant_transactions (
    org_id        LowCardinality(String),
    transaction_id UUID,
    user_id       String,
    amount        Decimal64(4),
    currency      LowCardinality(String),
    merchant_id   Nullable(String),
    ip_address    Nullable(String),
    device_id     Nullable(String),
    latitude      Nullable(Float64),
    longitude     Nullable(Float64),
    timestamp     DateTime64(3),

    -- ML features snapshot
    txn_count_1m        UInt32,
    txn_count_5m        UInt32,
    txn_count_1h        UInt32,
    txn_count_24h       UInt32,
    amount_zscore       Float64,
    amount_to_avg_ratio Float64,
    distance_km         Float64,
    is_new_device       UInt8,
    is_new_merchant     UInt8,

    -- Outcome
    fraud_score   Float32,
    decision      LowCardinality(String),
    is_fraud      UInt8 DEFAULT 0,

    inserted_at   DateTime64(3) DEFAULT now64(3)
) ENGINE = ReplicatedMergeTree('/clickhouse/tables/{{shard}}/tenant_transactions', '{{replica}}')
PARTITION BY (org_id, toYYYYMM(timestamp))
ORDER BY (org_id, user_id, timestamp)
TTL timestamp + INTERVAL 365 DAY
SETTINGS index_granularity = 8192
"""

CREATE_TENANT_SCORES_MV = """
CREATE MATERIALIZED VIEW IF NOT EXISTS {db}.tenant_hourly_scores
ENGINE = AggregatingMergeTree()
PARTITION BY (org_id, toYYYYMM(hour))
ORDER BY (org_id, hour)
AS
SELECT
    org_id,
    toStartOfHour(timestamp) AS hour,
    count()                                     AS total_transactions,
    countIf(decision = 'BLOCK')                AS blocked,
    countIf(decision = 'REVIEW')               AS reviewed,
    countIf(decision = 'ALLOW')                AS allowed,
    avg(fraud_score)                            AS avg_fraud_score,
    max(fraud_score)                            AS max_fraud_score,
    sum(is_fraud)                               AS confirmed_fraud_count
FROM {db}.tenant_transactions
GROUP BY org_id, hour
"""

CREATE_TENANT_USER_STATS_MV = """
CREATE MATERIALIZED VIEW IF NOT EXISTS {db}.tenant_user_risk_profiles
ENGINE = AggregatingMergeTree()
PARTITION BY org_id
ORDER BY (org_id, user_id)
AS
SELECT
    org_id,
    user_id,
    count()             AS total_txns,
    sum(is_fraud)       AS fraud_count,
    avg(fraud_score)    AS avg_risk_score,
    max(fraud_score)    AS max_risk_score,
    max(timestamp)      AS last_seen
FROM {db}.tenant_transactions
GROUP BY org_id, user_id
"""

# ── Retention management ──────────────────────────────────────────────────────

UPDATE_ORG_TTL_QUERY = """
ALTER TABLE {db}.tenant_transactions
MODIFY TTL timestamp + INTERVAL {days} DAY
WHERE org_id = '{org_id}'
"""


class TenantClickHouseStore:
    """Manages multi-tenant ClickHouse operations with org_id isolation."""

    def __init__(self, client, database: str = "fraud") -> None:
        self._client = client
        self._db = database
        self._buffer: list[dict] = []
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create tenant-partitioned tables and materialized views."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._create_schema)

    def _create_schema(self) -> None:
        self._client.command(
            CREATE_TENANT_TRANSACTIONS_TABLE.format(db=self._db)
        )
        self._client.command(
            CREATE_TENANT_SCORES_MV.format(db=self._db)
        )
        self._client.command(
            CREATE_TENANT_USER_STATS_MV.format(db=self._db)
        )
        logger.info("tenant_clickhouse_schema_created", db=self._db)

    async def insert_transaction(self, org_id: str, row: dict) -> None:
        """Buffer a transaction row for batch insert."""
        row["org_id"] = org_id
        async with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= 500:
                await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._do_insert, batch)

    def _do_insert(self, rows: list[dict]) -> None:
        if not rows:
            return
        columns = list(rows[0].keys())
        data = [[r.get(c) for c in columns] for r in rows]
        self._client.insert(
            f"{self._db}.tenant_transactions",
            data=data,
            column_names=columns,
        )

    async def query_org_stats(self, org_id: str, hours: int = 24) -> dict:
        """Return hourly aggregated stats for an org."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._query_org_stats, org_id, hours)

    def _query_org_stats(self, org_id: str, hours: int) -> dict:
        result = self._client.query(
            f"""
            SELECT
                hour,
                total_transactions,
                blocked,
                reviewed,
                allowed,
                avg_fraud_score,
                confirmed_fraud_count
            FROM {self._db}.tenant_hourly_scores
            WHERE org_id = %(org_id)s
              AND hour >= now() - INTERVAL %(hours)s HOUR
            ORDER BY hour DESC
            """,
            parameters={"org_id": org_id, "hours": hours},
        )
        return {
            "org_id": org_id,
            "hours": hours,
            "rows": result.result_rows,
            "columns": result.column_names,
        }

    async def set_retention_days(self, org_id: str, plan: str) -> None:
        """Update the TTL for an org's data based on their plan."""
        days = PLAN_RETENTION_DAYS.get(plan, 90)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._client.command,
            UPDATE_ORG_TTL_QUERY.format(db=self._db, org_id=org_id, days=days),
        )
        await logger.ainfo("tenant_retention_updated", org_id=org_id, plan=plan, days=days)
