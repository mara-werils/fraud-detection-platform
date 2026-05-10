"""ClickHouse-based offline feature store for historical analysis and training data."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

import clickhouse_connect
import structlog
from clickhouse_connect.driver.client import Client as CHClient

from shared.schemas import FeatureVector, Transaction

logger = structlog.get_logger(__name__)

_CREATE_TRANSACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS {db}.transactions (
    transaction_id UUID,
    user_id UUID,
    amount Decimal64(4),
    currency LowCardinality(String),
    transaction_type LowCardinality(String),
    merchant_id Nullable(String),
    merchant_category Nullable(String),
    ip_address Nullable(String),
    device_id Nullable(String),
    latitude Nullable(Float64),
    longitude Nullable(Float64),
    timestamp DateTime64(3),

    -- Feature snapshot
    txn_count_1h UInt32,
    txn_count_24h UInt32,
    txn_amount_sum_1h Decimal64(4),
    txn_amount_sum_24h Decimal64(4),
    txn_amount_avg_24h Decimal64(4),
    unique_merchants_24h UInt16,
    max_amount_24h Decimal64(4),
    time_since_last_txn_seconds Float64,
    distance_from_last_txn_km Float64,
    is_new_device UInt8,
    is_new_merchant UInt8,
    amount_zscore Float64,
    amount_to_avg_ratio Float64,

    -- Score
    fraud_score Float32,
    is_fraud UInt8 DEFAULT 0,

    -- Metadata
    inserted_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (user_id, timestamp)
TTL timestamp + INTERVAL 365 DAY
"""


class OfflineFeatureStore:
    """Manages historical feature/transaction storage in ClickHouse.

    Buffers inserts and flushes either when the buffer reaches *batch_size*
    or every *flush_interval* seconds, whichever comes first.
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        batch_size: int = 500,
        flush_interval: float = 5.0,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        self._client: CHClient | None = None
        self._buffer: list[list[Any]] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create ClickHouse client and ensure tables exist."""
        self._client = await asyncio.to_thread(
            clickhouse_connect.get_client,
            host=self._host,
            port=self._port,
            username=self._user,
            password=self._password,
            database=self._database,
        )
        await asyncio.to_thread(
            self._client.command,
            _CREATE_TRANSACTIONS_TABLE.format(db=self._database),
        )
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        await logger.ainfo("offline_store_connected", host=self._host, database=self._database)

    async def close(self) -> None:
        """Flush remaining rows and close the ClickHouse client."""
        self._running = False
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush()
        if self._client is not None:
            self._client.close()
            self._client = None
        await logger.ainfo("offline_store_closed")

    async def health_check(self) -> bool:
        """Return True if ClickHouse is reachable."""
        if self._client is None:
            return False
        try:
            result = await asyncio.to_thread(self._client.command, "SELECT 1")
            return result == 1
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def insert_transaction(
        self,
        transaction: Transaction,
        features: FeatureVector,
        score: float,
    ) -> None:
        """Buffer a transaction row for batch insert into ClickHouse."""
        row = [
            str(transaction.transaction_id),
            str(transaction.user_id),
            float(transaction.amount),
            transaction.currency,
            transaction.transaction_type.value,
            transaction.merchant_id,
            transaction.merchant_category,
            transaction.ip_address,
            transaction.device_id,
            transaction.latitude,
            transaction.longitude,
            transaction.timestamp,
            features.txn_count_1h,
            features.txn_count_24h,
            float(features.txn_amount_sum_1h),
            float(features.txn_amount_sum_24h),
            float(features.txn_amount_avg_24h),
            features.unique_merchants_24h,
            float(features.max_amount_24h),
            features.time_since_last_txn_seconds,
            features.distance_from_last_txn_km,
            int(features.is_new_device),
            int(features.is_new_merchant),
            features.amount_zscore,
            features.amount_to_avg_ratio,
            score,
            0,  # is_fraud — labelled later
        ]

        async with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self._batch_size:
                await self._flush()

    async def get_user_profile(self, user_id: UUID) -> dict[str, Any]:
        """Return aggregate statistics from the user's transaction history."""
        if self._client is None:
            raise RuntimeError("OfflineFeatureStore is not connected")

        query = f"""
        SELECT
            count() AS total_txns,
            sum(amount) AS total_amount,
            avg(amount) AS avg_amount,
            max(amount) AS max_amount,
            min(timestamp) AS first_txn,
            max(timestamp) AS last_txn,
            uniqExact(merchant_id) AS unique_merchants,
            avg(fraud_score) AS avg_fraud_score,
            countIf(is_fraud = 1) AS fraud_count
        FROM {self._database}.transactions
        WHERE user_id = %(user_id)s
        """

        result = await asyncio.to_thread(
            self._client.query,
            query,
            parameters={"user_id": str(user_id)},
        )
        if not result.result_rows:
            return {}

        row = result.result_rows[0]
        columns = result.column_names
        return dict(zip(columns, row))

    async def get_training_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch labelled feature rows for ML training.

        Returns a list of dicts (column_name -> value) suitable for
        constructing a pandas DataFrame on the caller side.
        """
        if self._client is None:
            raise RuntimeError("OfflineFeatureStore is not connected")

        query = f"""
        SELECT
            transaction_id,
            user_id,
            amount,
            transaction_type,
            txn_count_1h,
            txn_count_24h,
            txn_amount_sum_1h,
            txn_amount_sum_24h,
            txn_amount_avg_24h,
            unique_merchants_24h,
            max_amount_24h,
            time_since_last_txn_seconds,
            distance_from_last_txn_km,
            is_new_device,
            is_new_merchant,
            amount_zscore,
            amount_to_avg_ratio,
            fraud_score,
            is_fraud,
            timestamp
        FROM {self._database}.transactions
        WHERE timestamp >= %(start)s AND timestamp < %(end)s
        ORDER BY timestamp
        """

        result = await asyncio.to_thread(
            self._client.query,
            query,
            parameters={"start": start_date.isoformat(), "end": end_date.isoformat()},
        )
        columns = result.column_names
        return [dict(zip(columns, row)) for row in result.result_rows]

    async def compute_merchant_risk(self, merchant_id: str) -> float:
        """Compute the historical fraud rate for a specific merchant.

        Returns a float in [0.0, 1.0] representing the fraction of
        fraudulent transactions for this merchant.
        """
        if self._client is None:
            raise RuntimeError("OfflineFeatureStore is not connected")

        query = f"""
        SELECT
            countIf(is_fraud = 1) AS fraud_count,
            count() AS total_count
        FROM {self._database}.transactions
        WHERE merchant_id = %(merchant_id)s
        """

        result = await asyncio.to_thread(
            self._client.query,
            query,
            parameters={"merchant_id": merchant_id},
        )
        if not result.result_rows:
            return 0.0
        fraud_count, total_count = result.result_rows[0]
        if total_count == 0:
            return 0.0
        return float(fraud_count) / float(total_count)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _flush(self) -> None:
        """Flush the buffered rows to ClickHouse."""
        if not self._buffer or self._client is None:
            return

        async with self._lock:
            rows = self._buffer[:]
            self._buffer.clear()

        column_names = [
            "transaction_id",
            "user_id",
            "amount",
            "currency",
            "transaction_type",
            "merchant_id",
            "merchant_category",
            "ip_address",
            "device_id",
            "latitude",
            "longitude",
            "timestamp",
            "txn_count_1h",
            "txn_count_24h",
            "txn_amount_sum_1h",
            "txn_amount_sum_24h",
            "txn_amount_avg_24h",
            "unique_merchants_24h",
            "max_amount_24h",
            "time_since_last_txn_seconds",
            "distance_from_last_txn_km",
            "is_new_device",
            "is_new_merchant",
            "amount_zscore",
            "amount_to_avg_ratio",
            "fraud_score",
            "is_fraud",
        ]

        try:
            await asyncio.to_thread(
                self._client.insert,
                f"{self._database}.transactions",
                rows,
                column_names=column_names,
            )
            await logger.ainfo("offline_store_flushed", row_count=len(rows))
        except Exception:
            await logger.aexception("offline_store_flush_error", row_count=len(rows))
            # Re-add rows to buffer so they are retried on the next flush
            async with self._lock:
                self._buffer = rows + self._buffer

    async def _periodic_flush(self) -> None:
        """Background task that flushes the buffer at regular intervals."""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            try:
                await self._flush()
            except Exception:
                await logger.aexception("periodic_flush_error")
