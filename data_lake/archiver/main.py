"""Kafka-to-MinIO archiver: buffers scored transactions and writes Parquet files
with Delta Lake metadata to object storage.
"""

from __future__ import annotations

import asyncio
import io
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from minio import Minio

from data_lake.archiver.config import ArchiverConfig
from shared.kafka_utils import (
    KafkaConsumerWrapper,
    setup_graceful_shutdown,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Delta Lake log helpers
# ---------------------------------------------------------------------------


def _delta_add_action(
    path: str, size: int, partition_values: dict[str, str], timestamp_ms: int
) -> dict[str, Any]:
    """Build a Delta Lake ``add`` action entry."""
    return {
        "add": {
            "path": path,
            "partitionValues": partition_values,
            "size": size,
            "modificationTime": timestamp_ms,
            "dataChange": True,
        }
    }


def _write_delta_log(
    client: Minio,
    bucket: str,
    base_prefix: str,
    version: int,
    actions: list[dict[str, Any]],
) -> None:
    """Write a Delta Lake JSON log entry to MinIO."""
    log_path = f"{base_prefix}/_delta_log/{version:020d}.json"
    lines = [orjson.dumps(a).decode() for a in actions]
    payload = "\n".join(lines).encode()
    client.put_object(
        bucket,
        log_path,
        io.BytesIO(payload),
        length=len(payload),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Archiver
# ---------------------------------------------------------------------------


class DataLakeArchiver:
    """Consumes scored transactions from Kafka, buffers them, and flushes
    to MinIO as Parquet files partitioned by date.
    """

    def __init__(self, config: ArchiverConfig) -> None:
        self._config = config

        self._consumer = KafkaConsumerWrapper(
            bootstrap_servers=config.kafka_bootstrap_servers,
            topics=[config.kafka_topic_scored],
            group_id=config.kafka_consumer_group,
            max_retries=config.kafka_max_retries,
            retry_backoff_ms=config.kafka_retry_backoff_ms,
        )

        self._minio = Minio(
            endpoint=config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_use_ssl,
        )

        # Internal state
        self._buffer: list[dict[str, Any]] = []
        self._last_flush_time: float = time.monotonic()
        self._total_archived: int = 0
        self._total_bytes_written: int = 0
        self._delta_version: int = 0
        self._running: bool = False
        self._flush_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the archiver: ensure bucket, start consumer, begin flush loop."""
        self._ensure_bucket()

        await self._consumer.start()
        self._running = True

        setup_graceful_shutdown(consumer=self._consumer)

        # Start periodic flush in background
        self._flush_task = asyncio.create_task(self._periodic_flush())

        await logger.ainfo(
            "archiver_started",
            topic=self._config.kafka_topic_scored,
            buffer_max=self._config.buffer_max_messages,
            flush_interval=self._config.buffer_flush_interval_seconds,
        )

        try:
            await self._consumer.consume(self._handle_message)
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Flush remaining buffer and shut down."""
        self._running = False

        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        if self._buffer:
            await logger.ainfo("flushing_remaining_buffer", count=len(self._buffer))
            await self._flush()

        await self._consumer.stop()

        await logger.ainfo(
            "archiver_stopped",
            total_archived=self._total_archived,
            total_bytes_written=self._total_bytes_written,
        )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: dict[str, Any]) -> None:
        """Buffer an incoming scored-transaction message."""
        self._buffer.append(raw)

        if len(self._buffer) >= self._config.buffer_max_messages:
            await self._flush()

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    async def _periodic_flush(self) -> None:
        """Periodically flush the buffer regardless of size."""
        while self._running:
            await asyncio.sleep(self._config.buffer_flush_interval_seconds)
            if self._buffer:
                elapsed = time.monotonic() - self._last_flush_time
                await logger.ainfo(
                    "periodic_flush_triggered",
                    buffer_size=len(self._buffer),
                    elapsed_seconds=round(elapsed, 1),
                )
                await self._flush()

    async def _flush(self) -> None:
        """Write buffered messages to MinIO as a Parquet file."""
        if not self._buffer:
            return

        messages = self._buffer.copy()
        self._buffer.clear()
        self._last_flush_time = time.monotonic()

        # Run blocking I/O in executor
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_parquet, messages)

    def _write_parquet(self, messages: list[dict[str, Any]]) -> None:
        """Serialize messages to Parquet and upload to MinIO (blocking)."""
        now = datetime.now(tz=UTC)
        partition = f"year={now.year}/month={now.month:02d}/day={now.day:02d}"
        file_id = uuid.uuid4().hex[:12]
        file_name = f"part-{file_id}.parquet"
        object_path = f"{self._config.base_prefix}/{partition}/{file_name}"

        # Flatten messages into columnar format
        columns = self._messages_to_columns(messages)
        table = pa.table(columns)

        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        data = buf.getvalue()
        size = len(data)

        self._minio.put_object(
            self._config.minio_bucket,
            object_path,
            io.BytesIO(data),
            length=size,
            content_type="application/octet-stream",
        )

        self._total_archived += len(messages)
        self._total_bytes_written += size

        # Write Delta Lake log entry
        try:
            action = _delta_add_action(
                path=object_path,
                size=size,
                partition_values={
                    "year": str(now.year),
                    "month": f"{now.month:02d}",
                    "day": f"{now.day:02d}",
                },
                timestamp_ms=int(now.timestamp() * 1000),
            )
            _write_delta_log(
                client=self._minio,
                bucket=self._config.minio_bucket,
                base_prefix=self._config.base_prefix,
                version=self._delta_version,
                actions=[action],
            )
            self._delta_version += 1
        except Exception:
            structlog.get_logger(__name__).exception(
                "delta_log_write_failed",
                version=self._delta_version,
            )

        structlog.get_logger(__name__).info(
            "parquet_written",
            path=object_path,
            records=len(messages),
            bytes=size,
            total_archived=self._total_archived,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_columns(messages: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """Convert a list of message dicts to a column-oriented dict for Arrow."""
        columns: dict[str, list[Any]] = {
            "transaction_id": [],
            "user_id": [],
            "amount": [],
            "currency": [],
            "transaction_type": [],
            "timestamp": [],
            "fraud_score": [],
            "model_version": [],
            "scoring_latency_ms": [],
            "scored_at": [],
            "is_flagged": [],
            "flag_reason": [],
            "raw_json": [],
        }

        for msg in messages:
            columns["transaction_id"].append(str(msg.get("transaction_id", "")))
            columns["user_id"].append(str(msg.get("user_id", "")))
            columns["amount"].append(float(msg.get("amount", 0)))
            columns["currency"].append(msg.get("currency", "USD"))
            columns["transaction_type"].append(msg.get("transaction_type", ""))
            columns["timestamp"].append(msg.get("timestamp", ""))
            columns["fraud_score"].append(float(msg.get("fraud_score", 0.0)))
            columns["model_version"].append(msg.get("model_version", ""))
            columns["scoring_latency_ms"].append(float(msg.get("scoring_latency_ms", 0.0)))
            columns["scored_at"].append(msg.get("scored_at", ""))
            columns["is_flagged"].append(bool(msg.get("is_flagged", False)))
            columns["flag_reason"].append(msg.get("flag_reason", ""))
            columns["raw_json"].append(orjson.dumps(msg).decode())

        return columns

    def _ensure_bucket(self) -> None:
        """Create the MinIO bucket if it does not already exist."""
        if not self._minio.bucket_exists(self._config.minio_bucket):
            self._minio.make_bucket(self._config.minio_bucket)
            structlog.get_logger(__name__).info(
                "bucket_created",
                bucket=self._config.minio_bucket,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run() -> None:
    """Bootstrap and run the archiver."""
    config = ArchiverConfig()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(config.log_level),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    archiver = DataLakeArchiver(config)
    try:
        await archiver.start()
    except KeyboardInterrupt:
        await logger.ainfo("archiver_interrupted")
    finally:
        await archiver.stop()


def main() -> None:
    """Synchronous entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
