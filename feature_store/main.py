"""Feature Store service entry point.

Consumes raw transactions from Kafka, computes online features in Redis,
and persists everything to ClickHouse for offline analysis.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any

import structlog

from shared.kafka_utils import KafkaConsumerWrapper, setup_graceful_shutdown
from shared.logging import setup_logging
from shared.schemas import Transaction

from feature_store.config import FeatureStoreConfig
from feature_store.offline_store import OfflineFeatureStore
from feature_store.online_store import OnlineFeatureStore

logger = structlog.get_logger(__name__)


async def _handle_transaction(
    message: dict[str, Any],
    online_store: OnlineFeatureStore,
    offline_store: OfflineFeatureStore,
) -> None:
    """Process a single raw transaction message.

    1. Parse the raw dict into a Transaction.
    2. Update online features in Redis.
    3. Retrieve the computed feature vector.
    4. Write the transaction + features to ClickHouse (buffered).
    """
    txn = Transaction.model_validate(message)

    await online_store.update_features(txn)

    features = await online_store.get_features(txn.user_id, txn)

    # Score is not computed here — feature store stores 0.0 as placeholder.
    # The scoring service will read from the "scored" topic for final scores.
    await offline_store.insert_transaction(txn, features, score=0.0)

    await logger.adebug(
        "transaction_processed",
        txn_id=str(txn.transaction_id),
        user_id=str(txn.user_id),
        amount=str(txn.amount),
    )


async def main() -> None:
    """Start the Feature Store service."""
    config = FeatureStoreConfig()
    setup_logging(
        service_name=config.service_name,
        log_level=config.log_level,
        log_format=config.log_format,
    )

    await logger.ainfo("feature_store_starting")

    # --- Initialise stores ---------------------------------------------------
    online_store = OnlineFeatureStore(
        redis_url=config.redis_url,
        max_connections=config.redis_max_connections,
        socket_timeout=config.redis_socket_timeout,
    )
    await online_store.connect()

    offline_store = OfflineFeatureStore(
        host=config.clickhouse_host,
        port=config.clickhouse_port,
        user=config.clickhouse_user,
        password=config.clickhouse_password,
        database=config.clickhouse_database,
        batch_size=config.batch_size,
        flush_interval=config.batch_flush_interval_seconds,
    )
    await offline_store.connect()

    # --- Kafka consumer -------------------------------------------------------
    consumer = KafkaConsumerWrapper(
        bootstrap_servers=config.kafka_bootstrap_servers,
        topics=[config.kafka_topic_raw_txn],
        group_id=config.kafka_consumer_group,
        max_retries=config.kafka_max_retries,
        retry_backoff_ms=config.kafka_retry_backoff_ms,
    )
    await consumer.start()

    # --- Graceful shutdown ----------------------------------------------------
    shutdown_event = asyncio.Event()

    async def _shutdown(sig: signal.Signals) -> None:
        await logger.ainfo("shutdown_signal_received", signal=sig.name)
        shutdown_event.set()
        await consumer.stop()
        await offline_store.close()
        await online_store.close()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(_shutdown(s)),
        )

    await logger.ainfo(
        "feature_store_started",
        kafka_topic=config.kafka_topic_raw_txn,
        consumer_group=config.kafka_consumer_group,
    )

    # --- Consume loop ---------------------------------------------------------
    async def handler(message: dict[str, Any]) -> None:
        await _handle_transaction(message, online_store, offline_store)

    try:
        await consumer.consume(handler)
    except Exception:
        await logger.aexception("feature_store_fatal_error")
    finally:
        if not shutdown_event.is_set():
            await consumer.stop()
            await offline_store.close()
            await online_store.close()
        await logger.ainfo("feature_store_stopped")


if __name__ == "__main__":
    asyncio.run(main())
