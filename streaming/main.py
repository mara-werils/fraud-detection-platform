"""Main entry point for the streaming enrichment pipeline.

Pipeline: Kafka(raw_txn) -> enrich -> window aggregates -> update features -> Kafka(enriched)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from feature_store.online_store import OnlineFeatureStore
from shared.kafka_utils import (
    KafkaConsumerWrapper,
    KafkaProducerWrapper,
    setup_graceful_shutdown,
)
from shared.schemas import Transaction
from streaming.config import StreamingConfig
from streaming.enrichment import TransactionEnricher
from streaming.windowed_aggregates import WindowedAggregator

logger = structlog.get_logger(__name__)


class StreamingPipeline:
    """Async pipeline that consumes raw transactions, enriches them,
    computes window aggregates, updates the online feature store,
    and produces enriched events downstream.
    """

    def __init__(self, config: StreamingConfig) -> None:
        self._config = config

        self._consumer = KafkaConsumerWrapper(
            bootstrap_servers=config.kafka_bootstrap_servers,
            topics=[config.kafka_topic_raw_txn],
            group_id=config.kafka_consumer_group,
            max_retries=config.kafka_max_retries,
            retry_backoff_ms=config.kafka_retry_backoff_ms,
        )
        self._producer = KafkaProducerWrapper(
            bootstrap_servers=config.kafka_bootstrap_servers,
            max_retries=config.kafka_max_retries,
            retry_backoff_ms=config.kafka_retry_backoff_ms,
        )
        self._enricher = TransactionEnricher(geoip_db_path=config.geoip_db_path)
        self._aggregator = WindowedAggregator(
            redis_url=config.redis_url,
            max_connections=config.redis_max_connections,
            socket_timeout=config.redis_socket_timeout,
        )
        self._feature_store = OnlineFeatureStore(
            redis_url=config.redis_url,
            max_connections=config.redis_max_connections,
            socket_timeout=config.redis_socket_timeout,
        )

        # Metrics
        self._processed_count: int = 0
        self._error_count: int = 0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all sub-components and begin consuming."""
        self._start_time = time.monotonic()

        await self._enricher.start()
        await self._aggregator.connect()
        await self._feature_store.connect()
        await self._producer.start()
        await self._consumer.start()

        setup_graceful_shutdown(producer=self._producer, consumer=self._consumer)

        await logger.ainfo(
            "streaming_pipeline_started",
            consumer_group=self._config.kafka_consumer_group,
            topic=self._config.kafka_topic_raw_txn,
        )

        await self._consumer.consume(self._handle_message)

    async def stop(self) -> None:
        """Gracefully shut down all sub-components."""
        await self._consumer.stop()
        await self._producer.stop()
        await self._enricher.close()
        await self._aggregator.close()
        await self._feature_store.close()

        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        await logger.ainfo(
            "streaming_pipeline_stopped",
            processed=self._processed_count,
            errors=self._error_count,
            uptime_seconds=round(uptime, 1),
        )

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: dict[str, Any]) -> None:
        """Process a single raw transaction message through the full pipeline."""
        start = time.monotonic()

        try:
            # 1. Parse raw transaction
            txn = Transaction.model_validate(raw)

            # 2. Enrich
            enrichment = self._enricher.enrich(raw)

            # 3. Window aggregates
            txn_ts = txn.timestamp.timestamp()
            await self._aggregator.record_transaction(
                user_id=txn.user_id,
                timestamp=txn_ts,
                amount=float(txn.amount),
                merchant_id=txn.merchant_id,
                device_id=txn.device_id,
            )
            window_aggs = await self._aggregator.get_aggregates(
                user_id=txn.user_id,
                now_ts=txn_ts,
            )

            # 4. Update online feature store
            await self._feature_store.update_features(txn)

            # 5. Build enriched event
            latency_ms = (time.monotonic() - start) * 1000.0
            enriched_event = self._build_enriched_event(
                txn=txn,
                enrichment_geo=enrichment.geo,
                enrichment_device=enrichment.device,
                merchant_risk=enrichment.merchant_risk_score,
                window_aggregates=window_aggs,
                pipeline_latency_ms=latency_ms,
            )

            # 6. Produce to enriched topic
            await self._producer.send(
                topic=self._config.kafka_topic_enriched,
                value=enriched_event,
                key=str(txn.user_id),
            )

            self._processed_count += 1

            if self._processed_count % 1000 == 0:
                await logger.ainfo(
                    "pipeline_progress",
                    processed=self._processed_count,
                    errors=self._error_count,
                    latency_ms=round(latency_ms, 2),
                )

        except Exception:
            self._error_count += 1
            await logger.aexception(
                "pipeline_message_error",
                processed=self._processed_count,
                errors=self._error_count,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_enriched_event(
        txn: Transaction,
        enrichment_geo: Any,
        enrichment_device: Any,
        merchant_risk: float,
        window_aggregates: dict[str, Any],
        pipeline_latency_ms: float,
    ) -> dict[str, Any]:
        """Merge original transaction data with enrichment outputs."""
        base = txn.model_dump(mode="json")
        base["enrichment"] = {
            "geo": {
                "country": enrichment_geo.country,
                "region": enrichment_geo.region,
                "city": enrichment_geo.city,
                "latitude": enrichment_geo.latitude,
                "longitude": enrichment_geo.longitude,
                "is_vpn": enrichment_geo.is_vpn,
                "is_proxy": enrichment_geo.is_proxy,
            },
            "device": {
                "os_family": enrichment_device.os_family,
                "browser_family": enrichment_device.browser_family,
                "device_type": enrichment_device.device_type,
                "fingerprint_hash": enrichment_device.fingerprint_hash,
            },
            "merchant_risk_score": merchant_risk,
        }
        base["window_aggregates"] = window_aggregates
        base["pipeline_latency_ms"] = round(pipeline_latency_ms, 2)
        return base


async def run() -> None:
    """Bootstrap and run the streaming pipeline."""
    config = StreamingConfig()

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

    pipeline = StreamingPipeline(config)
    try:
        await pipeline.start()
    except KeyboardInterrupt:
        await logger.ainfo("streaming_pipeline_interrupted")
    finally:
        await pipeline.stop()


def main() -> None:
    """Synchronous entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
