"""Async Kafka producer/consumer wrappers with retry logic and graceful shutdown."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable, Coroutine
from typing import Any

import orjson
import structlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, KafkaError

logger = structlog.get_logger(__name__)

__all__ = [
    "KafkaProducerWrapper",
    "KafkaConsumerWrapper",
    "setup_graceful_shutdown",
]


class KafkaProducerWrapper:
    """Async Kafka producer with retry logic and graceful shutdown."""

    def __init__(
        self,
        bootstrap_servers: str,
        max_retries: int = 5,
        retry_backoff_ms: int = 1000,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_ms / 1000.0
        self._producer: AIOKafkaProducer | None = None
        self._healthy = False

    @property
    def is_healthy(self) -> bool:
        """Check if the producer is connected and healthy."""
        return self._healthy and self._producer is not None

    async def start(self) -> None:
        """Start the producer with retry logic."""
        for attempt in range(1, self._max_retries + 1):
            try:
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self._bootstrap_servers,
                    value_serializer=lambda v: orjson.dumps(v) if isinstance(v, dict) else v,
                    key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
                    acks="all",
                    enable_idempotence=True,
                    max_request_size=1_048_576,
                    compression_type="snappy",
                )
                await self._producer.start()
                self._healthy = True
                await logger.ainfo(
                    "kafka_producer_started",
                    bootstrap_servers=self._bootstrap_servers,
                )
                return
            except KafkaConnectionError as exc:
                await logger.awarning(
                    "kafka_producer_connection_failed",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=str(exc),
                )
                if attempt == self._max_retries:
                    raise
                await asyncio.sleep(self._retry_backoff_s * attempt)

    async def send(
        self,
        topic: str,
        value: bytes | dict[str, Any],
        key: str | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> None:
        """Send a message to a Kafka topic with retry logic.

        Args:
            topic: Target Kafka topic.
            value: Message payload (bytes or dict for auto-serialization).
            key: Optional partition key.
            headers: Optional message headers.
        """
        if self._producer is None:
            raise RuntimeError("Producer not started. Call start() first.")

        for attempt in range(1, self._max_retries + 1):
            try:
                await self._producer.send_and_wait(
                    topic=topic,
                    value=value,
                    key=key,
                    headers=headers,
                )
                return
            except KafkaError as exc:
                await logger.awarning(
                    "kafka_send_failed",
                    topic=topic,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt == self._max_retries:
                    self._healthy = False
                    raise
                await asyncio.sleep(self._retry_backoff_s * attempt)

    async def stop(self) -> None:
        """Gracefully stop the producer, flushing pending messages."""
        if self._producer is not None:
            self._healthy = False
            await self._producer.stop()
            self._producer = None
            await logger.ainfo("kafka_producer_stopped")


class KafkaConsumerWrapper:
    """Async Kafka consumer with retry logic and graceful shutdown."""

    def __init__(
        self,
        bootstrap_servers: str,
        topics: list[str],
        group_id: str,
        max_retries: int = 5,
        retry_backoff_ms: int = 1000,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._topics = topics
        self._group_id = group_id
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_ms / 1000.0
        self._consumer: AIOKafkaConsumer | None = None
        self._running = False
        self._healthy = False

    @property
    def is_healthy(self) -> bool:
        """Check if the consumer is connected and healthy."""
        return self._healthy and self._consumer is not None

    async def start(self) -> None:
        """Start the consumer with retry logic."""
        for attempt in range(1, self._max_retries + 1):
            try:
                self._consumer = AIOKafkaConsumer(
                    *self._topics,
                    bootstrap_servers=self._bootstrap_servers,
                    group_id=self._group_id,
                    value_deserializer=lambda v: orjson.loads(v),
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                    auto_commit_interval_ms=5000,
                    max_poll_records=500,
                )
                await self._consumer.start()
                self._healthy = True
                self._running = True
                await logger.ainfo(
                    "kafka_consumer_started",
                    topics=self._topics,
                    group_id=self._group_id,
                )
                return
            except KafkaConnectionError as exc:
                await logger.awarning(
                    "kafka_consumer_connection_failed",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=str(exc),
                )
                if attempt == self._max_retries:
                    raise
                await asyncio.sleep(self._retry_backoff_s * attempt)

    async def consume(
        self,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Consume messages and pass them to the handler.

        Args:
            handler: Async callable that processes each deserialized message.
        """
        if self._consumer is None:
            raise RuntimeError("Consumer not started. Call start() first.")

        try:
            async for message in self._consumer:
                if not self._running:
                    break
                try:
                    await handler(message.value)
                except Exception:
                    await logger.aexception(
                        "kafka_message_handler_error",
                        topic=message.topic,
                        partition=message.partition,
                        offset=message.offset,
                    )
        except KafkaError as exc:
            self._healthy = False
            await logger.aerror("kafka_consumer_error", error=str(exc))
            raise

    async def stop(self) -> None:
        """Gracefully stop the consumer."""
        self._running = False
        if self._consumer is not None:
            self._healthy = False
            await self._consumer.stop()
            self._consumer = None
            await logger.ainfo("kafka_consumer_stopped")


def setup_graceful_shutdown(
    producer: KafkaProducerWrapper | None = None,
    consumer: KafkaConsumerWrapper | None = None,
) -> None:
    """Register signal handlers for graceful shutdown.

    Args:
        producer: Optional producer to stop on shutdown.
        consumer: Optional consumer to stop on shutdown.
    """
    loop = asyncio.get_event_loop()

    async def _shutdown(sig: signal.Signals) -> None:
        await logger.ainfo("shutdown_signal_received", signal=sig.name)
        if consumer is not None:
            await consumer.stop()
        if producer is not None:
            await producer.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(_shutdown(s)),
        )
