"""Main simulator loop -- async Kafka producer that generates realistic financial transactions."""

from __future__ import annotations

import asyncio
import random
import signal
import time
from collections import Counter
from typing import Any

import structlog

from shared.kafka_utils import KafkaProducerWrapper
from shared.logging import setup_logging

from simulator.config import SimulatorConfig
from simulator.fraud_patterns import generate_fraud_transactions
from simulator.generators import UserProfile, build_user_pool, generate_legitimate_transaction


async def run_simulator(
    rate: int,
    fraud_ratio: float,
    duration: int,
    config: SimulatorConfig | None = None,
) -> None:
    """Run the transaction simulator.

    Args:
        rate: Target transactions per second.
        fraud_ratio: Fraction of transactions that should be fraudulent (0.0-1.0).
        duration: Total seconds to run. ``0`` means run until interrupted.
        config: Optional pre-built configuration; one is created from env if omitted.
    """
    if config is None:
        config = SimulatorConfig()

    setup_logging(service_name="simulator", log_level=config.log_level, log_format=config.log_format)
    logger: structlog.stdlib.BoundLogger = structlog.get_logger("simulator.main")

    await logger.ainfo(
        "simulator_starting",
        rate=rate,
        fraud_ratio=fraud_ratio,
        duration=duration if duration > 0 else "infinite",
        kafka_bootstrap=config.kafka_bootstrap_servers,
        topic=config.kafka_topic_raw_txn,
        user_pool_size=config.user_pool_size,
    )

    # Build user pool
    users: list[UserProfile] = build_user_pool(config.user_pool_size)
    await logger.ainfo("user_pool_created", count=len(users))

    # Kafka producer
    producer = KafkaProducerWrapper(
        bootstrap_servers=config.kafka_bootstrap_servers,
        max_retries=config.kafka_max_retries,
        retry_backoff_ms=config.kafka_retry_backoff_ms,
    )
    await producer.start()

    # Graceful shutdown handling
    shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    def _signal_handler(sig: signal.Signals) -> None:
        logger_sync = structlog.get_logger("simulator.main")
        # Cannot await in a signal handler callback registered via add_signal_handler,
        # so we just set the event and let the main loop clean up.
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler, sig)

    # Metrics
    total_sent = 0
    fraud_sent = 0
    legit_sent = 0
    pattern_counter: Counter[str] = Counter()
    start_time = time.monotonic()

    # Pre-compute interval between transactions
    interval = 1.0 / rate if rate > 0 else 0.1

    # Fraud transaction buffer -- some fraud patterns emit multiple txns at once
    fraud_buffer: list[Any] = []

    try:
        while not shutdown_event.is_set():
            elapsed = time.monotonic() - start_time
            if duration > 0 and elapsed >= duration:
                await logger.ainfo("duration_reached", elapsed_seconds=round(elapsed, 1))
                break

            tick_start = time.monotonic()

            # Decide: fraud or legitimate
            if fraud_buffer:
                # Drain buffered fraud transactions one at a time
                txn = fraud_buffer.pop(0)
                is_fraud = True
            elif random.random() < fraud_ratio:
                user = random.choice(users)
                fraud_txns = generate_fraud_transactions(user)
                txn = fraud_txns.pop(0)
                if fraud_txns:
                    fraud_buffer.extend(fraud_txns)
                is_fraud = True
            else:
                user = random.choice(users)
                txn = generate_legitimate_transaction(user)
                is_fraud = False

            # Publish to Kafka
            payload = txn.to_json_bytes()
            await producer.send(
                topic=config.kafka_topic_raw_txn,
                value=payload,
                key=str(txn.user_id),
            )

            total_sent += 1
            if is_fraud:
                fraud_sent += 1
                pattern = txn.metadata.get("fraud_pattern", "unknown")
                pattern_counter[pattern] += 1
            else:
                legit_sent += 1

            # Log every transaction
            await logger.ainfo(
                "transaction_sent",
                transaction_id=str(txn.transaction_id),
                user_id=str(txn.user_id),
                amount=str(txn.amount),
                currency=txn.currency,
                type=txn.transaction_type.value,
                merchant=txn.merchant_id,
                category=txn.merchant_category,
                is_fraud=is_fraud,
                fraud_pattern=txn.metadata.get("fraud_pattern"),
                total_sent=total_sent,
            )

            # Periodic summary every 100 transactions
            if total_sent % 100 == 0:
                runtime = time.monotonic() - start_time
                await logger.ainfo(
                    "simulator_progress",
                    total_sent=total_sent,
                    fraud_sent=fraud_sent,
                    legit_sent=legit_sent,
                    actual_fraud_ratio=round(fraud_sent / total_sent, 4) if total_sent else 0,
                    effective_tps=round(total_sent / runtime, 2) if runtime > 0 else 0,
                    pattern_distribution=dict(pattern_counter),
                    runtime_seconds=round(runtime, 1),
                )

            # Rate limiting
            tick_elapsed = time.monotonic() - tick_start
            sleep_time = interval - tick_elapsed
            if sleep_time > 0:
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=sleep_time)
                except asyncio.TimeoutError:
                    pass  # Normal -- timeout means we continue to next tick

    except Exception:
        await logger.aexception("simulator_error")
        raise
    finally:
        runtime = time.monotonic() - start_time
        await logger.ainfo(
            "simulator_shutting_down",
            total_sent=total_sent,
            fraud_sent=fraud_sent,
            legit_sent=legit_sent,
            final_fraud_ratio=round(fraud_sent / total_sent, 4) if total_sent else 0,
            pattern_distribution=dict(pattern_counter),
            runtime_seconds=round(runtime, 1),
            effective_tps=round(total_sent / runtime, 2) if runtime > 0 else 0,
        )
        await producer.stop()
        await logger.ainfo("simulator_stopped")
