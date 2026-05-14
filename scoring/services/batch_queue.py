"""Async batch inference queue for throughput optimization.

Groups individual scoring requests into mini-batches before calling the
ML model. This reduces per-request overhead and improves GPU/CPU utilization.

Flow:
  request → asyncio.Queue → BatchProcessor groups → EnsembleScorer.score_batch()
          → Future.set_result() → response returned to caller
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BatchRequest:
    """A single inference request waiting to be batched."""

    payload: dict
    future: asyncio.Future = field(default_factory=asyncio.get_event_loop().create_future)
    enqueued_at: float = field(default_factory=time.monotonic)


class BatchInferenceQueue:
    """Collects individual score requests and dispatches them in batches.

    Args:
        scorer: EnsembleScorer instance (must expose score() coroutine).
        max_batch_size: Maximum items per batch.
        max_wait_ms: Maximum milliseconds to wait before flushing an incomplete batch.
        max_queue_size: Maximum items in flight before backpressure kicks in.
    """

    def __init__(
        self,
        scorer,
        max_batch_size: int = 64,
        max_wait_ms: int = 10,
        max_queue_size: int = 10_000,
    ) -> None:
        self._scorer = scorer
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[BatchRequest] = asyncio.Queue(maxsize=max_queue_size)
        self._processor_task: asyncio.Task | None = None
        self._total_batches = 0
        self._total_requests = 0
        self._total_wait_ms = 0.0

    def start(self) -> None:
        self._processor_task = asyncio.create_task(self._process_loop(), name="batch-inference")

    async def stop(self) -> None:
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

    async def score(self, payload: dict) -> dict:
        """Submit a single request; blocks until the batch result is ready.

        Args:
            payload: Transaction dict to score.

        Returns:
            Scored result dict from EnsembleScorer.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        request = BatchRequest(payload=payload, future=future)

        try:
            await asyncio.wait_for(
                self._queue.put(request),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Batch inference queue full — backpressure exceeded")

        return await future

    async def _process_loop(self) -> None:
        """Continuously collect items and dispatch batches."""
        while True:
            batch: list[BatchRequest] = []
            deadline = time.monotonic() + self._max_wait_ms / 1000.0

            # Collect first item (blocks until available)
            try:
                first = await self._queue.get()
                batch.append(first)
            except asyncio.CancelledError:
                return

            # Fill the batch up to max_batch_size within the time window
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    # Complete the current batch before stopping
                    break

            if batch:
                await self._dispatch_batch(batch)

    async def _dispatch_batch(self, batch: list[BatchRequest]) -> None:
        """Score all items in the batch and resolve their futures."""
        payloads = [r.payload for r in batch]
        self._total_batches += 1
        self._total_requests += len(batch)

        try:
            # Score all payloads; scorer returns a list of results
            results = await self._scorer.score_batch(payloads)

            for request, result in zip(batch, results):
                wait_ms = (time.monotonic() - request.enqueued_at) * 1000
                self._total_wait_ms += wait_ms
                if not request.future.done():
                    request.future.set_result(result)

        except Exception as exc:
            await logger.awarning("batch_inference_error", error=str(exc), batch_size=len(batch))
            for request in batch:
                if not request.future.done():
                    # Fall back to individual scoring
                    try:
                        result = await self._scorer.score(request.payload)
                        request.future.set_result(result)
                    except Exception as inner_exc:
                        request.future.set_exception(inner_exc)

    def stats(self) -> dict:
        avg_wait = (
            self._total_wait_ms / self._total_requests if self._total_requests > 0 else 0.0
        )
        avg_batch = (
            self._total_requests / self._total_batches if self._total_batches > 0 else 0.0
        )
        return {
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_size": round(avg_batch, 2),
            "avg_queue_wait_ms": round(avg_wait, 2),
            "queue_size": self._queue.qsize(),
        }
