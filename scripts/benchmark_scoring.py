#!/usr/bin/env python3
"""Scoring pipeline benchmark with latency percentiles.

Measures latency, throughput, and peak memory consumption of the scoring
pipeline by replaying synthetic transactions through the ensemble scorer.

Usage:
    python scripts/benchmark_scoring.py --iterations 5000 --warmup 500
    python scripts/benchmark_scoring.py --iterations 10000 --output results.json

Metrics reported:
  - p50 / p90 / p95 / p99 latency (ms)
  - Mean latency and standard deviation
  - Throughput (transactions per second)
  - Peak resident-set-size (RSS) delta during the run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Any, Callable
from uuid import uuid4


# ---------------------------------------------------------------------------
# Synthetic payload generator
# ---------------------------------------------------------------------------

_MERCHANTS = [
    "amazon.com", "walmart.com", "target.com", "bestbuy.com",
    "etsy.com", "shopify-store.com", "gas-station-42", "airline-intl",
]

_CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD"]


def _random_transaction() -> dict[str, Any]:
    return {
        "transaction_id": str(uuid4()),
        "amount": round(random.uniform(0.50, 9999.99), 2),
        "currency": random.choice(_CURRENCIES),
        "merchant": random.choice(_MERCHANTS),
        "card_hash": f"card_{random.randint(1, 500):04d}",
        "timestamp": time.time(),
        "ip_address": f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "device_id": f"dev_{random.randint(1, 200):04d}",
    }


# ---------------------------------------------------------------------------
# Stub scorer (replace with real import when available)
# ---------------------------------------------------------------------------

def _stub_score(txn: dict[str, Any]) -> float:
    """Lightweight stand-in for the ensemble scorer.

    Performs enough work to be a realistic lower-bound benchmark: hashes
    fields, does floating-point arithmetic, and returns a 0-100 score.
    """
    h = hash(frozenset(txn.items()))
    score = abs(h) % 10001 / 100.0
    # Simulate feature lookup latency (~0.02 ms of busy work)
    _dummy = sum(x * x for x in range(200))
    return min(score, 100.0)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    iterations: int
    warmup: int
    latencies_ms: list[float] = field(repr=False)
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    stddev_ms: float = 0.0
    throughput_tps: float = 0.0
    peak_memory_mb: float = 0.0

    def summary_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("latencies_ms")
        return d


def _percentile(data: list[float], pct: float) -> float:
    k = (len(data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    d = k - f
    return data[f] + d * (data[c] - data[f])


def run_benchmark(
    scorer: Callable[[dict[str, Any]], float],
    iterations: int,
    warmup: int,
) -> BenchmarkResult:
    """Execute the benchmark and collect latency samples."""
    # Warmup phase
    for _ in range(warmup):
        scorer(_random_transaction())

    tracemalloc.start()
    latencies: list[float] = []
    wall_start = time.perf_counter()

    for _ in range(iterations):
        txn = _random_transaction()
        t0 = time.perf_counter()
        scorer(txn)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    wall_elapsed = time.perf_counter() - wall_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies.sort()
    result = BenchmarkResult(
        iterations=iterations,
        warmup=warmup,
        latencies_ms=latencies,
        p50_ms=round(_percentile(latencies, 50), 4),
        p90_ms=round(_percentile(latencies, 90), 4),
        p95_ms=round(_percentile(latencies, 95), 4),
        p99_ms=round(_percentile(latencies, 99), 4),
        mean_ms=round(statistics.mean(latencies), 4),
        stddev_ms=round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
        throughput_tps=round(iterations / wall_elapsed, 2),
        peak_memory_mb=round(peak_bytes / (1024 * 1024), 2),
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the scoring pipeline")
    parser.add_argument("--iterations", type=int, default=5000, help="Number of scored transactions")
    parser.add_argument("--warmup", type=int, default=500, help="Warmup iterations (not measured)")
    parser.add_argument("--output", type=str, default="", help="Path to write JSON results")
    args = parser.parse_args()

    print(f"Running benchmark: {args.iterations} iterations, {args.warmup} warmup")
    result = run_benchmark(_stub_score, args.iterations, args.warmup)

    summary = result.summary_dict()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
