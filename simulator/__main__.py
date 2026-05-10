"""CLI entry point for the transaction simulator.

Usage:
    python -m simulator --rate 50 --fraud-ratio 0.05 --duration 300
"""

from __future__ import annotations

import argparse
import asyncio


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="simulator",
        description="Generate realistic financial transactions and publish them to Kafka.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=10,
        help="Target transactions per second (default: 10).",
    )
    parser.add_argument(
        "--fraud-ratio",
        type=float,
        default=0.02,
        help="Fraction of transactions that are fraudulent, 0.0-1.0 (default: 0.02).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Total seconds to run. 0 means run until interrupted (default: 0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and launch the async simulator loop."""
    args = _parse_args(argv)

    from simulator.main import run_simulator

    try:
        asyncio.run(
            run_simulator(
                rate=args.rate,
                fraud_ratio=args.fraud_ratio,
                duration=args.duration,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
