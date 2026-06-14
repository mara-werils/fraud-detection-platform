"""Enhanced CLI commands for the Fraud Detection Platform.

Provides subcommands for scoring, rule management, case workflows,
simulation, health checks, statistics, and data export.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path
from typing import Any

import httpx
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(args: argparse.Namespace) -> httpx.Client:
    """Build an httpx client from shared CLI flags."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if getattr(args, "api_key", None):
        headers["X-API-Key"] = args.api_key
    return httpx.Client(base_url=args.base_url, headers=headers, timeout=30.0)


def _output(data: Any, args: argparse.Namespace) -> None:
    """Print *data* in the chosen output format."""
    fmt = getattr(args, "output", "json")
    if fmt == "table":
        _print_table(data)
    else:
        print(json.dumps(data, indent=2, default=str))


def _print_table(data: Any) -> None:
    """Render a dict or list-of-dicts as an aligned text table."""
    if isinstance(data, dict):
        width = max((len(str(k)) for k in data), default=0)
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            print(f"  {str(key):<{width + 2}} {value}")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        cols = list(data[0].keys())
        widths = {c: max(len(c), *(len(str(row.get(c, ""))) for row in data)) for c in cols}
        header = "  ".join(f"{c:<{widths[c]}}" for c in cols)
        print(header)
        print("-" * len(header))
        for row in data:
            print("  ".join(f"{str(row.get(c, '')):<{widths[c]}}" for c in cols))
    elif isinstance(data, list):
        for item in data:
            print(json.dumps(item, indent=2, default=str))
    else:
        print(data)


def _load_file(path: str) -> tuple[str, Any]:
    """Load a JSON or CSV file and return ``(format, data)``."""
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix == ".json":
        return "json", json.loads(text)
    elif suffix in {".csv", ".tsv"}:
        reader = csv.DictReader(io.StringIO(text))
        return "csv", list(reader)
    elif suffix in {".yaml", ".yml"}:
        return "yaml", yaml.safe_load(text)
    else:
        print(f"Error: unsupported file format '{suffix}'", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI builder
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the full ``fraud-cli`` argument parser."""

    parser = argparse.ArgumentParser(
        prog="fraud-cli",
        description="Fraud Detection Platform CLI",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Scoring service URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication",
    )
    parser.add_argument(
        "--output",
        choices=["json", "table"],
        default="json",
        help="Output format (default: json)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- score ---------------------------------------------------------------
    score_p = subparsers.add_parser("score", help="Score a single transaction")
    score_p.add_argument(
        "transaction_json",
        help="Transaction as inline JSON string or path to a .json file",
    )

    # -- batch-score ---------------------------------------------------------
    batch_p = subparsers.add_parser("batch-score", help="Batch score from CSV/JSON file")
    batch_p.add_argument("file", help="Path to CSV or JSON file with transactions")
    batch_p.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent scoring requests (default: 5)",
    )

    # -- rules ---------------------------------------------------------------
    rules_p = subparsers.add_parser("rules", help="Manage fraud detection rules")
    rules_sub = rules_p.add_subparsers(dest="rules_action", help="Rule actions")

    rules_sub.add_parser("list", help="List all active rules")

    rules_validate_p = rules_sub.add_parser("validate", help="Validate a rule YAML file")
    rules_validate_p.add_argument("file", help="Path to the rule YAML file to validate")

    # -- cases ---------------------------------------------------------------
    cases_p = subparsers.add_parser("cases", help="Manage fraud cases")
    cases_sub = cases_p.add_subparsers(dest="cases_action", help="Case actions")

    cases_list_p = cases_sub.add_parser("list", help="List cases with filters")
    cases_list_p.add_argument("--status", default=None, help="Filter by status (open, closed, in_review)")
    cases_list_p.add_argument("--priority", default=None, help="Filter by priority (low, medium, high, critical)")
    cases_list_p.add_argument("--assigned-to", default=None, help="Filter by assignee")
    cases_list_p.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")

    # -- simulate ------------------------------------------------------------
    sim_p = subparsers.add_parser("simulate", help="Run a fraud simulation scenario")
    sim_p.add_argument(
        "--scenario",
        required=True,
        choices=[
            "card_testing",
            "account_takeover",
            "friendly_fraud",
            "bust_out",
            "synthetic_identity",
        ],
        help="Simulation scenario to execute",
    )
    sim_p.add_argument(
        "--transactions",
        type=int,
        default=100,
        help="Number of transactions to generate (default: 100)",
    )
    sim_p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # -- health --------------------------------------------------------------
    subparsers.add_parser("health", help="Check all service health endpoints")

    # -- stats ---------------------------------------------------------------
    subparsers.add_parser("stats", help="Show platform statistics")

    # -- export --------------------------------------------------------------
    export_p = subparsers.add_parser("export", help="Export platform data")
    export_p.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        dest="export_format",
        help="Export file format (default: csv)",
    )
    export_p.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of data to export (default: 30)",
    )
    export_p.add_argument(
        "--type",
        choices=["transactions", "cases", "feedback", "rules"],
        default="transactions",
        dest="export_type",
        help="Data type to export (default: transactions)",
    )
    export_p.add_argument("--out", default=None, help="Output file path (default: stdout)")

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_score(client: httpx.Client, args: argparse.Namespace) -> Any:
    """Score a single transaction from inline JSON or a file path."""
    raw = args.transaction_json
    path = Path(raw)
    if path.exists() and path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Error: invalid JSON: {raw}", file=sys.stderr)
            sys.exit(1)

    resp = client.post("/score", json=payload)
    resp.raise_for_status()
    return resp.json()


def cmd_batch_score(client: httpx.Client, args: argparse.Namespace) -> Any:
    """Batch-score transactions from a CSV or JSON file."""
    fmt, data = _load_file(args.file)

    if isinstance(data, dict):
        # JSON file might wrap rows in a key like "transactions"
        data = data.get("transactions", [data])

    results: list[dict[str, Any]] = []
    total = len(data)
    for idx, txn in enumerate(data, start=1):
        # Coerce CSV string values where needed
        if fmt == "csv":
            if "amount" in txn:
                txn["amount"] = float(txn["amount"])

        try:
            resp = client.post("/score", json=txn)
            resp.raise_for_status()
            result = resp.json()
        except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
            result = {"error": str(exc), "transaction": txn}

        results.append(result)
        print(f"\rScoring {idx}/{total}...", end="", file=sys.stderr)

    print(file=sys.stderr)  # newline after progress
    return {"total": total, "results": results}


def cmd_rules_list(client: httpx.Client, _args: argparse.Namespace) -> Any:
    """List all active fraud detection rules."""
    resp = client.get("/api/v1/rules")
    resp.raise_for_status()
    return resp.json()


def cmd_rules_validate(client: httpx.Client, args: argparse.Namespace) -> Any:
    """Validate a rule definition YAML file."""
    fmt, data = _load_file(args.file)
    if fmt not in ("yaml",):
        print("Error: rule file must be YAML (.yaml / .yml)", file=sys.stderr)
        sys.exit(1)

    errors: list[str] = []

    # Basic structural validation
    if not isinstance(data, dict):
        errors.append("Root element must be a mapping")
    else:
        required_keys = {"name", "conditions"}
        missing = required_keys - set(data.keys())
        if missing:
            errors.append(f"Missing required keys: {', '.join(sorted(missing))}")

        if "conditions" in data and not isinstance(data["conditions"], list):
            errors.append("'conditions' must be a list")

        if "action" in data and data["action"] not in {"block", "review", "flag", "allow"}:
            errors.append(
                f"Invalid action '{data['action']}'; must be one of: block, review, flag, allow"
            )

        if "threshold" in data:
            try:
                val = float(data["threshold"])
                if not 0.0 <= val <= 1.0:
                    errors.append("'threshold' must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("'threshold' must be a number")

    # Optionally validate via the API if the server supports it
    try:
        resp = client.post("/api/v1/rules/validate", json=data)
        resp.raise_for_status()
        server_result = resp.json()
        if server_result.get("errors"):
            errors.extend(server_result["errors"])
    except (httpx.HTTPStatusError, httpx.ConnectError):
        # Server validation unavailable; rely on local checks only
        pass

    if errors:
        return {"valid": False, "errors": errors}
    return {"valid": True, "rule": data.get("name", "<unnamed>"), "message": "Rule is valid"}


def cmd_cases_list(client: httpx.Client, args: argparse.Namespace) -> Any:
    """List fraud cases with optional filters."""
    params: dict[str, Any] = {"limit": args.limit}
    if args.status:
        params["status"] = args.status
    if args.priority:
        params["priority"] = args.priority
    if args.assigned_to:
        params["assigned_to"] = args.assigned_to

    resp = client.get("/api/v1/cases", params=params)
    resp.raise_for_status()
    return resp.json()


def cmd_simulate(client: httpx.Client, args: argparse.Namespace) -> Any:
    """Run a fraud simulation scenario."""
    payload: dict[str, Any] = {
        "scenario": args.scenario,
        "num_transactions": args.transactions,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    resp = client.post("/api/v1/simulate", json=payload)
    resp.raise_for_status()
    return resp.json()


def cmd_health(client: httpx.Client, _args: argparse.Namespace) -> Any:
    """Check health of all platform services."""
    services = {
        "scoring": "/health",
        "liveness": "/health/live",
        "readiness": "/health/ready",
        "model": "/model/info",
    }

    report: dict[str, Any] = {}
    for name, path in services.items():
        try:
            resp = client.get(path)
            resp.raise_for_status()
            report[name] = {"status": "healthy", "details": resp.json()}
        except httpx.HTTPStatusError as exc:
            report[name] = {"status": "unhealthy", "code": exc.response.status_code}
        except httpx.ConnectError:
            report[name] = {"status": "unreachable"}

    healthy = sum(1 for v in report.values() if v["status"] == "healthy")
    report["summary"] = f"{healthy}/{len(services)} services healthy"
    return report


def cmd_stats(client: httpx.Client, _args: argparse.Namespace) -> Any:
    """Aggregate platform statistics from multiple endpoints."""
    endpoints = {
        "transactions": "/api/v1/transactions/stats",
        "cases": "/api/v1/cases/stats",
        "model": "/model/info",
    }

    stats: dict[str, Any] = {}
    for name, path in endpoints.items():
        try:
            resp = client.get(path)
            resp.raise_for_status()
            stats[name] = resp.json()
        except (httpx.HTTPStatusError, httpx.ConnectError):
            stats[name] = {"error": "unavailable"}

    return stats


def cmd_export(client: httpx.Client, args: argparse.Namespace) -> Any:
    """Export platform data for a given time range."""
    since = datetime.now(tz=UTC) - timedelta(days=args.days)
    payload: dict[str, Any] = {
        "format": args.export_format,
        "since": since.isoformat(),
    }

    resp = client.post(f"/api/v1/export/{args.export_type}", json=payload)
    resp.raise_for_status()

    if args.out:
        Path(args.out).write_text(resp.text, encoding="utf-8")
        return {"status": "exported", "file": args.out, "format": args.export_format}

    if args.export_format == "csv":
        print(resp.text)
        return {"status": "exported", "format": "csv", "rows": resp.text.count("\n")}

    return resp.json()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(client: httpx.Client, args: argparse.Namespace) -> Any:
    """Route the parsed arguments to the correct handler."""
    cmd = args.command

    if cmd == "score":
        return cmd_score(client, args)
    elif cmd == "batch-score":
        return cmd_batch_score(client, args)
    elif cmd == "rules":
        action = getattr(args, "rules_action", None)
        if action == "list":
            return cmd_rules_list(client, args)
        elif action == "validate":
            return cmd_rules_validate(client, args)
        else:
            print("Usage: fraud-cli rules {list,validate}", file=sys.stderr)
            sys.exit(1)
    elif cmd == "cases":
        action = getattr(args, "cases_action", None)
        if action == "list":
            return cmd_cases_list(client, args)
        else:
            print("Usage: fraud-cli cases {list}", file=sys.stderr)
            sys.exit(1)
    elif cmd == "simulate":
        return cmd_simulate(client, args)
    elif cmd == "health":
        return cmd_health(client, args)
    elif cmd == "stats":
        return cmd_stats(client, args)
    elif cmd == "export":
        return cmd_export(client, args)
    else:
        return {"error": f"Unknown command: {cmd}"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the requested command."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = _make_client(args)

    try:
        result = dispatch(client, args)
        _output(result, args)
    except httpx.HTTPStatusError as exc:
        print(f"Error {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: cannot connect to {args.base_url}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
