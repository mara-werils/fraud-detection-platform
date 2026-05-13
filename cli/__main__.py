"""Fraud Detection Platform CLI.

Usage:
    fraud-cli score       Score a transaction
    fraud-cli batch       Batch score from a file
    fraud-cli search      Search scored transactions
    fraud-cli cases       List fraud cases
    fraud-cli drift       Check model drift
    fraud-cli health      Check service health
    fraud-cli model       Show model info
    fraud-cli export      Export data
    fraud-cli sanctions   Screen a name
    fraud-cli api-key     Generate an API key
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx


def main() -> None:
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

    # score
    score_parser = subparsers.add_parser("score", help="Score a transaction")
    score_parser.add_argument("--user-id", required=True, help="User ID")
    score_parser.add_argument("--amount", type=float, required=True, help="Transaction amount")
    score_parser.add_argument("--currency", default="USD", help="Currency code")
    score_parser.add_argument("--merchant-id", default=None, help="Merchant ID")
    score_parser.add_argument("--type", default="purchase", help="Transaction type")

    # search
    search_parser = subparsers.add_parser("search", help="Search transactions")
    search_parser.add_argument("--user-id", default=None)
    search_parser.add_argument("--min-score", type=float, default=None)
    search_parser.add_argument("--max-score", type=float, default=None)
    search_parser.add_argument("--flagged", action="store_true", default=None)
    search_parser.add_argument("--decision", default=None, choices=["BLOCK", "REVIEW", "ALLOW"])
    search_parser.add_argument("--limit", type=int, default=20)

    # cases
    cases_parser = subparsers.add_parser("cases", help="List fraud cases")
    cases_parser.add_argument("--status", default=None)
    cases_parser.add_argument("--priority", default=None)
    cases_parser.add_argument("--limit", type=int, default=20)

    # drift
    subparsers.add_parser("drift", help="Check model drift")

    # health
    subparsers.add_parser("health", help="Check service health")

    # model
    subparsers.add_parser("model", help="Show model info")

    # export
    export_parser = subparsers.add_parser("export", help="Export data")
    export_parser.add_argument("type", choices=["transactions", "cases", "feedback"])
    export_parser.add_argument("--format", choices=["csv", "json"], default="json")
    export_parser.add_argument("--min-score", type=float, default=None)
    export_parser.add_argument("--flagged", action="store_true", default=None)

    # sanctions
    sanctions_parser = subparsers.add_parser("sanctions", help="Screen a name")
    sanctions_parser.add_argument("name", help="Name to screen")

    # stats
    subparsers.add_parser("stats", help="Show transaction statistics")

    # ab
    subparsers.add_parser("ab", help="Show A/B test results")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=30.0)

    try:
        result = dispatch(client, args)
        if args.output == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print_table(result)
    except httpx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {args.base_url}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()


def dispatch(client: httpx.Client, args: argparse.Namespace) -> Any:
    """Dispatch to the appropriate command handler."""
    if args.command == "score":
        return cmd_score(client, args)
    elif args.command == "search":
        return cmd_search(client, args)
    elif args.command == "cases":
        return cmd_cases(client, args)
    elif args.command == "drift":
        return cmd_get(client, "/api/v1/drift/report")
    elif args.command == "health":
        return cmd_get(client, "/health")
    elif args.command == "model":
        return cmd_get(client, "/model/info")
    elif args.command == "export":
        return cmd_export(client, args)
    elif args.command == "sanctions":
        return cmd_sanctions(client, args)
    elif args.command == "stats":
        return cmd_get(client, "/api/v1/transactions/stats")
    elif args.command == "ab":
        return cmd_get(client, "/api/v1/ab/results")
    else:
        return {"error": f"Unknown command: {args.command}"}


def cmd_score(client: httpx.Client, args: argparse.Namespace) -> Any:
    payload: dict[str, Any] = {
        "user_id": args.user_id,
        "amount": args.amount,
        "currency": args.currency,
        "transaction_type": args.type,
    }
    if args.merchant_id:
        payload["merchant_id"] = args.merchant_id

    resp = client.post("/score", json=payload)
    resp.raise_for_status()
    return resp.json()


def cmd_search(client: httpx.Client, args: argparse.Namespace) -> Any:
    params: dict[str, Any] = {"limit": args.limit}
    if args.user_id:
        params["user_id"] = args.user_id
    if args.min_score is not None:
        params["min_score"] = args.min_score
    if args.max_score is not None:
        params["max_score"] = args.max_score
    if args.flagged:
        params["is_flagged"] = True
    if args.decision:
        params["decision"] = args.decision

    resp = client.get("/api/v1/transactions", params=params)
    resp.raise_for_status()
    return resp.json()


def cmd_cases(client: httpx.Client, args: argparse.Namespace) -> Any:
    params: dict[str, Any] = {"limit": args.limit}
    if args.status:
        params["status"] = args.status
    if args.priority:
        params["priority"] = args.priority

    resp = client.get("/api/v1/cases", params=params)
    resp.raise_for_status()
    return resp.json()


def cmd_export(client: httpx.Client, args: argparse.Namespace) -> Any:
    payload: dict[str, Any] = {"format": args.format}
    if args.min_score is not None:
        payload["min_score"] = args.min_score
    if args.flagged:
        payload["is_flagged"] = True

    resp = client.post(f"/api/v1/export/{args.type}", json=payload)
    resp.raise_for_status()

    if args.format == "csv":
        print(resp.text)
        return {"status": "exported", "format": "csv"}
    return resp.json()


def cmd_sanctions(client: httpx.Client, args: argparse.Namespace) -> Any:
    resp = client.post("/api/v1/sanctions/check", json={"name": args.name})
    resp.raise_for_status()
    return resp.json()


def cmd_get(client: httpx.Client, path: str) -> Any:
    resp = client.get(path)
    resp.raise_for_status()
    return resp.json()


def print_table(data: Any) -> None:
    """Simple table output for terminal display."""
    if isinstance(data, dict):
        max_key = max(len(str(k)) for k in data.keys()) if data else 0
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            print(f"  {str(key):<{max_key + 2}} {value}")
    elif isinstance(data, list):
        for item in data:
            print(json.dumps(item, indent=2, default=str))
    else:
        print(data)


if __name__ == "__main__":
    main()
