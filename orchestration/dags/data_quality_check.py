"""Airflow DAG: hourly data quality checks.

Schedule: every hour
Checks: null rate, fraud rate anomaly, data freshness, schema validation
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_ARGS = {
    "owner": "fraud-detection",
    "depends_on_past": False,
    "email": ["fraud-alerts@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

_CLICKHOUSE_URL = "clickhouse://default:clickhouse@clickhouse:8123/fraud"

# Thresholds
_MAX_NULL_RATE = 0.05  # 5 %
_FRAUD_RATE_MIN = 0.001  # 0.1 %
_FRAUD_RATE_MAX = 0.10  # 10 %
_MAX_DATA_AGE_SECONDS = 600  # 10 minutes


def _get_clickhouse_client():  # type: ignore[no-untyped-def]
    from clickhouse_driver import Client  # type: ignore[import-untyped]
    return Client.from_url(_CLICKHOUSE_URL)


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def check_null_rates(**context: object) -> dict[str, float]:
    """Check null rates for critical columns in scored_transactions."""
    ch = _get_clickhouse_client()

    one_hour_ago = datetime.utcnow() - timedelta(hours=1)

    columns = [
        "transaction_id", "user_id", "amount", "fraud_score",
        "model_version", "timestamp",
    ]

    results: dict[str, float] = {}
    violations: list[str] = []

    total = ch.execute(
        "SELECT count() FROM fraud.scored_transactions WHERE timestamp >= %(since)s",
        {"since": one_hour_ago},
    )[0][0]

    if total == 0:
        structlog.get_logger(__name__).warning("no_recent_data", since=one_hour_ago.isoformat())
        return results

    for col in columns:
        null_count = ch.execute(
            f"SELECT countIf({col} IS NULL) FROM fraud.scored_transactions WHERE timestamp >= %(since)s",
            {"since": one_hour_ago},
        )[0][0]
        rate = null_count / total
        results[col] = rate
        if rate > _MAX_NULL_RATE:
            violations.append(f"{col}: {rate:.2%} null (threshold {_MAX_NULL_RATE:.2%})")

    if violations:
        raise ValueError(f"Null rate violations: {'; '.join(violations)}")

    structlog.get_logger(__name__).info("null_rate_check_passed", rates=results)
    return results


def check_fraud_rate_anomaly(**context: object) -> float:
    """Detect anomalous fraud flagging rates in the last hour."""
    ch = _get_clickhouse_client()

    one_hour_ago = datetime.utcnow() - timedelta(hours=1)

    row = ch.execute(
        """
        SELECT
            count()                              AS total,
            countIf(is_flagged = 1)              AS flagged
        FROM fraud.scored_transactions
        WHERE timestamp >= %(since)s
        """,
        {"since": one_hour_ago},
    )[0]

    total, flagged = row[0], row[1]
    if total == 0:
        structlog.get_logger(__name__).warning("no_recent_data_for_fraud_rate")
        return 0.0

    fraud_rate = flagged / total

    if fraud_rate < _FRAUD_RATE_MIN:
        raise ValueError(
            f"Fraud rate suspiciously low: {fraud_rate:.4%} "
            f"(min threshold {_FRAUD_RATE_MIN:.4%})"
        )
    if fraud_rate > _FRAUD_RATE_MAX:
        raise ValueError(
            f"Fraud rate suspiciously high: {fraud_rate:.4%} "
            f"(max threshold {_FRAUD_RATE_MAX:.4%})"
        )

    structlog.get_logger(__name__).info("fraud_rate_check_passed", fraud_rate=fraud_rate)
    return fraud_rate


def check_data_freshness(**context: object) -> float:
    """Ensure data is arriving within acceptable latency."""
    ch = _get_clickhouse_client()

    row = ch.execute(
        "SELECT max(timestamp) FROM fraud.scored_transactions"
    )

    if not row or row[0][0] is None:
        raise ValueError("No data found in scored_transactions table")

    latest_ts: datetime = row[0][0]
    age_seconds = (datetime.utcnow() - latest_ts).total_seconds()

    if age_seconds > _MAX_DATA_AGE_SECONDS:
        raise ValueError(
            f"Data is stale: latest record is {age_seconds:.0f}s old "
            f"(threshold {_MAX_DATA_AGE_SECONDS}s)"
        )

    structlog.get_logger(__name__).info(
        "data_freshness_check_passed",
        age_seconds=round(age_seconds, 1),
    )
    return age_seconds


def check_schema_validation(**context: object) -> None:
    """Validate that the scored_transactions table schema matches expectations."""
    ch = _get_clickhouse_client()

    expected_columns = {
        "transaction_id",
        "user_id",
        "amount",
        "currency",
        "transaction_type",
        "timestamp",
        "fraud_score",
        "model_version",
        "scoring_latency_ms",
        "scored_at",
        "is_flagged",
    }

    rows = ch.execute(
        "SELECT name FROM system.columns WHERE database = 'fraud' AND table = 'scored_transactions'"
    )
    actual_columns = {r[0] for r in rows}

    missing = expected_columns - actual_columns
    if missing:
        raise ValueError(f"Missing columns in scored_transactions: {missing}")

    structlog.get_logger(__name__).info(
        "schema_validation_passed",
        expected=len(expected_columns),
        actual=len(actual_columns),
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="data_quality_check",
    default_args=_DEFAULT_ARGS,
    description="Hourly data quality checks for the fraud detection pipeline",
    schedule_interval="0 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["data-quality", "monitoring", "fraud"],
) as dag:
    t_null_rates = PythonOperator(
        task_id="check_null_rates",
        python_callable=check_null_rates,
    )

    t_fraud_rate = PythonOperator(
        task_id="check_fraud_rate_anomaly",
        python_callable=check_fraud_rate_anomaly,
    )

    t_freshness = PythonOperator(
        task_id="check_data_freshness",
        python_callable=check_data_freshness,
    )

    t_schema = PythonOperator(
        task_id="check_schema_validation",
        python_callable=check_schema_validation,
    )

    # All checks run in parallel
    [t_null_rates, t_fraud_rate, t_freshness, t_schema]
