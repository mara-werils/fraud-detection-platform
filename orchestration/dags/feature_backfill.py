"""Airflow DAG: daily offline feature backfill.

Schedule: daily at 4 AM UTC
Recomputes offline features in ClickHouse for user and merchant profiles.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = structlog.get_logger(__name__)

_DEFAULT_ARGS = {
    "owner": "fraud-detection",
    "depends_on_past": False,
    "email": ["fraud-alerts@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

_CLICKHOUSE_URL = "clickhouse://default:clickhouse@clickhouse:8123/fraud"


def _get_clickhouse_client():  # type: ignore[no-untyped-def]
    from clickhouse_driver import Client  # type: ignore[import-untyped]

    return Client.from_url(_CLICKHOUSE_URL)


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def backfill_user_profiles(**context: object) -> int:
    """Recompute the user_profiles aggregate table from scored transactions."""
    ch = _get_clickhouse_client()

    query = """
        INSERT INTO fraud.user_profiles
        SELECT
            user_id,
            count()                                         AS total_txn_count,
            sum(amount)                                     AS total_txn_amount,
            avg(amount)                                     AS avg_txn_amount,
            max(amount)                                     AS max_txn_amount,
            stddevPop(amount)                               AS std_txn_amount,
            uniqExact(merchant_id)                          AS unique_merchants,
            uniqExact(device_id)                            AS unique_devices,
            uniqExact(ip_address)                           AS unique_ips,
            countIf(is_flagged = 1)                         AS flagged_count,
            avg(fraud_score)                                AS avg_fraud_score,
            max(fraud_score)                                AS max_fraud_score,
            min(timestamp)                                  AS first_txn_at,
            max(timestamp)                                  AS last_txn_at,
            now()                                           AS computed_at
        FROM fraud.scored_transactions
        GROUP BY user_id
    """

    ch.execute("TRUNCATE TABLE IF EXISTS fraud.user_profiles")
    ch.execute(query)
    row_count = ch.execute("SELECT count() FROM fraud.user_profiles")[0][0]

    structlog.get_logger(__name__).info("user_profiles_backfilled", rows=row_count)
    context["ti"].xcom_push(key="user_profiles_count", value=row_count)  # type: ignore[union-attr]
    return row_count


def backfill_merchant_profiles(**context: object) -> int:
    """Recompute the merchant_profiles aggregate table."""
    ch = _get_clickhouse_client()

    query = """
        INSERT INTO fraud.merchant_profiles
        SELECT
            merchant_id,
            merchant_category,
            count()                                         AS total_txn_count,
            sum(amount)                                     AS total_txn_amount,
            avg(amount)                                     AS avg_txn_amount,
            uniqExact(user_id)                              AS unique_users,
            countIf(is_flagged = 1)                         AS flagged_count,
            avg(fraud_score)                                AS avg_fraud_score,
            max(fraud_score)                                AS max_fraud_score,
            min(timestamp)                                  AS first_txn_at,
            max(timestamp)                                  AS last_txn_at,
            now()                                           AS computed_at
        FROM fraud.scored_transactions
        WHERE merchant_id IS NOT NULL
        GROUP BY merchant_id, merchant_category
    """

    ch.execute("TRUNCATE TABLE IF EXISTS fraud.merchant_profiles")
    ch.execute(query)
    row_count = ch.execute("SELECT count() FROM fraud.merchant_profiles")[0][0]

    structlog.get_logger(__name__).info("merchant_profiles_backfilled", rows=row_count)
    context["ti"].xcom_push(key="merchant_profiles_count", value=row_count)  # type: ignore[union-attr]
    return row_count


def validate_profiles(**context: object) -> None:
    """Run basic validation on backfilled profiles."""
    ch = _get_clickhouse_client()

    user_count = ch.execute("SELECT count() FROM fraud.user_profiles")[0][0]
    merchant_count = ch.execute("SELECT count() FROM fraud.merchant_profiles")[0][0]

    if user_count == 0:
        raise ValueError("user_profiles table is empty after backfill")
    if merchant_count == 0:
        raise ValueError("merchant_profiles table is empty after backfill")

    structlog.get_logger(__name__).info(
        "profiles_validated",
        user_profiles=user_count,
        merchant_profiles=merchant_count,
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="feature_backfill",
    default_args=_DEFAULT_ARGS,
    description="Daily offline feature backfill for user and merchant profiles",
    schedule_interval="0 4 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["features", "backfill", "fraud"],
) as dag:
    t_users = PythonOperator(
        task_id="backfill_user_profiles",
        python_callable=backfill_user_profiles,
    )

    t_merchants = PythonOperator(
        task_id="backfill_merchant_profiles",
        python_callable=backfill_merchant_profiles,
    )

    t_validate = PythonOperator(
        task_id="validate_profiles",
        python_callable=validate_profiles,
    )

    [t_users, t_merchants] >> t_validate
