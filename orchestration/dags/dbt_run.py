"""Airflow DAG: dbt model execution pipeline.

Schedule: every 6 hours
Tasks: dbt run (staging -> intermediate -> marts) -> dbt test -> dbt docs generate
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

_DEFAULT_ARGS = {
    "owner": "fraud-detection",
    "depends_on_past": False,
    "email": ["fraud-alerts@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

_DBT_PROJECT_DIR = "/opt/airflow/dbt_project"
_DBT_PROFILES_DIR = "/opt/airflow/dbt_project"

_DBT_BASE_CMD = f"cd {_DBT_PROJECT_DIR} && dbt --profiles-dir {_DBT_PROFILES_DIR}"

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="dbt_run",
    default_args=_DEFAULT_ARGS,
    description="Run dbt models, tests, and docs on a 6-hour schedule",
    schedule_interval="0 */6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["dbt", "analytics", "fraud"],
) as dag:
    t_staging = BashOperator(
        task_id="dbt_run_staging",
        bash_command=f"{_DBT_BASE_CMD} run --select staging.*",
    )

    t_intermediate = BashOperator(
        task_id="dbt_run_intermediate",
        bash_command=f"{_DBT_BASE_CMD} run --select intermediate.*",
    )

    t_marts = BashOperator(
        task_id="dbt_run_marts",
        bash_command=f"{_DBT_BASE_CMD} run --select marts.*",
    )

    t_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"{_DBT_BASE_CMD} test",
    )

    t_docs = BashOperator(
        task_id="dbt_docs_generate",
        bash_command=f"{_DBT_BASE_CMD} docs generate",
    )

    t_staging >> t_intermediate >> t_marts >> t_test >> t_docs
