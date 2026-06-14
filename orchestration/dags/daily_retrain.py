"""Airflow DAG: daily model retraining pipeline.

Schedule: daily at 2 AM UTC
Tasks: export data -> feature engineering -> train XGBoost -> evaluate -> promote
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

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

_CLICKHOUSE_CONN = "clickhouse_default"
_MLFLOW_TRACKING_URI = "http://mlflow:5000"
_TRAINING_DAYS = 30


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def export_training_data(**context: object) -> str:
    """Query ClickHouse for the last 30 days of scored transactions."""
    from clickhouse_driver import Client  # type: ignore[import-untyped]

    ch = Client.from_url("clickhouse://default:clickhouse@clickhouse:8123/fraud")

    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=_TRAINING_DAYS)

    query = """
        SELECT
            transaction_id,
            user_id,
            amount,
            currency,
            transaction_type,
            merchant_id,
            merchant_category,
            ip_address,
            device_id,
            latitude,
            longitude,
            timestamp,
            fraud_score,
            is_flagged,
            COALESCE(label, 0) AS label
        FROM fraud.scored_transactions AS st
        LEFT JOIN fraud.fraud_labels AS fl
            ON st.transaction_id = fl.transaction_id
        WHERE st.timestamp >= %(start)s
          AND st.timestamp < %(end)s
    """

    rows = ch.execute(query, {"start": start_date, "end": end_date})
    output_path = f"/tmp/training_data_{end_date:%Y%m%d}.parquet"

    import pyarrow as pa
    import pyarrow.parquet as pq

    if rows:
        columns = [
            "transaction_id",
            "user_id",
            "amount",
            "currency",
            "transaction_type",
            "merchant_id",
            "merchant_category",
            "ip_address",
            "device_id",
            "latitude",
            "longitude",
            "timestamp",
            "fraud_score",
            "is_flagged",
            "label",
        ]
        table = pa.table({col: [r[i] for r in rows] for i, col in enumerate(columns)})
        pq.write_table(table, output_path)

    context["ti"].xcom_push(key="training_data_path", value=output_path)  # type: ignore[union-attr]
    context["ti"].xcom_push(key="row_count", value=len(rows))  # type: ignore[union-attr]
    return output_path


def run_feature_engineering(**context: object) -> str:
    """Compute training features from the exported data."""
    import pandas as pd

    ti = context["ti"]  # type: ignore[index]
    data_path: str = ti.xcom_pull(task_ids="export_training_data", key="training_data_path")
    df = pd.read_parquet(data_path)

    # Basic feature engineering
    df["hour_of_day"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["amount_log"] = (
        df["amount"]
        .apply(lambda x: float(x))
        .apply(pd.np.log1p if hasattr(pd, "np") else __import__("numpy").log1p)
    )
    df["is_international"] = df["currency"].apply(lambda c: 0 if c == "USD" else 1)

    features_path = data_path.replace(".parquet", "_features.parquet")
    df.to_parquet(features_path, index=False)

    ti.xcom_push(key="features_path", value=features_path)
    return features_path


def train_xgboost(**context: object) -> str:
    """Train an XGBoost model on the feature-engineered data."""
    import mlflow
    import mlflow.xgboost
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    ti = context["ti"]  # type: ignore[index]
    features_path: str = ti.xcom_pull(task_ids="run_feature_engineering", key="features_path")
    df = pd.read_parquet(features_path)

    feature_cols = [
        "amount",
        "hour_of_day",
        "day_of_week",
        "amount_log",
        "is_international",
        "latitude",
        "longitude",
    ]
    existing_cols = [c for c in feature_cols if c in df.columns]

    X = df[existing_cols].fillna(0).values.astype(np.float32)
    y = (
        df["label"].values.astype(np.int32)
        if "label" in df.columns
        else np.zeros(len(df), dtype=np.int32)
    )

    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud-detection-xgboost")

    with mlflow.start_run() as run:
        dtrain = xgb.DMatrix(X, label=y, feature_names=existing_cols)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": max(1, int((y == 0).sum() / max((y == 1).sum(), 1))),
            "tree_method": "hist",
        }

        model = xgb.train(params, dtrain, num_boost_round=200)

        mlflow.xgboost.log_model(model, "model")
        mlflow.log_params(params)
        mlflow.log_metric("training_samples", len(df))
        mlflow.log_metric("positive_ratio", float((y == 1).sum()) / max(len(y), 1))

        model_path = f"/tmp/xgb_model_{run.info.run_id}.json"
        model.save_model(model_path)

    ti.xcom_push(key="model_path", value=model_path)
    ti.xcom_push(key="mlflow_run_id", value=run.info.run_id)
    return model_path


def evaluate_model(**context: object) -> dict[str, float]:
    """Evaluate the trained model and compute key metrics."""
    import mlflow
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    ti = context["ti"]  # type: ignore[index]
    model_path: str = ti.xcom_pull(task_ids="train_xgboost", key="model_path")
    features_path: str = ti.xcom_pull(task_ids="run_feature_engineering", key="features_path")
    run_id: str = ti.xcom_pull(task_ids="train_xgboost", key="mlflow_run_id")

    df = pd.read_parquet(features_path)
    feature_cols = [
        "amount",
        "hour_of_day",
        "day_of_week",
        "amount_log",
        "is_international",
        "latitude",
        "longitude",
    ]
    existing_cols = [c for c in feature_cols if c in df.columns]

    X = df[existing_cols].fillna(0).values.astype(np.float32)
    y = (
        df["label"].values.astype(np.int32)
        if "label" in df.columns
        else np.zeros(len(df), dtype=np.int32)
    )

    model = xgb.Booster()
    model.load_model(model_path)
    dmatrix = xgb.DMatrix(X, feature_names=existing_cols)
    preds = model.predict(dmatrix)

    threshold = 0.5
    y_pred = (preds >= threshold).astype(int)

    metrics: dict[str, float] = {}
    if y.sum() > 0:
        metrics["precision"] = float(precision_score(y, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y, y_pred, zero_division=0))
        metrics["auc_roc"] = float(roc_auc_score(y, preds))
        metrics["auc_pr"] = float(average_precision_score(y, preds))
    else:
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0, "auc_pr": 0.0}

    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    ti.xcom_push(key="metrics", value=metrics)
    return metrics


def promote_model(**context: object) -> None:
    """Promote the model to production if metrics improved."""
    import mlflow
    from mlflow.tracking import MlflowClient

    ti = context["ti"]  # type: ignore[index]
    run_id: str = ti.xcom_pull(task_ids="train_xgboost", key="mlflow_run_id")
    metrics: dict[str, float] = ti.xcom_pull(task_ids="evaluate_model", key="metrics")

    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    client = MlflowClient()

    model_name = "fraud-xgboost"
    auc_pr = metrics.get("auc_pr", 0.0)

    # Check current production model performance
    should_promote = True
    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if latest_versions:
            prod_run = client.get_run(latest_versions[0].run_id)
            prod_auc = float(prod_run.data.metrics.get("auc_pr", 0.0))
            if auc_pr <= prod_auc:
                should_promote = False
                structlog.get_logger(__name__).info(
                    "model_not_promoted",
                    reason="New model did not improve over production",
                    new_auc_pr=auc_pr,
                    prod_auc_pr=prod_auc,
                )
    except Exception:
        pass  # No existing model — promote the first one

    if should_promote:
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        structlog.get_logger(__name__).info(
            "model_promoted",
            model_name=model_name,
            version=mv.version,
            auc_pr=auc_pr,
        )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="daily_retrain",
    default_args=_DEFAULT_ARGS,
    description="Daily fraud model retraining pipeline",
    schedule_interval="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "fraud", "retrain"],
) as dag:
    t_export = PythonOperator(
        task_id="export_training_data",
        python_callable=export_training_data,
    )

    t_features = PythonOperator(
        task_id="run_feature_engineering",
        python_callable=run_feature_engineering,
    )

    t_train = PythonOperator(
        task_id="train_xgboost",
        python_callable=train_xgboost,
    )

    t_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    t_export >> t_features >> t_train >> t_evaluate >> t_promote
