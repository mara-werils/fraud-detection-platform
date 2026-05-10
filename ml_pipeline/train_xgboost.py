"""XGBoost training with Optuna hyper-parameter optimisation.

CLI::

    python -m ml_pipeline.train_xgboost --data data/training.parquet --output models/xgboost_v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from ml_pipeline.data_prep import prepare_data
from ml_pipeline.feature_engineering import ALL_FEATURE_NAMES, engineer_features
from ml_pipeline.hyperopt_search import (
    create_study,
    export_best_params,
    log_study_summary,
    save_visualisations,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    n_folds: int = 5,
) -> Any:
    """Return an Optuna objective that optimises PR-AUC via cross-validation."""
    import optuna

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "verbosity": 0,
            "nthread": -1,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
        }

        n_estimators = params.pop("n_estimators")

        # 5-fold stratified cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val)

            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            preds = bst.predict(dval)
            pr_auc = average_precision_score(y_fold_val, preds)
            cv_scores.append(pr_auc)

            # Report intermediate value for pruning
            trial.report(pr_auc, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_pr_auc = float(np.mean(cv_scores))
        log.info(
            "trial_complete",
            trial=trial.number,
            pr_auc=round(mean_pr_auc, 5),
            std=round(float(np.std(cv_scores)), 5),
        )
        return mean_pr_auc

    return objective


# ---------------------------------------------------------------------------
# Final model training
# ---------------------------------------------------------------------------


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: dict[str, Any],
    scale_pos_weight: float,
) -> xgb.Booster:
    """Train the final XGBoost model with the best hyper-parameters."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "verbosity": 1,
        "nthread": -1,
        "scale_pos_weight": scale_pos_weight,
    }
    # Merge best params, separating n_estimators
    n_estimators = best_params.pop("n_estimators", 500)
    params.update(best_params)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=int(n_estimators),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    val_preds = bst.predict(dval)
    val_pr_auc = average_precision_score(y_val, val_preds)
    log.info("final_model_trained", val_pr_auc=round(val_pr_auc, 5), best_iteration=bst.best_iteration)

    return bst


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def run_training(
    data_path: str,
    output_path: str,
    *,
    n_trials: int = 50,
    n_folds: int = 5,
    use_mlflow: bool = True,
) -> dict[str, Any]:
    """Full XGBoost training pipeline.

    1. Load and prepare data
    2. Engineer features
    3. Optuna hyper-parameter search (PR-AUC)
    4. Train final model with best params
    5. Save model + log to MLflow

    Args:
        data_path: Path to training Parquet.
        output_path: Path to save the model JSON.
        n_trials: Optuna search trials.
        n_folds: Cross-validation folds.
        use_mlflow: Whether to log to MLflow.

    Returns:
        Summary dict with metrics and paths.
    """
    import pandas as pd

    log.info("pipeline_start", data=data_path, output=output_path, n_trials=n_trials)
    start_time = time.time()

    # --- Load raw data and engineer features ---
    raw_df = pd.read_parquet(data_path)
    df = engineer_features(raw_df, feature_names=ALL_FEATURE_NAMES)

    # --- Prepare splits ---
    split = prepare_data(
        data_path,
        apply_smote=True,
        scale=False,  # XGBoost doesn't need scaling
        feature_cols=ALL_FEATURE_NAMES,
    )

    # --- Scale pos weight ---
    n_neg = int((split.y_train == 0).sum())
    n_pos = int((split.y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    log.info("class_balance", n_neg=n_neg, n_pos=n_pos, scale_pos_weight=round(scale_pos_weight, 2))

    # --- Optuna search ---
    study = create_study("xgboost-fraud", direction="maximize")
    objective = _make_objective(
        split.X_train,
        split.y_train,
        split.X_val,
        split.y_val,
        scale_pos_weight=scale_pos_weight,
        n_folds=n_folds,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    summary = log_study_summary(study)

    # Export best params
    params_path = Path(output_path).parent / "best_params.json"
    export_best_params(study, params_path)

    # Save visualisations
    viz_dir = Path(output_path).parent / "optuna_plots"
    save_visualisations(study, viz_dir)

    # --- Train final model ---
    best_params = dict(study.best_params)
    bst = train_final_model(
        split.X_train,
        split.y_train,
        split.X_val,
        split.y_val,
        best_params=dict(best_params),  # copy to avoid mutation
        scale_pos_weight=scale_pos_weight,
    )

    # --- Save model ---
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    bst.save_model(output_path)
    log.info("model_saved", path=output_path)

    # --- Evaluate on test set ---
    dtest = xgb.DMatrix(split.X_test)
    test_preds = bst.predict(dtest)
    test_pr_auc = average_precision_score(split.y_test, test_preds)

    elapsed = time.time() - start_time
    result = {
        "model_path": output_path,
        "test_pr_auc": round(test_pr_auc, 5),
        "best_params": study.best_params,
        "best_trial_pr_auc": round(study.best_value, 5),
        "n_trials": len(study.trials),
        "feature_names": split.feature_names,
        "training_time_s": round(elapsed, 1),
    }

    # --- MLflow logging ---
    if use_mlflow:
        try:
            from ml_pipeline.mlflow_registry import (
                init_experiment,
                log_artifact,
                log_metrics,
                log_model,
                log_params,
                start_run,
            )

            init_experiment("fraud-xgboost")
            with start_run(run_name="xgboost-optuna"):
                log_params(study.best_params)
                log_params({"n_trials": n_trials, "n_folds": n_folds, "scale_pos_weight": scale_pos_weight})
                log_metrics({"test_pr_auc": test_pr_auc, "val_pr_auc": study.best_value})
                log_artifact(output_path)
                log_artifact(str(params_path))
                log_model(bst, "xgboost-model", model_type="xgboost", registered_name="fraud-xgboost")
            log.info("mlflow_logged")
        except Exception:
            log.warning("mlflow_logging_failed", exc_info=True)

    log.info("pipeline_complete", **result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost fraud detection model")
    parser.add_argument("--data", required=True, help="Path to training Parquet file")
    parser.add_argument("--output", default="models/xgboost_v1.json", help="Output model path")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials (default: 50)")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds (default: 5)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    run_training(
        data_path=args.data,
        output_path=args.output,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
