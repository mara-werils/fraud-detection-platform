"""Model evaluation: metrics, SHAP, calibration, latency benchmarks.

CLI::

    python -m ml_pipeline.evaluate --model models/xgboost_v1.json --data data/test.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_pipeline.data_prep import FEATURE_PREFIX, LABEL_COL
from ml_pipeline.feature_engineering import ALL_FEATURE_NAMES, engineer_features

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Returns:
        Dict with AUC-ROC, PR-AUC, precision, recall, F1, accuracy.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {
        "auc_roc": round(roc_auc_score(y_true, y_prob), 5),
        "pr_auc": round(average_precision_score(y_true, y_prob), 5),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 5),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 5),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 5),
        "accuracy": round(accuracy_score(y_true, y_pred), 5),
        "threshold": threshold,
    }
    return metrics


def precision_at_recall_levels(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_levels: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict[str, dict[str, float]]:
    """Compute precision at specified recall levels.

    Returns:
        Dict mapping recall level string to {precision, threshold}.
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    results: dict[str, dict[str, float]] = {}

    for target_recall in recall_levels:
        # Find the threshold that achieves at least this recall
        valid = recall_arr >= target_recall
        if valid.any():
            idx = np.where(valid)[0][-1]
            results[f"recall_{int(target_recall * 100)}"] = {
                "precision": round(float(precision_arr[idx]), 5),
                "threshold": round(float(thresholds[min(idx, len(thresholds) - 1)]), 5),
                "recall": round(float(recall_arr[idx]), 5),
            }
        else:
            results[f"recall_{int(target_recall * 100)}"] = {
                "precision": 0.0,
                "threshold": 0.0,
                "recall": 0.0,
            }

    return results


def compute_confusion(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, int]:
    """Compute confusion matrix values."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


# ---------------------------------------------------------------------------
# Per-fraud-type breakdown
# ---------------------------------------------------------------------------


def per_fraud_type_metrics(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by fraud_pattern.

    Args:
        df: DataFrame with ``fraud_pattern`` and ``is_fraud`` columns.
        y_prob: Predicted probabilities (aligned with df index).
        threshold: Classification threshold.

    Returns:
        Dict mapping fraud pattern to metrics.
    """
    if "fraud_pattern" not in df.columns:
        return {}

    results: dict[str, dict[str, float]] = {}
    for pattern in df["fraud_pattern"].unique():
        if pattern == "none":
            continue
        mask = df["fraud_pattern"] == pattern
        y_t = df.loc[mask, "is_fraud"].values
        y_p = y_prob[mask.values]
        if len(y_t) == 0 or y_t.sum() == 0:
            continue
        results[pattern] = compute_metrics(y_t, y_p, threshold=threshold)

    return results


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------


def compute_shap_importance(
    model: xgb.Booster,
    X: np.ndarray,
    feature_names: list[str],
    *,
    max_samples: int = 5000,
) -> dict[str, float]:
    """Compute SHAP feature importance for an XGBoost model.

    Falls back to built-in feature importance if shap is not available.
    """
    try:
        import shap

        sample_idx = np.random.choice(len(X), min(max_samples, len(X)), replace=False)
        X_sample = X[sample_idx]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        importance = np.abs(shap_values).mean(axis=0)
        result = {
            feature_names[i]: round(float(importance[i]), 6)
            for i in range(len(feature_names))
        }
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        log.info("shap_importance_computed", n_features=len(result))
        return result

    except ImportError:
        log.warning("shap_not_installed", msg="Falling back to built-in importance")
        importance = model.get_score(importance_type="gain")
        result = {}
        for i, name in enumerate(feature_names):
            key = f"f{i}"
            result[name] = round(importance.get(key, 0.0), 6)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Compute calibration curve data."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return {
        "mean_predicted_probability": [round(float(p), 5) for p in prob_pred],
        "fraction_of_positives": [round(float(p), 5) for p in prob_true],
        "n_bins": n_bins,
    }


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------


def benchmark_latency(
    model: xgb.Booster,
    X: np.ndarray,
    *,
    n_iterations: int = 1000,
) -> dict[str, float]:
    """Benchmark single-sample inference latency.

    Returns:
        Dict with p50, p95, p99 latencies in milliseconds.
    """
    # Warm up
    dmat = xgb.DMatrix(X[:1])
    for _ in range(10):
        model.predict(dmat)

    latencies: list[float] = []
    for i in range(min(n_iterations, len(X))):
        sample = xgb.DMatrix(X[i : i + 1])
        start = time.perf_counter()
        model.predict(sample)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

    arr = np.array(latencies)
    result = {
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "mean_ms": round(float(arr.mean()), 4),
        "n_iterations": len(latencies),
    }
    log.info("latency_benchmark", **result)
    return result


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------


def evaluate_model(
    model_path: str,
    data_path: str,
    *,
    threshold: float = 0.5,
    output_path: str | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive evaluation report.

    Args:
        model_path: Path to saved XGBoost model.
        data_path: Path to test Parquet file.
        threshold: Classification threshold.
        output_path: If set, save the report as JSON here.
        feature_cols: Feature column names (auto-detected if None).

    Returns:
        Full evaluation report as a dict.
    """
    log.info("evaluation_start", model=model_path, data=data_path)

    # --- Load model ---
    model = xgb.Booster()
    model.load_model(model_path)

    # --- Load and prepare data ---
    df = pd.read_parquet(data_path)
    df = engineer_features(df, feature_names=ALL_FEATURE_NAMES)

    feat_cols = feature_cols or ALL_FEATURE_NAMES
    # Ensure all feature columns exist
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feat_cols].values.astype(np.float64)
    y = df[LABEL_COL].values.astype(np.int32)

    # --- Predict ---
    dmat = xgb.DMatrix(X)
    y_prob = model.predict(dmat)

    # --- Compute everything ---
    report: dict[str, Any] = {
        "model_path": model_path,
        "data_path": data_path,
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "fraud_rate": round(float(y.mean()), 5),
        "threshold": threshold,
    }

    report["metrics"] = compute_metrics(y, y_prob, threshold=threshold)
    report["precision_at_recall"] = precision_at_recall_levels(y, y_prob)
    report["confusion_matrix"] = compute_confusion(y, y_prob, threshold=threshold)
    report["per_fraud_type"] = per_fraud_type_metrics(df, y_prob, threshold=threshold)
    report["shap_importance"] = compute_shap_importance(model, X, feat_cols)
    report["calibration"] = compute_calibration(y, y_prob)
    report["latency"] = benchmark_latency(model, X)

    log.info("evaluation_complete", auc_roc=report["metrics"]["auc_roc"], pr_auc=report["metrics"]["pr_auc"])

    # --- Save report ---
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        log.info("report_saved", path=output_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained fraud detection model")
    parser.add_argument("--model", required=True, help="Path to saved XGBoost model")
    parser.add_argument("--data", required=True, help="Path to test Parquet file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--output", default=None, help="Output JSON report path")
    args = parser.parse_args()

    output = args.output or str(Path(args.model).parent / "evaluation_report.json")
    report = evaluate_model(
        model_path=args.model,
        data_path=args.data,
        threshold=args.threshold,
        output_path=output,
    )

    # Print summary
    m = report["metrics"]
    print(f"\n{'='*50}")
    print(f"AUC-ROC:   {m['auc_roc']}")
    print(f"PR-AUC:    {m['pr_auc']}")
    print(f"Precision: {m['precision']}")
    print(f"Recall:    {m['recall']}")
    print(f"F1:        {m['f1']}")
    print(f"Accuracy:  {m['accuracy']}")
    print(f"{'='*50}")

    cm = report["confusion_matrix"]
    print(f"TP: {cm['TP']}  FP: {cm['FP']}")
    print(f"FN: {cm['FN']}  TN: {cm['TN']}")

    lat = report["latency"]
    print(f"\nLatency: p50={lat['p50_ms']:.2f}ms  p95={lat['p95_ms']:.2f}ms  p99={lat['p99_ms']:.2f}ms")
    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
