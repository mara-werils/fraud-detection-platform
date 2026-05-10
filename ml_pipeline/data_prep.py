"""Data preparation: loading, splitting, balancing, and scaling.

Usage::

    from ml_pipeline.data_prep import prepare_data
    split = prepare_data("data/training_data.parquet")
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

log = structlog.get_logger(__name__)

FEATURE_PREFIX = "f_"
LABEL_COL = "is_fraud"


@dataclass
class DataSplit:
    """Container for train / validation / test splits."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    class_weights: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_dataframe(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Check for nulls, infinities, and expected distributions."""
    n_before = len(df)

    # Check nulls in feature columns
    null_counts = df[feature_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        log.warning("null_values_detected", columns=cols_with_nulls.to_dict())
        df[feature_cols] = df[feature_cols].fillna(0.0)

    # Check for infinities
    inf_mask = np.isinf(df[feature_cols].values)
    if inf_mask.any():
        n_inf = int(inf_mask.sum())
        log.warning("infinite_values_detected", count=n_inf)
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Check label column
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    unique_labels = set(df[LABEL_COL].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Label column must be binary (0/1), got: {unique_labels}")

    fraud_rate = df[LABEL_COL].mean()
    log.info(
        "data_validated",
        rows=len(df),
        dropped=n_before - len(df),
        fraud_rate=round(fraud_rate, 4),
        n_features=len(feature_cols),
    )

    return df


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute class weights inversely proportional to class frequency."""
    n_samples = len(y)
    n_pos = int(y.sum())
    n_neg = n_samples - n_pos
    weight_neg = n_samples / (2.0 * max(n_neg, 1))
    weight_pos = n_samples / (2.0 * max(n_pos, 1))
    return {0: round(weight_neg, 4), 1: round(weight_pos, 4)}


def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to the minority class.

    Falls back gracefully if imblearn is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=random_state, n_jobs=-1)
        X_res, y_res = smote.fit_resample(X, y)
        log.info(
            "smote_applied",
            original_size=len(y),
            resampled_size=len(y_res),
            original_fraud_rate=round(float(y.mean()), 4),
            resampled_fraud_rate=round(float(y_res.mean()), 4),
        )
        return X_res, y_res
    except ImportError:
        log.warning("imblearn_not_installed", msg="Skipping SMOTE; install imbalanced-learn")
        return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_features(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load a Parquet file and return (df, feature_column_names)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_parquet(path)
    feature_cols = sorted([c for c in df.columns if c.startswith(FEATURE_PREFIX)])
    if not feature_cols:
        raise ValueError("No feature columns found (expected prefix 'f_')")

    log.info("data_loaded", path=str(path), rows=len(df), features=len(feature_cols))
    return df, feature_cols


def prepare_data(
    path: str | Path,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    apply_smote: bool = True,
    scale: bool = True,
    random_state: int = 42,
    feature_cols: Sequence[str] | None = None,
) -> DataSplit:
    """Full data preparation pipeline.

    Steps:
        1. Load Parquet
        2. Validate (nulls, inf, distributions)
        3. Stratified 70/15/15 split
        4. Fit scaler on train only (no leakage)
        5. SMOTE on train only
        6. Return DataSplit

    Args:
        path: Path to the training Parquet file.
        test_size: Fraction for test set.
        val_size: Fraction for validation set.
        apply_smote: Whether to apply SMOTE oversampling on train.
        scale: Whether to apply StandardScaler.
        random_state: Random seed.
        feature_cols: Override feature column list (auto-detected by default).

    Returns:
        A fully prepared ``DataSplit``.
    """
    df, auto_features = load_features(path)
    feat_cols = list(feature_cols) if feature_cols is not None else auto_features
    df = _validate_dataframe(df, feat_cols)

    X = df[feat_cols].values.astype(np.float64)
    y = df[LABEL_COL].values.astype(np.int32)

    # --- Stratified split: train+val vs test ---
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # --- Split train+val into train and val ---
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        stratify=y_trainval,
        random_state=random_state,
    )

    log.info(
        "data_split",
        train=len(y_train),
        val=len(y_val),
        test=len(y_test),
        train_fraud_rate=round(float(y_train.mean()), 4),
    )

    # --- Scaling (fit on train only) ---
    scaler = StandardScaler()
    if scale:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        log.info("scaler_fit", n_features=X_train.shape[1])
    else:
        scaler.fit(X_train)  # fit anyway so it's available for inference

    # --- Class weights ---
    class_weights = _compute_class_weights(y_train)
    log.info("class_weights", weights=class_weights)

    # --- SMOTE on train only ---
    if apply_smote:
        X_train, y_train = _apply_smote(X_train, y_train, random_state=random_state)

    return DataSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feat_cols,
        scaler=scaler,
        class_weights=class_weights,
    )
