"""XGBoost-based fraud scoring model with lazy loading and fallback."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from prometheus_client import Histogram

from shared.schemas import FeatureVector, Transaction

from scoring.models.rule_engine import RuleEngine, ScoringResult

logger = structlog.get_logger(__name__)

XGBOOST_PREDICT_LATENCY = Histogram(
    "xgboost_predict_latency_seconds",
    "Latency of XGBoost predict calls",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

# Ordered feature names expected by the XGBoost model.
# Must match the training feature order exactly.
XGBOOST_FEATURE_ORDER: tuple[str, ...] = (
    "txn_count_1h",
    "txn_count_24h",
    "txn_amount_sum_1h",
    "txn_amount_sum_24h",
    "txn_amount_avg_24h",
    "unique_merchants_24h",
    "unique_countries_24h",
    "max_amount_24h",
    "time_since_last_txn_seconds",
    "distance_from_last_txn_km",
    "is_new_device",
    "is_new_merchant",
    "amount_zscore",
    "amount_to_avg_ratio",
)


def _feature_vector_to_array(features: FeatureVector) -> np.ndarray:
    """Convert a FeatureVector to a numpy array in the canonical feature order.

    Args:
        features: The feature vector to convert.

    Returns:
        1-D numpy float32 array with values in XGBOOST_FEATURE_ORDER.
    """
    values: list[float] = []
    for name in XGBOOST_FEATURE_ORDER:
        raw = getattr(features, name)
        # Convert Decimal / bool to float
        values.append(float(raw))
    return np.array(values, dtype=np.float32)


class XGBoostScorer:
    """XGBoost fraud scoring model with lazy loading, warm-up, and fallback.

    Thread-safe: XGBoost's ``predict`` is safe for concurrent calls once the
    model (Booster) is loaded.

    Args:
        model_path: Local filesystem path to a saved XGBoost model
            (``Booster.save_model`` format).  Mutually exclusive with
            ``mlflow_model_uri``.
        mlflow_model_uri: MLflow model URI (e.g. ``models:/fraud-xgb/Production``).
        model_version: Human-readable version string for metrics/logging.
        fallback_rule_engine: Optional RuleEngine used when the model cannot
            be loaded.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        mlflow_model_uri: str | None = None,
        model_version: str = "xgboost-v1",
        fallback_rule_engine: RuleEngine | None = None,
    ) -> None:
        if model_path is None and mlflow_model_uri is None:
            raise ValueError("Either model_path or mlflow_model_uri must be provided")

        self._model_path = Path(model_path) if model_path else None
        self._mlflow_model_uri = mlflow_model_uri
        self._model_version = model_version
        self._fallback_rule_engine = fallback_rule_engine or RuleEngine()

        self._model: Any | None = None  # xgboost.Booster
        self._load_lock = threading.Lock()
        self._loaded = False
        self._load_error: Exception | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the XGBoost model from disk or MLflow (called once).

        Raises:
            RuntimeError: If model loading fails from all sources.
        """
        with self._load_lock:
            if self._loaded:
                return

            try:
                if self._model_path is not None:
                    import xgboost as xgb

                    booster = xgb.Booster()
                    booster.load_model(str(self._model_path))
                    self._model = booster
                    logger.info(
                        "xgboost_model_loaded",
                        source="file",
                        path=str(self._model_path),
                        version=self._model_version,
                    )
                elif self._mlflow_model_uri is not None:
                    import mlflow.xgboost  # type: ignore[import-untyped]

                    self._model = mlflow.xgboost.load_model(self._mlflow_model_uri)
                    logger.info(
                        "xgboost_model_loaded",
                        source="mlflow",
                        uri=self._mlflow_model_uri,
                        version=self._model_version,
                    )

                self._loaded = True
                self._load_error = None
            except Exception as exc:
                self._load_error = exc
                logger.error(
                    "xgboost_model_load_failed",
                    error=str(exc),
                    version=self._model_version,
                )
                raise RuntimeError(f"Failed to load XGBoost model: {exc}") from exc

    def _ensure_model(self) -> bool:
        """Ensure the model is loaded; return True on success."""
        if self._loaded:
            return True
        try:
            self._load_model()
            return True
        except RuntimeError:
            return False

    def warmup(self) -> None:
        """Run a dummy prediction to trigger any JIT compilation.

        Safe to call multiple times; will load the model if needed.
        """
        if not self._ensure_model():
            logger.warning("xgboost_warmup_skipped", reason="model_not_loaded")
            return

        import xgboost as xgb

        dummy = np.zeros((1, len(XGBOOST_FEATURE_ORDER)), dtype=np.float32)
        dmatrix = xgb.DMatrix(dummy, feature_names=list(XGBOOST_FEATURE_ORDER))
        self._model.predict(dmatrix)  # type: ignore[union-attr]
        logger.info("xgboost_warmup_complete", version=self._model_version)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, features: FeatureVector) -> float:
        """Return fraud probability in [0, 1].

        Falls back to the rule engine if the model is unavailable.

        Args:
            features: Pre-computed feature vector.

        Returns:
            Float fraud probability.
        """
        if not self._ensure_model():
            return self._fallback_score(features)

        try:
            return self._predict(features)
        except Exception as exc:
            logger.error(
                "xgboost_predict_error",
                error=str(exc),
                version=self._model_version,
            )
            return self._fallback_score(features)

    def _predict(self, features: FeatureVector) -> float:
        """Run XGBoost prediction with latency tracking."""
        import xgboost as xgb

        arr = _feature_vector_to_array(features)
        dmatrix = xgb.DMatrix(
            arr.reshape(1, -1), feature_names=list(XGBOOST_FEATURE_ORDER)
        )

        start = time.perf_counter()
        preds = self._model.predict(dmatrix)  # type: ignore[union-attr]
        elapsed = time.perf_counter() - start

        XGBOOST_PREDICT_LATENCY.observe(elapsed)

        prob = float(preds[0])
        return max(0.0, min(1.0, prob))

    def _fallback_score(self, features: FeatureVector) -> float:
        """Use rule engine as fallback and return its fraud score.

        We construct a minimal Transaction for the rule engine since it
        requires one; the features carry the important signals.
        """
        from datetime import datetime
        from decimal import Decimal
        from uuid import uuid4

        from shared.schemas import Transaction, TransactionType

        dummy_txn = Transaction(
            transaction_id=features.transaction_id,
            user_id=features.user_id,
            amount=Decimal("0"),
            transaction_type=TransactionType.PURCHASE,
            timestamp=features.timestamp or datetime.utcnow(),
        )

        logger.warning(
            "xgboost_fallback_to_rules",
            version=self._model_version,
            load_error=str(self._load_error) if self._load_error else None,
        )

        result: ScoringResult = self._fallback_rule_engine.score(dummy_txn, features)
        return result.fraud_score
