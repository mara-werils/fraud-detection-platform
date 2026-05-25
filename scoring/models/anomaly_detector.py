"""Isolation Forest anomaly detection service for fraud detection.

Implements Isolation Forest from scratch (no sklearn dependency for the core
algorithm) with:
  - Tree-based isolation scoring
  - Online / incremental learning via partial_fit
  - Multi-variate anomaly scoring
  - Per-feature contribution attribution
  - Adaptive thresholds calibrated from historical data
  - Ensemble of multiple Isolation Forests for robustness
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field

from shared.schemas import FeatureVector

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

ANOMALY_SCORE_HISTOGRAM = Histogram(
    "anomaly_detection_score",
    "Distribution of anomaly scores produced by the Isolation Forest ensemble",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

ANOMALY_DETECT_LATENCY = Histogram(
    "anomaly_detection_latency_seconds",
    "End-to-end latency of AnomalyDetectionService.detect()",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25),
)

ANOMALY_DETECTED_TOTAL = Counter(
    "anomaly_detected_total",
    "Number of samples classified as anomalies",
)

ANOMALY_THRESHOLD_GAUGE = Gauge(
    "anomaly_detection_threshold",
    "Current adaptive anomaly score threshold",
)

MODEL_UPDATE_TOTAL = Counter(
    "anomaly_model_update_total",
    "Number of partial-fit / model-update calls",
)

# ---------------------------------------------------------------------------
# Feature names (must match FeatureVector field ordering used for scoring)
# ---------------------------------------------------------------------------

FEATURE_NAMES: tuple[str, ...] = (
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

N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Pydantic result model
# ---------------------------------------------------------------------------


class AnomalyResult(BaseModel):
    """Result produced by the Isolation Forest anomaly detector."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Anomaly score in [0, 1]; higher means more anomalous.",
    )
    is_anomaly: bool = Field(
        ...,
        description="True when score exceeds the current adaptive threshold.",
    )
    feature_contributions: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-feature contribution to the anomaly score, "
            "values in [0, 1] and sum to approximately 1."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the anomaly decision, derived from ensemble "
            "agreement (low variance → high confidence)."
        ),
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Internal isolation tree node
# ---------------------------------------------------------------------------


@dataclass
class _IsolationNode:
    """A single node in an isolation tree.

    Leaf nodes have ``left`` and ``right`` set to None and store
    ``size`` (number of training samples that fell in this region).
    Internal nodes store the split dimension and threshold.
    """

    left: _IsolationNode | None = field(default=None, repr=False)
    right: _IsolationNode | None = field(default=None, repr=False)
    split_feature: int = -1
    split_threshold: float = 0.0
    size: int = 0  # meaningful only at leaf
    depth: int = 0


# ---------------------------------------------------------------------------
# c(n) — average path length of unsuccessful search in a BST
# ---------------------------------------------------------------------------


def _c(n: int) -> float:
    """Expected path length for an isolation tree trained on *n* samples.

    This is the normalisation constant from the original IF paper
    (Liu et al. 2008):  c(n) = 2·H(n-1) - 2(n-1)/n
    where H is the harmonic number, approximated by ln + Euler–Mascheroni.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


# ---------------------------------------------------------------------------
# Single Isolation Tree
# ---------------------------------------------------------------------------


class _IsolationTree:
    """A single randomised isolation tree.

    Args:
        max_depth: Maximum tree depth (limits memory and enforces sub-sampling).
        rng: NumPy random Generator used for reproducible splits.
    """

    def __init__(self, max_depth: int, rng: np.random.Generator) -> None:
        self._max_depth = max_depth
        self._rng = rng
        self._root: _IsolationNode | None = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> None:
        """Build the isolation tree from training data *X* (shape [n, d])."""
        self._root = self._build(X, depth=0)

    def _build(self, X: np.ndarray, depth: int) -> _IsolationNode:
        n, d = X.shape
        node = _IsolationNode(size=n, depth=depth)

        # Leaf conditions: single sample, or max depth reached
        if n <= 1 or depth >= self._max_depth:
            return node

        # Pick a random feature that actually has variance
        attempts = 0
        while attempts < d:
            feat = int(self._rng.integers(0, d))
            col = X[:, feat]
            col_min, col_max = float(col.min()), float(col.max())
            if col_min < col_max:
                break
            attempts += 1
        else:
            # All features are constant — treat as leaf
            return node

        # Random split threshold uniform in [min, max)
        threshold = float(self._rng.uniform(col_min, col_max))

        node.split_feature = feat
        node.split_threshold = threshold

        mask = X[:, feat] < threshold
        node.left = self._build(X[mask], depth + 1)
        node.right = self._build(X[~mask], depth + 1)
        return node

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def path_length(self, x: np.ndarray) -> float:
        """Return the path length for sample *x* (1-D array of length d).

        Adds the expected additional path length when a leaf is reached
        before the point is isolated (i.e., leaf size > 1).
        """
        return self._path_length_node(x, self._root, current_depth=0)

    def _path_length_node(
        self, x: np.ndarray, node: _IsolationNode | None, current_depth: int
    ) -> float:
        if node is None:
            return current_depth
        # Leaf node
        if node.left is None and node.right is None:
            return current_depth + _c(node.size)
        # Internal node
        if x[node.split_feature] < node.split_threshold:
            return self._path_length_node(x, node.left, current_depth + 1)
        return self._path_length_node(x, node.right, current_depth + 1)

    def path_length_with_contributions(
        self, x: np.ndarray
    ) -> tuple[float, dict[int, int]]:
        """Return path length and per-feature split counts along the path."""
        contributions: dict[int, int] = {}
        length = self._traverse_with_contributions(
            x, self._root, current_depth=0, contributions=contributions
        )
        return length, contributions

    def _traverse_with_contributions(
        self,
        x: np.ndarray,
        node: _IsolationNode | None,
        current_depth: int,
        contributions: dict[int, int],
    ) -> float:
        if node is None:
            return current_depth
        if node.left is None and node.right is None:
            return current_depth + _c(node.size)
        feat = node.split_feature
        contributions[feat] = contributions.get(feat, 0) + 1
        if x[feat] < node.split_threshold:
            return self._traverse_with_contributions(
                x, node.left, current_depth + 1, contributions
            )
        return self._traverse_with_contributions(
            x, node.right, current_depth + 1, contributions
        )


# ---------------------------------------------------------------------------
# Isolation Forest (ensemble of trees)
# ---------------------------------------------------------------------------


class IsolationForest:
    """Isolation Forest anomaly detector built from scratch.

    Args:
        n_estimators: Number of isolation trees in the ensemble.
        max_samples: Number of samples drawn (without replacement) to train
            each tree.  ``None`` uses the full training set.
        max_depth: Maximum depth per tree.  Defaults to ``ceil(log2(max_samples))``.
        contamination: Expected fraction of anomalies; used only to derive
            a default threshold via ``fit``.
        random_state: Integer seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | None = 256,
        max_depth: int | None = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_samples = max_samples
        self._contamination = contamination
        self._random_state = random_state

        self._trees: list[_IsolationTree] = []
        self._n_train: int = 0  # samples used for normalisation c(n)
        self._fitted = False

        # Derived max depth (set during fit)
        self._max_depth = max_depth

        # Master RNG; child RNGs are derived per tree for reproducibility
        self._master_rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Fit / partial_fit
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray) -> None:
        """Train all isolation trees on *data* (shape [n_samples, n_features]).

        Replaces any previously fitted trees.

        Args:
            data: 2-D float array of training samples.
        """
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError("data must be a 2-D array with at least 2 rows.")

        n, _ = data.shape
        effective_samples = min(n, self._max_samples) if self._max_samples else n
        max_depth = self._max_depth or math.ceil(math.log2(max(effective_samples, 2)))

        self._n_train = effective_samples
        self._trees = []

        for _ in range(self._n_estimators):
            tree_rng = np.random.default_rng(self._master_rng.integers(0, 2**31))
            idx = tree_rng.choice(n, size=effective_samples, replace=False)
            subset = data[idx].astype(np.float64)
            tree = _IsolationTree(max_depth=max_depth, rng=tree_rng)
            tree.fit(subset)
            self._trees.append(tree)

        self._fitted = True
        logger.info(
            "isolation_forest_fitted",
            n_estimators=self._n_estimators,
            n_train=n,
            effective_samples=effective_samples,
            max_depth=max_depth,
        )

    def partial_fit(self, new_data: np.ndarray) -> None:
        """Incrementally update the forest with *new_data*.

        Replaces the oldest ``len(new_data)`` trees with trees trained on
        *new_data*, keeping the rest of the ensemble intact.  This is a
        sliding-window online update strategy.

        Args:
            new_data: 2-D float array of new samples.
        """
        if not self._fitted:
            self.fit(new_data)
            return

        if new_data.ndim != 2 or new_data.shape[0] < 2:
            logger.warning("partial_fit_skipped", reason="insufficient_new_data")
            return

        n, _ = new_data.shape
        effective_samples = min(n, self._max_samples) if self._max_samples else n
        max_depth = self._max_depth or math.ceil(math.log2(max(effective_samples, 2)))

        # Replace the first k trees (round-robin oldest-first)
        n_replace = min(max(1, self._n_estimators // 4), self._n_estimators)
        for i in range(n_replace):
            tree_rng = np.random.default_rng(self._master_rng.integers(0, 2**31))
            idx = tree_rng.choice(n, size=min(effective_samples, n), replace=False)
            subset = new_data[idx].astype(np.float64)
            tree = _IsolationTree(max_depth=max_depth, rng=tree_rng)
            tree.fit(subset)
            self._trees[i] = tree

        MODEL_UPDATE_TOTAL.inc()
        logger.info(
            "isolation_forest_partial_fit",
            n_replaced=n_replace,
            n_new_samples=n,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, sample: np.ndarray) -> float:
        """Return anomaly score in [0, 1] for a single sample.

        Score ≈ 1.0  →  highly anomalous (isolated quickly).
        Score ≈ 0.5  →  normal (deep paths, hard to isolate).

        Args:
            sample: 1-D float array of length n_features.

        Returns:
            Anomaly score in [0, 1].
        """
        if not self._fitted or not self._trees:
            raise RuntimeError("IsolationForest must be fitted before scoring.")
        x = np.asarray(sample, dtype=np.float64).ravel()
        avg_path = np.mean([t.path_length(x) for t in self._trees])
        c_n = _c(self._n_train)
        if c_n <= 0:
            return 0.5
        anomaly_score = 2.0 ** (-avg_path / c_n)
        return float(np.clip(anomaly_score, 0.0, 1.0))

    def predict(self, sample: np.ndarray, threshold: float = 0.6) -> AnomalyResult:
        """Score *sample* and return a full :class:`AnomalyResult`.

        Args:
            sample: 1-D float array of length n_features.
            threshold: Anomaly score above which the sample is flagged.

        Returns:
            AnomalyResult with score, label, contributions, and confidence.
        """
        x = np.asarray(sample, dtype=np.float64).ravel()
        tree_scores: list[float] = []
        agg_contributions: dict[int, int] = {}

        c_n = _c(self._n_train)
        for tree in self._trees:
            path, contribs = tree.path_length_with_contributions(x)
            tree_score = float(2.0 ** (-path / c_n)) if c_n > 0 else 0.5
            tree_scores.append(tree_score)
            for feat_idx, cnt in contribs.items():
                agg_contributions[feat_idx] = agg_contributions.get(feat_idx, 0) + cnt

        score = float(np.clip(np.mean(tree_scores), 0.0, 1.0))
        confidence = self._ensemble_confidence(tree_scores)
        feature_contributions = self._normalise_contributions(agg_contributions)

        return AnomalyResult(
            score=score,
            is_anomaly=score >= threshold,
            feature_contributions=feature_contributions,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, sample: np.ndarray) -> dict[str, float]:
        """Return per-feature anomaly contribution for *sample*.

        Each value is the fraction of all tree splits (along the isolation
        path) that used that feature; higher means that feature drove isolation.

        Args:
            sample: 1-D float array of length n_features.

        Returns:
            Dict mapping feature name → contribution in [0, 1].
        """
        x = np.asarray(sample, dtype=np.float64).ravel()
        agg: dict[int, int] = {}
        for tree in self._trees:
            _, contribs = tree.path_length_with_contributions(x)
            for feat_idx, cnt in contribs.items():
                agg[feat_idx] = agg.get(feat_idx, 0) + cnt
        return self._normalise_contributions(agg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensemble_confidence(scores: list[float]) -> float:
        """Confidence derived from inter-tree score variance.

        Low variance → high confidence.  Max variance for [0,1] is 0.25.
        """
        if len(scores) <= 1:
            return 1.0
        variance = float(np.var(scores))
        return float(np.clip(1.0 - variance / 0.25, 0.0, 1.0))

    @staticmethod
    def _normalise_contributions(raw: dict[int, int]) -> dict[str, float]:
        """Convert raw per-feature split counts to named, normalised fractions."""
        total = sum(raw.values()) or 1
        result: dict[str, float] = {}
        for feat_idx, cnt in raw.items():
            if feat_idx < len(FEATURE_NAMES):
                result[FEATURE_NAMES[feat_idx]] = round(cnt / total, 6)
        # Features not traversed get 0.0
        for name in FEATURE_NAMES:
            result.setdefault(name, 0.0)
        return result


# ---------------------------------------------------------------------------
# Adaptive threshold manager
# ---------------------------------------------------------------------------


class _AdaptiveThreshold:
    """Maintains a rolling percentile-based anomaly threshold.

    Args:
        initial_threshold: Starting threshold value.
        percentile: The score percentile used to set the threshold
            when calibrated from historical scores.
        window_size: Maximum number of recent scores retained.
    """

    def __init__(
        self,
        initial_threshold: float = 0.6,
        percentile: float = 95.0,
        window_size: int = 10_000,
    ) -> None:
        self._threshold = initial_threshold
        self._percentile = percentile
        self._window_size = window_size
        self._scores: list[float] = []
        self._lock = threading.Lock()

    @property
    def value(self) -> float:
        return self._threshold

    def set(self, threshold: float) -> None:
        with self._lock:
            self._threshold = float(np.clip(threshold, 0.01, 0.99))
        ANOMALY_THRESHOLD_GAUGE.set(self._threshold)

    def observe(self, score: float) -> None:
        """Record a new score for future threshold calibration."""
        with self._lock:
            self._scores.append(score)
            if len(self._scores) > self._window_size:
                self._scores = self._scores[-self._window_size :]

    def calibrate(self) -> float:
        """Recalculate threshold from the observed score distribution.

        Returns:
            Newly calibrated threshold.
        """
        with self._lock:
            if len(self._scores) < 100:
                return self._threshold
            new_threshold = float(np.percentile(self._scores, self._percentile))
            new_threshold = float(np.clip(new_threshold, 0.3, 0.95))
            self._threshold = new_threshold

        ANOMALY_THRESHOLD_GAUGE.set(self._threshold)
        logger.info("anomaly_threshold_calibrated", threshold=round(new_threshold, 4))
        return new_threshold


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------


def _feature_vector_to_array(features: FeatureVector) -> np.ndarray:
    """Convert a FeatureVector to a float64 numpy array in FEATURE_NAMES order."""
    return np.array(
        [float(getattr(features, name)) for name in FEATURE_NAMES],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# AnomalyDetectionService
# ---------------------------------------------------------------------------


class AnomalyDetectionService:
    """Production wrapper around an ensemble of Isolation Forests.

    Supports:
      - Single detect() call for live transaction scoring
      - Batch partial_fit via update_model() for online learning
      - Adaptive threshold management
      - Full Prometheus observability

    Args:
        n_forests: Number of independent Isolation Forests in the meta-ensemble.
        n_estimators: Number of trees per forest.
        max_samples: Sub-sample size per tree.
        contamination: Expected anomaly fraction (used for initial threshold).
        random_state: Master seed for reproducibility.
        initial_threshold: Starting anomaly score threshold.
    """

    def __init__(
        self,
        n_forests: int = 3,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.05,
        random_state: int = 42,
        initial_threshold: float = 0.6,
    ) -> None:
        self._n_forests = n_forests
        self._lock = threading.RLock()

        # Build independent forests with distinct seeds
        self._forests: list[IsolationForest] = [
            IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=contamination,
                random_state=random_state + i * 1000,
            )
            for i in range(n_forests)
        ]

        self._threshold = _AdaptiveThreshold(initial_threshold=initial_threshold)
        ANOMALY_THRESHOLD_GAUGE.set(initial_threshold)

        self._fitted = False

        logger.info(
            "anomaly_detection_service_init",
            n_forests=n_forests,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            initial_threshold=initial_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, features: FeatureVector) -> AnomalyResult:
        """Score a transaction feature vector for anomalies.

        If the forests have not been fitted yet, returns a neutral result
        (score=0.5, is_anomaly=False) to avoid blocking the scoring path.

        Args:
            features: Pre-computed feature vector for the transaction.

        Returns:
            AnomalyResult with score, label, contributions, and confidence.
        """
        start = time.perf_counter()

        if not self._fitted:
            logger.warning("anomaly_detector_not_fitted", action="returning_neutral")
            return AnomalyResult(
                score=0.5,
                is_anomaly=False,
                feature_contributions={name: 0.0 for name in FEATURE_NAMES},
                confidence=0.0,
            )

        x = _feature_vector_to_array(features)
        threshold = self._threshold.value

        with self._lock:
            forests = list(self._forests)

        # Collect per-forest results
        forest_scores: list[float] = []
        agg_contributions: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
        forest_confidences: list[float] = []

        for forest in forests:
            try:
                result = forest.predict(x, threshold=threshold)
                forest_scores.append(result.score)
                forest_confidences.append(result.confidence)
                for fname, val in result.feature_contributions.items():
                    agg_contributions[fname] = agg_contributions.get(fname, 0.0) + val
            except Exception as exc:
                logger.error("anomaly_forest_predict_error", error=str(exc))

        if not forest_scores:
            return AnomalyResult(
                score=0.5,
                is_anomaly=False,
                feature_contributions={name: 0.0 for name in FEATURE_NAMES},
                confidence=0.0,
            )

        # Ensemble aggregation
        ensemble_score = float(np.mean(forest_scores))
        ensemble_confidence = float(np.mean(forest_confidences))

        # Normalise aggregated contributions
        total_contrib = sum(agg_contributions.values()) or 1.0
        normalised: dict[str, float] = {
            k: round(v / total_contrib, 6) for k, v in agg_contributions.items()
        }

        is_anomaly = ensemble_score >= threshold

        result = AnomalyResult(
            score=round(ensemble_score, 6),
            is_anomaly=is_anomaly,
            feature_contributions=normalised,
            confidence=round(ensemble_confidence, 6),
        )

        # Record observation for adaptive threshold
        self._threshold.observe(ensemble_score)

        # Prometheus
        ANOMALY_SCORE_HISTOGRAM.observe(ensemble_score)
        ANOMALY_DETECT_LATENCY.observe(time.perf_counter() - start)
        if is_anomaly:
            ANOMALY_DETECTED_TOTAL.inc()

        logger.debug(
            "anomaly_detection_result",
            transaction_id=str(features.transaction_id),
            score=round(ensemble_score, 4),
            is_anomaly=is_anomaly,
            confidence=round(ensemble_confidence, 4),
            threshold=threshold,
        )

        return result

    def update_model(self, recent_transactions: list[Any]) -> None:
        """Incrementally update all forests with recently seen transaction data.

        Each item in *recent_transactions* must either be a :class:`FeatureVector`
        or a plain dict / list convertible to the canonical feature array.

        Args:
            recent_transactions: Sequence of FeatureVector objects or
                raw numpy-compatible feature rows.
        """
        if not recent_transactions:
            logger.warning("update_model_called_with_empty_list")
            return

        rows: list[np.ndarray] = []
        for item in recent_transactions:
            # Accept FeatureVector instances or any duck-typed object that
            # exposes all canonical feature attributes.
            if isinstance(item, FeatureVector) or all(
                hasattr(item, name) for name in FEATURE_NAMES
            ):
                rows.append(_feature_vector_to_array(item))
            else:
                try:
                    arr = np.asarray(item, dtype=np.float64).ravel()
                except (TypeError, ValueError):
                    logger.warning("update_model_skipping_unconvertible_item")
                    continue
                if arr.shape[0] == N_FEATURES:
                    rows.append(arr)
                else:
                    logger.warning(
                        "update_model_skipping_bad_row",
                        expected_features=N_FEATURES,
                        got=arr.shape[0],
                    )

        if len(rows) < 2:
            logger.warning("update_model_insufficient_rows", n_rows=len(rows))
            return

        data = np.stack(rows)

        with self._lock:
            for forest in self._forests:
                forest.partial_fit(data)

            if not self._fitted:
                self._fitted = True

        # Recalibrate adaptive threshold after update
        self._threshold.calibrate()

        logger.info(
            "anomaly_model_updated",
            n_samples=len(rows),
            threshold=round(self._threshold.value, 4),
        )

    def fit(self, data: np.ndarray) -> None:
        """Bulk-train all forests from scratch on *data* (shape [n, n_features]).

        Args:
            data: 2-D float array of training samples.
        """
        with self._lock:
            for forest in self._forests:
                forest.fit(data)
            self._fitted = True

        self._threshold.calibrate()
        logger.info(
            "anomaly_detection_service_fitted",
            n_samples=data.shape[0],
            threshold=round(self._threshold.value, 4),
        )

    def get_threshold(self) -> float:
        """Return the current adaptive anomaly score threshold.

        Returns:
            Float in (0, 1).
        """
        return self._threshold.value

    def set_threshold(self, threshold: float) -> None:
        """Manually override the anomaly score threshold.

        Args:
            threshold: New threshold value in (0, 1).

        Raises:
            ValueError: If threshold is outside (0, 1).
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold!r}")
        self._threshold.set(threshold)
        logger.info("anomaly_threshold_manually_set", threshold=threshold)

    def feature_importance(self, features: FeatureVector) -> dict[str, float]:
        """Return per-feature importance for a single FeatureVector.

        Aggregates contributions across all forests.

        Args:
            features: Pre-computed feature vector.

        Returns:
            Dict mapping feature name → contribution in [0, 1].
        """
        if not self._fitted:
            return {name: 0.0 for name in FEATURE_NAMES}

        x = _feature_vector_to_array(features)
        agg: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}

        with self._lock:
            forests = list(self._forests)

        for forest in forests:
            try:
                contribs = forest.feature_importance(x)
                for fname, val in contribs.items():
                    agg[fname] = agg.get(fname, 0.0) + val
            except Exception as exc:
                logger.error("feature_importance_error", error=str(exc))

        total = sum(agg.values()) or 1.0
        return {k: round(v / total, 6) for k, v in agg.items()}

    @property
    def is_fitted(self) -> bool:
        """True once the service has been trained via fit() or update_model()."""
        return self._fitted
