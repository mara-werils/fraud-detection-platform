"""Graph Neural Network scorer using pre-computed embeddings from Redis."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import structlog
from prometheus_client import Histogram
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)

GNN_PREDICT_LATENCY = Histogram(
    "gnn_predict_latency_seconds",
    "Latency of GNN predict calls",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

# Default embedding dimension; must match the trained GNN output.
DEFAULT_EMBEDDING_DIM = 64


class GNNScorer:
    """GNN fraud scorer that combines pre-computed graph embeddings with
    transaction features to produce a fraud probability.

    Embeddings are read from Redis (key ``gnn:embedding:{user_id}``).  If an
    embedding is not cached the scorer returns 0.5 (neutral) and logs a warning.

    The underlying PyTorch Geometric model is loaded lazily from a state dict
    on first use.

    Args:
        model_path: Path to the saved PyTorch state dict (``.pt`` file).
        embedding_dim: Dimensionality of the GNN node embeddings.
        model_version: Human-readable version for logging/metrics.
        redis: Async Redis client for embedding lookups.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        model_version: str = "gnn-v1",
        redis: Redis | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._embedding_dim = embedding_dim
        self._model_version = model_version
        self._redis = redis

        self._model: Any | None = None  # torch.nn.Module
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
        """Load the GNN model state dict from disk.

        Raises:
            RuntimeError: If loading fails.
        """
        with self._load_lock:
            if self._loaded:
                return

            if self._model_path is None:
                self._load_error = ValueError("No model_path configured for GNN scorer")
                logger.error("gnn_model_load_failed", error="no model_path configured")
                raise RuntimeError("No model_path configured for GNN scorer")

            try:
                import torch

                state_dict = torch.load(
                    str(self._model_path), map_location="cpu", weights_only=True
                )
                # We store the state dict directly; downstream code can build
                # the Module architecture and call load_state_dict.  For inference
                # we use a simple linear classifier on top of the embedding.
                self._model = state_dict
                self._loaded = True
                self._load_error = None
                logger.info(
                    "gnn_model_loaded",
                    path=str(self._model_path),
                    version=self._model_version,
                )
            except Exception as exc:
                self._load_error = exc
                logger.error(
                    "gnn_model_load_failed",
                    error=str(exc),
                    version=self._model_version,
                )
                raise RuntimeError(f"Failed to load GNN model: {exc}") from exc

    def _ensure_model(self) -> bool:
        """Ensure model is loaded; return True on success."""
        if self._loaded:
            return True
        try:
            self._load_model()
            return True
        except RuntimeError:
            return False

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    async def _get_embedding(self, user_id: UUID) -> np.ndarray | None:
        """Fetch a pre-computed GNN embedding from Redis.

        Args:
            user_id: The user whose embedding to retrieve.

        Returns:
            Numpy array of shape ``(embedding_dim,)`` or ``None``.
        """
        if self._redis is None:
            logger.warning("gnn_redis_not_configured", user_id=str(user_id))
            return None

        key = f"gnn:embedding:{user_id}"
        try:
            raw: bytes | None = await self._redis.get(key)
            if raw is None:
                return None
            return np.frombuffer(raw, dtype=np.float32).copy()
        except Exception as exc:
            logger.error(
                "gnn_embedding_fetch_error",
                user_id=str(user_id),
                error=str(exc),
            )
            return None

    async def refresh_embeddings(
        self,
        user_embeddings: dict[UUID, np.ndarray],
        ttl_seconds: int = 3600,
    ) -> int:
        """Batch-write embeddings to Redis.

        Args:
            user_embeddings: Mapping of user_id to embedding array.
            ttl_seconds: Time-to-live for each cached embedding.

        Returns:
            Number of embeddings successfully written.
        """
        if self._redis is None:
            logger.error("gnn_redis_not_configured_for_refresh")
            return 0

        written = 0
        for user_id, embedding in user_embeddings.items():
            key = f"gnn:embedding:{user_id}"
            try:
                await self._redis.set(
                    key,
                    embedding.astype(np.float32).tobytes(),
                    ex=ttl_seconds,
                )
                written += 1
            except Exception as exc:
                logger.error(
                    "gnn_embedding_write_error",
                    user_id=str(user_id),
                    error=str(exc),
                )

        logger.info(
            "gnn_embeddings_refreshed",
            total=len(user_embeddings),
            written=written,
        )
        return written

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def score(
        self,
        user_id: UUID,
        features: Any,
    ) -> float:
        """Score a user/transaction using GNN embeddings.

        If the embedding is not cached, returns 0.5 (neutral).
        If the model is not loaded, returns 0.5 (neutral).

        Args:
            user_id: The user to score.
            features: FeatureVector (used for supplementary signals).

        Returns:
            Fraud probability in [0, 1].
        """
        embedding = await self._get_embedding(user_id)

        if embedding is None:
            logger.warning(
                "gnn_embedding_not_cached",
                user_id=str(user_id),
                returning="neutral_0.5",
            )
            return 0.5

        if not self._ensure_model():
            logger.warning(
                "gnn_model_not_loaded",
                user_id=str(user_id),
                returning="neutral_0.5",
            )
            return 0.5

        try:
            return self._predict(embedding)
        except Exception as exc:
            logger.error(
                "gnn_predict_error",
                user_id=str(user_id),
                error=str(exc),
                version=self._model_version,
            )
            return 0.5

    def _predict(self, embedding: np.ndarray) -> float:
        """Run the GNN classification head on an embedding.

        The model state dict is expected to contain ``classifier.weight``
        and ``classifier.bias`` tensors for a single-layer linear classifier.

        Args:
            embedding: Pre-computed node embedding array.

        Returns:
            Fraud probability in [0, 1].
        """
        import torch

        start = time.perf_counter()

        with torch.no_grad():
            emb_tensor = torch.from_numpy(embedding).float().unsqueeze(0)

            weight = self._model.get("classifier.weight")  # type: ignore[union-attr]
            bias = self._model.get("classifier.bias")  # type: ignore[union-attr]

            if weight is None or bias is None:
                logger.error("gnn_model_missing_classifier_params")
                return 0.5

            logits = torch.nn.functional.linear(emb_tensor, weight, bias)
            prob = torch.sigmoid(logits).item()

        elapsed = time.perf_counter() - start
        GNN_PREDICT_LATENCY.observe(elapsed)

        return max(0.0, min(1.0, float(prob)))
