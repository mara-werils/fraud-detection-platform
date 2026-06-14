"""Risk score normalizer for cross-model score calibration.

Different fraud detection models produce scores on varying scales and
distributions.  This module provides three normalisation strategies to map
raw model outputs to a common 0-100 scale so they can be combined in an
ensemble or compared on a single dashboard.

Strategies:
  - MinMax: linear rescaling based on observed min/max boundaries.
  - ZScore: standard-score normalisation clipped to [0, 100].
  - Percentile: empirical CDF mapping using a reference distribution.
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Sequence

import structlog

logger = structlog.get_logger(__name__)


class NormMethod(StrEnum):
    """Supported normalisation methods."""

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    PERCENTILE = "percentile"


@dataclass(frozen=True)
class NormConfig:
    """Configuration for a single normalisation profile."""

    method: NormMethod
    model_name: str
    # MinMax parameters
    raw_min: float = 0.0
    raw_max: float = 1.0
    # ZScore parameters
    mean: float = 0.0
    std: float = 1.0
    # Percentile reference distribution (sorted ascending)
    reference_scores: tuple[float, ...] = ()


@dataclass
class NormResult:
    """Outcome of a normalisation call."""

    model_name: str
    raw_score: float
    normalised_score: float
    method: NormMethod


class RiskScoreNormalizer:
    """Normalizes risk scores from heterogeneous models to a 0-100 scale."""

    def __init__(self) -> None:
        self._configs: dict[str, NormConfig] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, config: NormConfig) -> None:
        """Register a normalisation profile for a model."""
        if config.method == NormMethod.PERCENTILE and not config.reference_scores:
            raise ValueError(
                f"Percentile normalisation for '{config.model_name}' requires "
                "a non-empty reference_scores distribution."
            )
        self._configs[config.model_name] = config
        logger.info(
            "normalizer_registered",
            model=config.model_name,
            method=config.method.value,
        )

    def unregister(self, model_name: str) -> None:
        """Remove a model's normalisation profile."""
        self._configs.pop(model_name, None)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize(self, model_name: str, raw_score: float) -> NormResult:
        """Normalize *raw_score* using the registered profile for *model_name*."""
        config = self._configs.get(model_name)
        if config is None:
            raise KeyError(
                f"No normalisation config registered for model '{model_name}'"
            )

        if config.method == NormMethod.MIN_MAX:
            score = self._min_max(raw_score, config)
        elif config.method == NormMethod.Z_SCORE:
            score = self._z_score(raw_score, config)
        else:
            score = self._percentile(raw_score, config)

        return NormResult(
            model_name=model_name,
            raw_score=raw_score,
            normalised_score=score,
            method=config.method,
        )

    def normalize_batch(
        self, model_name: str, raw_scores: Sequence[float]
    ) -> list[NormResult]:
        """Normalize a batch of scores for a single model."""
        return [self.normalize(model_name, s) for s in raw_scores]

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _min_max(value: float, cfg: NormConfig) -> float:
        span = cfg.raw_max - cfg.raw_min
        if span == 0:
            return 50.0
        normalised = (value - cfg.raw_min) / span * 100.0
        return max(0.0, min(100.0, normalised))

    @staticmethod
    def _z_score(value: float, cfg: NormConfig) -> float:
        if cfg.std <= 0:
            return 50.0
        z = (value - cfg.mean) / cfg.std
        # Map z-score to 0-100 via sigmoid-like clamped linear transform.
        # z in [-3, 3] -> [0, 100]
        normalised = (z + 3.0) / 6.0 * 100.0
        return max(0.0, min(100.0, normalised))

    @staticmethod
    def _percentile(value: float, cfg: NormConfig) -> float:
        ref = cfg.reference_scores
        idx = bisect.bisect_left(ref, value)
        percentile = idx / len(ref) * 100.0
        return max(0.0, min(100.0, percentile))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def fit_from_sample(
        model_name: str, samples: Sequence[float], method: NormMethod
    ) -> NormConfig:
        """Build a NormConfig by fitting to a sample of historical scores."""
        if not samples:
            raise ValueError("Cannot fit normalisation from an empty sample.")

        sorted_samples = tuple(sorted(samples))
        return NormConfig(
            method=method,
            model_name=model_name,
            raw_min=sorted_samples[0],
            raw_max=sorted_samples[-1],
            mean=statistics.mean(samples),
            std=statistics.stdev(samples) if len(samples) > 1 else 1.0,
            reference_scores=sorted_samples,
        )
