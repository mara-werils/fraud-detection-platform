"""High-value transaction fraud scoring plugin — reference implementation.

This module serves as both a production-ready plugin and a canonical example
that plugin authors should consult when building their own scorers.

Scoring strategy
----------------
The plugin combines three independent signals to produce a final fraud
probability:

1. **Absolute amount threshold** — transactions above a configurable ceiling
   (default $10,000) receive an elevated base score.

2. **Amount-to-average ratio** — a transaction that is many multiples of a
   user's historical average is more suspicious than a large transaction from
   a high-spending user.

3. **Velocity penalty** — a burst of transactions in the same hour, each of
   high value, is a classic fraud pattern.

The three signals are blended with configurable weights and clamped to
``[0.0, 1.0]``.  The plugin also emits rich metadata so downstream
explainability components can surface human-readable reasons.

Configuration (``validate_config``)
------------------------------------
The plugin accepts an optional configuration dict with the following keys:

``amount_threshold``
    Float.  Transactions at or above this USD amount trigger the amount
    signal.  Default: ``10_000.0``.

``amount_weight``
    Float in ``[0, 1]``.  Weight of the amount signal in the final blend.
    Default: ``0.5``.

``ratio_weight``
    Float in ``[0, 1]``.  Weight of the amount-to-average ratio signal.
    Default: ``0.3``.

``velocity_weight``
    Float in ``[0, 1]``.  Weight of the velocity signal.  Default: ``0.2``.

``velocity_threshold``
    Integer.  Number of transactions in the last hour that triggers the
    maximum velocity penalty.  Default: ``5``.

Weights need not sum to 1.0; the final score is normalised.

Lifecycle
---------
* ``on_load`` — stores the resolved config and emits a startup log line.
* ``on_unload`` — emits a shutdown log line.
* ``on_config_change`` — hot-swaps internal thresholds without restart.
* ``health_check`` — always returns ``True`` (pure-Python, no external deps).
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any

import structlog

from scoring.plugins.base import PluginLifecycleMixin, PluginScore, ScorerPlugin
from shared.schemas import FeatureVector, Transaction

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_AMOUNT_THRESHOLD: float = 10_000.0
_DEFAULT_AMOUNT_WEIGHT: float = 0.5
_DEFAULT_RATIO_WEIGHT: float = 0.3
_DEFAULT_VELOCITY_WEIGHT: float = 0.2
_DEFAULT_VELOCITY_THRESHOLD: int = 5

PLUGIN_NAME = "high_value_scorer"
PLUGIN_VERSION = "1.2.0"
PLUGIN_DESCRIPTION = (
    "Scores transactions based on absolute amount, amount-to-average ratio, "
    "and intra-hour velocity to detect high-value fraud patterns."
)


# ---------------------------------------------------------------------------
# Internal config container
# ---------------------------------------------------------------------------


class _Config:
    """Validated configuration snapshot for :class:`HighValueScorer`.

    All fields are set once from a raw dict and then accessed concurrently
    without locking (reads of Python floats/ints are atomic at the CPython
    level; for strict safety we copy-on-write via :meth:`from_dict`).
    """

    __slots__ = (
        "amount_threshold",
        "amount_weight",
        "ratio_weight",
        "velocity_weight",
        "velocity_threshold",
    )

    def __init__(
        self,
        amount_threshold: float,
        amount_weight: float,
        ratio_weight: float,
        velocity_weight: float,
        velocity_threshold: int,
    ) -> None:
        self.amount_threshold = amount_threshold
        self.amount_weight = amount_weight
        self.ratio_weight = ratio_weight
        self.velocity_weight = velocity_weight
        self.velocity_threshold = velocity_threshold

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> _Config:
        return cls(
            amount_threshold=float(
                config.get("amount_threshold", _DEFAULT_AMOUNT_THRESHOLD)
            ),
            amount_weight=float(config.get("amount_weight", _DEFAULT_AMOUNT_WEIGHT)),
            ratio_weight=float(config.get("ratio_weight", _DEFAULT_RATIO_WEIGHT)),
            velocity_weight=float(
                config.get("velocity_weight", _DEFAULT_VELOCITY_WEIGHT)
            ),
            velocity_threshold=int(
                config.get("velocity_threshold", _DEFAULT_VELOCITY_THRESHOLD)
            ),
        )

    @classmethod
    def default(cls) -> _Config:
        return cls.from_dict({})

    def total_weight(self) -> float:
        return self.amount_weight + self.ratio_weight + self.velocity_weight


# ---------------------------------------------------------------------------
# Plugin implementation
# ---------------------------------------------------------------------------


class HighValueScorer(ScorerPlugin):
    """Production-quality example plugin that flags high-value transactions.

    This plugin is designed to be read alongside the framework's abstract base
    class.  Every method is documented in detail to help plugin authors
    understand the expected behaviour and best practices.

    Thread-safety:
        :attr:`_config` is replaced atomically via a :class:`threading.Lock`
        in :meth:`on_config_change`.  The ``score`` method reads it once per
        call without holding the lock, which is safe because Python reference
        assignment is atomic.
    """

    def __init__(self) -> None:
        # _config is replaced atomically; no lock needed during reads.
        self._config: _Config = _Config.default()
        self._config_lock = threading.Lock()
        self._log = logger.bind(plugin_name=PLUGIN_NAME, plugin_version=PLUGIN_VERSION)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return PLUGIN_NAME

    @property
    def version(self) -> str:
        return PLUGIN_VERSION

    @property
    def description(self) -> str:
        return PLUGIN_DESCRIPTION

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_load(self, config: dict[str, Any]) -> None:
        """Apply initial configuration and announce readiness.

        Args:
            config: Validated configuration dict from the registry.
        """
        with self._config_lock:
            self._config = _Config.from_dict(config)
        self._log.info(
            "high_value_scorer.loaded",
            amount_threshold=self._config.amount_threshold,
            velocity_threshold=self._config.velocity_threshold,
        )

    def on_unload(self) -> None:
        """Log shutdown.  No resources to release in this plugin."""
        self._log.info("high_value_scorer.unloaded")

    def on_config_change(
        self,
        old_config: dict[str, Any],
        new_config: dict[str, Any],
    ) -> None:
        """Hot-swap the internal config without restarting.

        Args:
            old_config: The previous configuration dict.
            new_config: The incoming (already validated) configuration dict.
        """
        new = _Config.from_dict(new_config)
        with self._config_lock:
            self._config = new
        self._log.info(
            "high_value_scorer.config_changed",
            old_threshold=old_config.get("amount_threshold"),
            new_threshold=new_config.get("amount_threshold"),
        )

    # ------------------------------------------------------------------
    # Core scoring logic
    # ------------------------------------------------------------------

    async def score(
        self,
        transaction: Transaction,
        features: FeatureVector,
    ) -> PluginScore:
        """Compute a fraud score for *transaction* using three blended signals.

        The method is ``async`` even though it performs no I/O, conforming to
        the plugin interface which allows implementations to call external
        services via ``await``.

        Args:
            transaction: The raw incoming transaction.
            features: Pre-computed feature vector aligned to this transaction.

        Returns:
            A :class:`~scoring.plugins.base.PluginScore` with:
            - ``score``: blended probability in ``[0.0, 1.0]``.
            - ``confidence``: ``None`` (this plugin does not estimate confidence).
            - ``metadata``: per-signal scores, weights, and decision factors.
        """
        # Snapshot config once for this request — safe if config is replaced
        # mid-flight because Python object references are atomic.
        cfg = self._config

        amount = float(transaction.amount)

        # ------------------------------------------------------------------
        # Signal 1: Absolute amount
        # ------------------------------------------------------------------
        amount_score = _compute_amount_score(amount, cfg.amount_threshold)

        # ------------------------------------------------------------------
        # Signal 2: Amount-to-average ratio
        # ------------------------------------------------------------------
        ratio_score = _compute_ratio_score(features.amount_to_avg_ratio)

        # ------------------------------------------------------------------
        # Signal 3: Intra-hour velocity
        # ------------------------------------------------------------------
        velocity_score = _compute_velocity_score(
            features.txn_count_1h, cfg.velocity_threshold
        )

        # ------------------------------------------------------------------
        # Weighted blend
        # ------------------------------------------------------------------
        total_weight = cfg.total_weight()
        if total_weight == 0.0:
            # Degenerate config — return neutral score rather than dividing by zero.
            blended = 0.0
        else:
            blended = (
                amount_score * cfg.amount_weight
                + ratio_score * cfg.ratio_weight
                + velocity_score * cfg.velocity_weight
            ) / total_weight

        final_score = max(0.0, min(blended, 1.0))

        # ------------------------------------------------------------------
        # Metadata for explainability
        # ------------------------------------------------------------------
        metadata: dict[str, Any] = {
            "signals": {
                "amount_score": round(amount_score, 4),
                "ratio_score": round(ratio_score, 4),
                "velocity_score": round(velocity_score, 4),
            },
            "weights": {
                "amount": cfg.amount_weight,
                "ratio": cfg.ratio_weight,
                "velocity": cfg.velocity_weight,
            },
            "inputs": {
                "transaction_amount_usd": amount,
                "amount_threshold_usd": cfg.amount_threshold,
                "amount_to_avg_ratio": float(features.amount_to_avg_ratio),
                "txn_count_1h": features.txn_count_1h,
                "velocity_threshold": cfg.velocity_threshold,
            },
            "decision_factors": _build_decision_factors(
                amount=amount,
                amount_score=amount_score,
                ratio_score=ratio_score,
                velocity_score=velocity_score,
                cfg=cfg,
                features=features,
            ),
        }

        self._log.debug(
            "high_value_scorer.scored",
            transaction_id=str(transaction.transaction_id),
            final_score=round(final_score, 4),
            amount_usd=amount,
        )

        return PluginScore(
            score=final_score,
            confidence=None,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate the configuration dict for this plugin.

        Checks:
        - All numeric values are non-negative.
        - Weights are in ``[0.0, 1.0]``.
        - ``velocity_threshold`` is a positive integer.
        - At least one weight is non-zero (otherwise scoring is pointless).

        Args:
            config: The candidate configuration.

        Returns:
            ``True`` if the config is acceptable, ``False`` otherwise.
        """
        try:
            amount_threshold = float(
                config.get("amount_threshold", _DEFAULT_AMOUNT_THRESHOLD)
            )
            amount_weight = float(config.get("amount_weight", _DEFAULT_AMOUNT_WEIGHT))
            ratio_weight = float(config.get("ratio_weight", _DEFAULT_RATIO_WEIGHT))
            velocity_weight = float(
                config.get("velocity_weight", _DEFAULT_VELOCITY_WEIGHT)
            )
            velocity_threshold = int(
                config.get("velocity_threshold", _DEFAULT_VELOCITY_THRESHOLD)
            )
        except (TypeError, ValueError):
            return False

        if amount_threshold < 0:
            return False
        if not (0.0 <= amount_weight <= 1.0):
            return False
        if not (0.0 <= ratio_weight <= 1.0):
            return False
        if not (0.0 <= velocity_weight <= 1.0):
            return False
        if velocity_threshold < 1:
            return False
        if amount_weight + ratio_weight + velocity_weight == 0.0:
            return False

        return True

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return ``True`` — this plugin has no external dependencies.

        A plugin that calls an external model server or database should probe
        that dependency here and return ``False`` if it is unavailable.

        Returns:
            Always ``True`` for this pure-Python implementation.
        """
        return True


# ---------------------------------------------------------------------------
# Pure scoring functions (testable in isolation)
# ---------------------------------------------------------------------------


def _compute_amount_score(amount: float, threshold: float) -> float:
    """Map an absolute amount to a fraud signal in ``[0.0, 1.0]``.

    Below *threshold* the score is 0.  Above threshold the score grows
    logarithmically toward 1.0, reaching ~0.9 at 10x the threshold.

    Args:
        amount: Transaction amount in USD.
        threshold: Amount at which the signal activates.

    Returns:
        Signal score in ``[0.0, 1.0]``.
    """
    if threshold <= 0 or amount < threshold:
        return 0.0

    import math

    # Sigmoid-like growth using log — score = log(1 + ratio) / log(10)
    # where ratio = amount / threshold.  At ratio=1 → score≈0.30;
    # at ratio=10 → score≈1.0.
    ratio = amount / threshold
    raw = math.log10(1.0 + ratio)
    return min(raw, 1.0)


def _compute_ratio_score(amount_to_avg_ratio: float) -> float:
    """Map an amount-to-average ratio to a fraud signal in ``[0.0, 1.0]``.

    A ratio <= 1.0 → 0.  A ratio of 10x → 0.9.

    Args:
        amount_to_avg_ratio: ``transaction.amount / user_avg_amount_24h``.

    Returns:
        Signal score in ``[0.0, 1.0]``.
    """
    if amount_to_avg_ratio <= 1.0:
        return 0.0

    import math

    # Same log-scale as amount, anchored at ratio=1.
    raw = math.log10(amount_to_avg_ratio)
    return min(raw, 1.0)


def _compute_velocity_score(txn_count_1h: int, velocity_threshold: int) -> float:
    """Map a 1-hour transaction count to a fraud signal in ``[0.0, 1.0]``.

    Linear ramp from 0 (at count=0) to 1.0 (at count >= *velocity_threshold*).

    Args:
        txn_count_1h: Number of transactions by this user in the last hour.
        velocity_threshold: Count at which the score reaches 1.0.

    Returns:
        Signal score in ``[0.0, 1.0]``.
    """
    if velocity_threshold <= 0:
        return 0.0
    return min(txn_count_1h / velocity_threshold, 1.0)


def _build_decision_factors(
    *,
    amount: float,
    amount_score: float,
    ratio_score: float,
    velocity_score: float,
    cfg: _Config,
    features: FeatureVector,
) -> list[str]:
    """Build a human-readable list of the factors that drove the score.

    Args:
        amount: Transaction amount.
        amount_score: Computed amount signal.
        ratio_score: Computed ratio signal.
        velocity_score: Computed velocity signal.
        cfg: Active plugin configuration.
        features: Feature vector for this transaction.

    Returns:
        List of plain-English reason strings (may be empty for low-risk txns).
    """
    factors: list[str] = []

    if amount_score > 0.0:
        factors.append(
            f"Transaction amount ${amount:,.2f} exceeds threshold "
            f"${cfg.amount_threshold:,.2f} (signal={amount_score:.2f})"
        )

    if ratio_score > 0.0:
        factors.append(
            f"Amount is {features.amount_to_avg_ratio:.1f}x the user's 24h average "
            f"(signal={ratio_score:.2f})"
        )

    if velocity_score >= 0.5:
        factors.append(
            f"{features.txn_count_1h} transactions in the last hour "
            f"(threshold {cfg.velocity_threshold}, signal={velocity_score:.2f})"
        )

    return factors
