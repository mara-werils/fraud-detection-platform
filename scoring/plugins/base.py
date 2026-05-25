"""Abstract base class and data models for fraud scoring plugins.

Every custom fraud scorer must subclass :class:`ScorerPlugin` and implement
all abstract methods.  The plugin contract is intentionally narrow so that
third-party authors only need to provide domain logic, while infrastructure
concerns (telemetry, lifecycle management, thread-safety) are handled by the
registry and loader layers.

Typical usage::

    class MyScorer(ScorerPlugin):
        @property
        def name(self) -> str:
            return "my_scorer"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def description(self) -> str:
            return "Example custom fraud scorer."

        async def score(
            self,
            transaction: Transaction,
            features: FeatureVector,
        ) -> PluginScore:
            raw = float(transaction.amount) / 10_000
            return PluginScore(score=min(raw, 1.0), metadata={"reason": "amount"})

        def validate_config(self, config: dict[str, Any]) -> bool:
            return True

        def health_check(self) -> bool:
            return True
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, field_validator

from shared.schemas import FeatureVector, Transaction

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class PluginScore(BaseModel):
    """Score produced by a :class:`ScorerPlugin`.

    Attributes:
        score: Fraud probability in ``[0.0, 1.0]``.  Higher values indicate a
            greater likelihood of fraud.
        confidence: Optional confidence level for the score in ``[0.0, 1.0]``.
            A plugin that cannot estimate confidence should leave this as
            ``None``.
        metadata: Arbitrary key-value pairs (e.g. feature contributions,
            decision path, rule IDs) that explain the score.  Must be JSON-
            serialisable.
        processing_time_ms: Wall-clock milliseconds spent inside the plugin's
            ``score`` method.  Populated automatically by the registry wrapper;
            plugins may leave this at ``0.0``.
        plugin_name: Name of the plugin that produced this score.  Populated
            automatically by the registry wrapper.
        plugin_version: Version of the plugin.  Populated automatically.
        scored_at: UTC timestamp when scoring completed.
        transaction_id: ID of the scored transaction for correlation.
    """

    model_config = {"frozen": True}

    score: float = Field(..., ge=0.0, le=1.0, description="Fraud probability in [0, 1]")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence of the score estimate",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Explanatory key-value pairs from the plugin",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock ms spent in score()",
    )
    plugin_name: str = Field(default="", description="Populated by registry")
    plugin_version: str = Field(default="", description="Populated by registry")
    scored_at: datetime = Field(default_factory=datetime.utcnow)
    transaction_id: UUID | None = Field(default=None, description="Correlated transaction")

    @field_validator("metadata", mode="before")
    @classmethod
    def _metadata_must_be_json_safe(cls, v: Any) -> dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("metadata must be a dict")
        return v

    def with_plugin_info(
        self,
        name: str,
        version: str,
        processing_time_ms: float,
        transaction_id: UUID | None = None,
    ) -> PluginScore:
        """Return a copy of this score with plugin attribution fields set."""
        return self.model_copy(
            update={
                "plugin_name": name,
                "plugin_version": version,
                "processing_time_ms": processing_time_ms,
                "transaction_id": transaction_id,
            }
        )


class PluginInfo(BaseModel):
    """Snapshot of a plugin's registration metadata."""

    model_config = {"frozen": True}

    name: str
    version: str
    description: str
    enabled: bool
    module_path: str = Field(default="", description="Python module where the class lives")
    loaded_at: datetime
    config: dict[str, Any] = Field(default_factory=dict, description="Active configuration")
    health: bool = Field(default=True, description="Last known health-check result")


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


class PluginLifecycleMixin:
    """Optional lifecycle hooks that plugins may override."""

    def on_load(self, config: dict[str, Any]) -> None:
        """Called once after the plugin is instantiated and registered."""

    def on_unload(self) -> None:
        """Called just before the plugin is removed from the registry."""

    def on_config_change(self, old_config: dict[str, Any], new_config: dict[str, Any]) -> None:
        """Called when the plugin's configuration is updated at runtime."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ScorerPlugin(PluginLifecycleMixin, ABC):
    """Abstract base class that all fraud scoring plugins must extend.

    A plugin encapsulates a single scoring strategy (e.g. a rule set, a
    machine-learning model, a third-party API call) and exposes a minimal,
    stable interface so the platform can orchestrate many plugins without
    tight coupling.

    Subclasses **must** implement:
        - :attr:`name`
        - :attr:`version`
        - :attr:`description`
        - :meth:`score`
        - :meth:`validate_config`
        - :meth:`health_check`

    Subclasses **may** override lifecycle hooks defined in
    :class:`PluginLifecycleMixin`.

    Thread-safety:
        :meth:`score` will be called concurrently from multiple asyncio tasks.
        Implementations must be re-entrant or protect shared mutable state
        with appropriate locks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique, human-readable identifier for this plugin."""

    @property
    @abstractmethod
    def version(self) -> str:
        """SemVer string for this plugin implementation (e.g. ``"1.3.0"``)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line human-readable description of what this plugin scores."""

    @abstractmethod
    async def score(
        self,
        transaction: Transaction,
        features: FeatureVector,
    ) -> PluginScore:
        """Compute a fraud score for the given transaction.

        Args:
            transaction: The raw incoming transaction.
            features: Pre-computed feature vector aligned to this transaction.

        Returns:
            A :class:`PluginScore` with ``score`` in ``[0.0, 1.0]``.
        """

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate a configuration dictionary for this plugin.

        Args:
            config: The candidate configuration mapping.

        Returns:
            ``True`` if the config is valid, ``False`` otherwise.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` if the plugin is operating normally.

        Returns:
            ``True`` if healthy, ``False`` if the plugin has detected an
            internal problem.
        """

    def _log(self) -> structlog.BoundLogger:
        """Return a logger bound with plugin identity fields."""
        return logger.bind(plugin_name=self.name, plugin_version=self.version)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} version={self.version!r}>"
