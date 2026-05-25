"""Fraud scoring plugin system for the fraud detection platform.

This package provides a production-grade, extensible plugin framework that
allows custom fraud scoring models to be registered, managed, and invoked
alongside the platform's built-in ensemble scorer.

Architecture overview
---------------------

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                      Scoring Service                            │
    │                                                                 │
    │  Consumer / API  ──►  EnsembleScorer  ──►  PluginRegistry      │
    │                                              │                  │
    │                                              ▼                  │
    │                                    [enabled plugins]            │
    │                                    ┌──────────────────┐        │
    │                                    │  ScorerPlugin A  │        │
    │                                    │  ScorerPlugin B  │        │
    │                                    │  ScorerPlugin C  │        │
    │                                    └──────────────────┘        │
    │                                                                 │
    │  PluginLoader ──► load_from_module / directory / entrypoint     │
    │                   hot_reload                                    │
    └─────────────────────────────────────────────────────────────────┘

Core components
---------------

:class:`~scoring.plugins.base.ScorerPlugin`
    Abstract base class for all scoring plugins.  Subclass this and implement
    the ``name``, ``version``, ``description`` properties plus the ``score``,
    ``validate_config``, and ``health_check`` methods.

:class:`~scoring.plugins.base.PluginScore`
    Pydantic v2 model returned by every plugin's ``score()`` call.  Contains
    the fraud probability, optional confidence, explanatory metadata, and
    timing information.

:class:`~scoring.plugins.base.PluginInfo`
    Read-only snapshot of a registered plugin's metadata (name, version,
    enabled status, last health result, active config, load time).

:class:`~scoring.plugins.registry.PluginRegistry`
    Thread-safe store of :class:`~scoring.plugins.base.ScorerPlugin`
    instances.  Provides register/unregister, enable/disable, config update,
    individual scoring, batch scoring of all enabled plugins, and health checks.

:data:`~scoring.plugins.registry.registry`
    Module-level singleton :class:`~scoring.plugins.registry.PluginRegistry`
    shared across the scoring service.

:class:`~scoring.plugins.loader.PluginLoader`
    Discovers and dynamically loads plugins from a Python module path, a
    filesystem directory, or installed package entry-points.  Supports
    hot-reload without service restart.

Quick-start
-----------

**1. Implement a plugin**::

    from scoring.plugins import ScorerPlugin, PluginScore
    from shared.schemas import Transaction, FeatureVector

    class VelocityPlugin(ScorerPlugin):
        @property
        def name(self) -> str:
            return "velocity_scorer"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def description(self) -> str:
            return "Flags high-velocity users."

        async def score(
            self,
            transaction: Transaction,
            features: FeatureVector,
        ) -> PluginScore:
            risk = min(features.txn_count_1h / 20.0, 1.0)
            return PluginScore(
                score=risk,
                metadata={"txn_count_1h": features.txn_count_1h},
            )

        def validate_config(self, config: dict) -> bool:
            return True

        def health_check(self) -> bool:
            return True

**2. Register it**::

    from scoring.plugins import registry, PluginLoader

    loader = PluginLoader(registry=registry)
    loader.load_from_module("mypackage.scorers.VelocityPlugin")

**3. Score a transaction**::

    scores = await registry.score_all(transaction, features)

**Entry-point discovery** (pyproject.toml)::

    [project.entry-points."fraud_detection.scorers"]
    velocity = "mypackage.scorers:VelocityPlugin"

Then load with::

    loader.load_from_entrypoint("fraud_detection.scorers")
"""

from __future__ import annotations

from scoring.plugins.base import (
    PluginInfo,
    PluginLifecycleMixin,
    PluginScore,
    ScorerPlugin,
)
from scoring.plugins.loader import PluginLoader, PluginLoadError
from scoring.plugins.registry import PluginRegistry, registry

__all__ = [
    # Base abstractions
    "ScorerPlugin",
    "PluginLifecycleMixin",
    # Data models
    "PluginScore",
    "PluginInfo",
    # Registry
    "PluginRegistry",
    "registry",
    # Loader
    "PluginLoader",
    "PluginLoadError",
]
