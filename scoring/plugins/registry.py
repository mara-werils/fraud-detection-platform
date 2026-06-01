"""Plugin registry for managing the lifecycle of fraud scoring plugins.

The :class:`PluginRegistry` is a thread-safe, singleton-style store that maps
plugin names to :class:`~scoring.plugins.base.ScorerPlugin` instances.

Design principles
-----------------
- **Thread-safety**: all mutations are protected by a :class:`threading.RLock`.
- **Observability**: every register, unregister, enable, disable, and score
  event is logged via structlog.  Prometheus counters and histograms are
  emitted where appropriate.
- **Fail-fast on duplicate**: registering two plugins with the same name is
  rejected unless ``overwrite=True`` is supplied.
- **Lifecycle hooks**: the registry calls :meth:`ScorerPlugin.on_load` after
  registration and :meth:`ScorerPlugin.on_unload` before removal.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram

from scoring.plugins.base import PluginInfo, PluginScore, ScorerPlugin
from shared.schemas import FeatureVector, Transaction

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_PLUGIN_SCORE_LATENCY = Histogram(
    "plugin_score_latency_seconds",
    "Wall-clock time for a single plugin's score() call",
    ["plugin_name", "plugin_version"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

_PLUGIN_SCORE_ERRORS = Counter(
    "plugin_score_errors_total",
    "Number of exceptions raised inside plugin score() calls",
    ["plugin_name", "plugin_version"],
)

_PLUGIN_REGISTERED = Counter(
    "plugin_registered_total",
    "Cumulative number of plugin registrations",
    ["plugin_name"],
)

_PLUGIN_UNREGISTERED = Counter(
    "plugin_unregistered_total",
    "Cumulative number of plugin unregistrations",
    ["plugin_name"],
)

_PLUGINS_ACTIVE = Gauge(
    "plugins_active",
    "Number of currently enabled plugins",
)

_PLUGIN_HEALTH = Gauge(
    "plugin_health",
    "1 if the plugin is healthy, 0 otherwise",
    ["plugin_name"],
)


# ---------------------------------------------------------------------------
# Internal state container
# ---------------------------------------------------------------------------


class _PluginEntry:
    """Internal bookkeeping for a registered plugin."""

    __slots__ = (
        "plugin",
        "enabled",
        "config",
        "loaded_at",
        "module_path",
        "last_health",
    )

    def __init__(
        self,
        plugin: ScorerPlugin,
        config: dict[str, Any],
        module_path: str,
    ) -> None:
        self.plugin = plugin
        self.enabled: bool = True
        self.config: dict[str, Any] = dict(config)
        self.loaded_at: datetime = datetime.now(UTC)
        self.module_path: str = module_path
        self.last_health: bool = True

    def to_info(self) -> PluginInfo:
        return PluginInfo(
            name=self.plugin.name,
            version=self.plugin.version,
            description=self.plugin.description,
            enabled=self.enabled,
            module_path=self.module_path,
            loaded_at=self.loaded_at,
            config=self.config,
            health=self.last_health,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Thread-safe registry for :class:`~scoring.plugins.base.ScorerPlugin` instances.

    A single shared instance is exposed as the module-level :data:`registry`
    singleton.  Platform code should import and use that object rather than
    constructing new registries.

    Example::

        from scoring.plugins.registry import registry

        registry.register(MyPlugin(), config={"threshold": 0.7})
        score = await registry.score_with_plugin("my_plugin", txn, features)

    Args:
        name: Optional human-readable label for this registry (useful in
            tests where multiple registries co-exist).
    """

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._lock = threading.RLock()
        self._entries: dict[str, _PluginEntry] = {}
        self._log = logger.bind(registry=name)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        plugin: ScorerPlugin,
        *,
        config: dict[str, Any] | None = None,
        module_path: str = "",
        overwrite: bool = False,
    ) -> None:
        """Register a plugin and call its ``on_load`` hook.

        Args:
            plugin: The plugin instance to register.
            config: Optional configuration dict.  Will be validated via
                :meth:`~ScorerPlugin.validate_config` before registration.
            module_path: The dotted Python module path where the plugin class
                lives (used for hot-reload and introspection).
            overwrite: If ``True``, silently replace an existing plugin with
                the same name.

        Raises:
            ValueError: If ``plugin.name`` is already registered and
                ``overwrite`` is ``False``, or if ``config`` fails validation.
            TypeError: If *plugin* is not a :class:`ScorerPlugin` subclass.
        """
        if not isinstance(plugin, ScorerPlugin):
            raise TypeError(
                f"Expected ScorerPlugin subclass, got {type(plugin).__name__!r}"
            )

        resolved_config: dict[str, Any] = config or {}

        if not plugin.validate_config(resolved_config):
            raise ValueError(
                f"Plugin {plugin.name!r} rejected its configuration: {resolved_config}"
            )

        with self._lock:
            if plugin.name in self._entries and not overwrite:
                raise ValueError(
                    f"A plugin named {plugin.name!r} is already registered. "
                    "Pass overwrite=True to replace it."
                )

            if plugin.name in self._entries:
                self._unload_entry(self._entries[plugin.name])

            entry = _PluginEntry(
                plugin=plugin,
                config=resolved_config,
                module_path=module_path,
            )
            self._entries[plugin.name] = entry

        try:
            plugin.on_load(resolved_config)
        except Exception:
            self._log.exception(
                "plugin.on_load raised an exception",
                plugin_name=plugin.name,
            )

        _PLUGIN_REGISTERED.labels(plugin_name=plugin.name).inc()
        _PLUGINS_ACTIVE.inc()
        self._log.info(
            "plugin.registered",
            plugin_name=plugin.name,
            plugin_version=plugin.version,
            module_path=module_path,
        )

    def unregister(self, name: str) -> None:
        """Remove a plugin and call its ``on_unload`` hook.

        Args:
            name: The name of the plugin to remove.

        Raises:
            KeyError: If no plugin with *name* is registered.
        """
        with self._lock:
            if name not in self._entries:
                raise KeyError(f"No plugin named {name!r} is registered.")
            entry = self._entries.pop(name)

        self._unload_entry(entry)
        _PLUGIN_UNREGISTERED.labels(plugin_name=name).inc()
        _PLUGINS_ACTIVE.dec()
        self._log.info("plugin.unregistered", plugin_name=name)

    def _unload_entry(self, entry: _PluginEntry) -> None:
        """Call ``on_unload`` on an entry's plugin, swallowing exceptions."""
        try:
            entry.plugin.on_unload()
        except Exception:
            self._log.exception(
                "plugin.on_unload raised an exception",
                plugin_name=entry.plugin.name,
            )

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(self, name: str) -> None:
        """Enable a previously disabled plugin.

        Args:
            name: Plugin name.

        Raises:
            KeyError: If *name* is not registered.
        """
        with self._lock:
            entry = self._get_entry(name)
            if not entry.enabled:
                entry.enabled = True
                _PLUGINS_ACTIVE.inc()
                self._log.info("plugin.enabled", plugin_name=name)

    def disable(self, name: str) -> None:
        """Disable a plugin without removing it from the registry.

        Args:
            name: Plugin name.

        Raises:
            KeyError: If *name* is not registered.
        """
        with self._lock:
            entry = self._get_entry(name)
            if entry.enabled:
                entry.enabled = False
                _PLUGINS_ACTIVE.dec()
                self._log.info("plugin.disabled", plugin_name=name)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def update_config(self, name: str, new_config: dict[str, Any]) -> None:
        """Update a plugin's runtime configuration.

        Args:
            name: Plugin name.
            new_config: The replacement configuration.

        Raises:
            KeyError: If *name* is not registered.
            ValueError: If *new_config* fails validation.
        """
        with self._lock:
            entry = self._get_entry(name)
            plugin = entry.plugin

        if not plugin.validate_config(new_config):
            raise ValueError(
                f"Plugin {name!r} rejected the new configuration: {new_config}"
            )

        with self._lock:
            old_config = dict(entry.config)
            entry.config = dict(new_config)

        try:
            plugin.on_config_change(old_config, new_config)
        except Exception:
            self._log.exception(
                "plugin.on_config_change raised an exception",
                plugin_name=name,
            )

        self._log.info("plugin.config_updated", plugin_name=name, new_config=new_config)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ScorerPlugin:
        """Retrieve a registered plugin by name.

        Args:
            name: Plugin name.

        Returns:
            The :class:`~ScorerPlugin` instance.

        Raises:
            KeyError: If *name* is not registered.
        """
        return self._get_entry(name).plugin

    def _get_entry(self, name: str) -> _PluginEntry:
        with self._lock:
            try:
                return self._entries[name]
            except KeyError as exc:
                raise KeyError(f"No plugin named {name!r} is registered.") from exc

    def list_plugins(self, *, include_disabled: bool = True) -> list[PluginInfo]:
        """Snapshot of all registered plugins.

        Args:
            include_disabled: When ``False``, omit disabled plugins.

        Returns:
            A list of :class:`~PluginInfo` instances ordered alphabetically.
        """
        with self._lock:
            entries = list(self._entries.values())

        infos = [e.to_info() for e in entries if include_disabled or e.enabled]
        return sorted(infos, key=lambda i: i.name)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def score_with_plugin(
        self,
        name: str,
        transaction: Transaction,
        features: FeatureVector,
    ) -> PluginScore:
        """Invoke a specific plugin's ``score()`` and return an annotated score.

        Args:
            name: Plugin name.
            transaction: Transaction to score.
            features: Pre-computed feature vector.

        Returns:
            :class:`~PluginScore` with ``plugin_name``, ``plugin_version``,
            ``processing_time_ms``, and ``transaction_id`` populated.

        Raises:
            KeyError: If *name* is not registered.
        """
        entry = self._get_entry(name)
        plugin = entry.plugin

        t0 = time.perf_counter()
        try:
            raw_score = await plugin.score(transaction, features)
        except Exception:
            _PLUGIN_SCORE_ERRORS.labels(
                plugin_name=plugin.name,
                plugin_version=plugin.version,
            ).inc()
            self._log.exception(
                "plugin.score.error",
                plugin_name=plugin.name,
                transaction_id=str(transaction.transaction_id),
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            _PLUGIN_SCORE_LATENCY.labels(
                plugin_name=plugin.name,
                plugin_version=plugin.version,
            ).observe(elapsed_ms / 1_000)

        return raw_score.with_plugin_info(
            name=plugin.name,
            version=plugin.version,
            processing_time_ms=elapsed_ms,
            transaction_id=transaction.transaction_id,
        )

    async def score_all(
        self,
        transaction: Transaction,
        features: FeatureVector,
        *,
        concurrency: int = 8,
        raise_on_error: bool = False,
    ) -> list[PluginScore]:
        """Score a transaction with all *enabled* plugins concurrently.

        Args:
            transaction: Transaction to score.
            features: Pre-computed feature vector.
            concurrency: Maximum number of plugin ``score()`` calls in flight
                simultaneously.  Default 8.
            raise_on_error: If ``True``, the first plugin exception is
                re-raised.  If ``False`` (default), errors are suppressed.

        Returns:
            List of :class:`~PluginScore` objects ordered alphabetically.
        """
        with self._lock:
            enabled = [e for e in self._entries.values() if e.enabled]

        if not enabled:
            self._log.warning("score_all.no_enabled_plugins")
            return []

        sem = asyncio.Semaphore(concurrency)
        errors: list[Exception] = []

        async def _score_one(entry: _PluginEntry) -> PluginScore | None:
            async with sem:
                try:
                    return await self.score_with_plugin(
                        entry.plugin.name, transaction, features
                    )
                except Exception as exc:
                    errors.append(exc)
                    return None

        results = await asyncio.gather(*[_score_one(e) for e in enabled])
        scores = [r for r in results if r is not None]

        if errors and raise_on_error:
            raise errors[0]

        return sorted(scores, key=lambda s: s.plugin_name)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def run_health_checks(self) -> dict[str, bool]:
        """Call ``health_check()`` on every registered plugin.

        Updates the ``plugin_health`` Prometheus gauge.

        Returns:
            Mapping of ``{plugin_name: is_healthy}``.
        """
        results: dict[str, bool] = {}
        with self._lock:
            entries = list(self._entries.values())

        for entry in entries:
            try:
                healthy = entry.plugin.health_check()
            except Exception:
                self._log.exception(
                    "plugin.health_check.error",
                    plugin_name=entry.plugin.name,
                )
                healthy = False

            with self._lock:
                entry.last_health = healthy

            _PLUGIN_HEALTH.labels(plugin_name=entry.plugin.name).set(int(healthy))
            results[entry.plugin.name] = healthy

        return results

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._entries)
        return f"<PluginRegistry name={self._name!r} plugins={count}>"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Shared registry used by the scoring service.
registry: PluginRegistry = PluginRegistry(name="default")
