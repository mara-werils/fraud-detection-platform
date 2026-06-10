"""Dynamic plugin loading for the fraud scoring plugin system.

:class:`PluginLoader` discovers :class:`~scoring.plugins.base.ScorerPlugin`
subclasses from three sources:

1. **Module path** — a dotted Python import path.
2. **Directory scan** — walks a filesystem directory and imports every ``*.py``.
3. **Entry-points** — reads ``importlib.metadata`` entry-point groups.

Hot-reload is supported: :meth:`PluginLoader.hot_reload` forces a module
re-import, instantiates a fresh plugin, and replaces the existing registration.

Usage::

    from scoring.plugins.loader import PluginLoader
    from scoring.plugins.registry import registry

    loader = PluginLoader(registry=registry)
    loader.load_from_module("mypackage.velocity_scorer.VelocityPlugin")
    plugins = loader.load_from_directory("/opt/fraud/custom_scorers")
    plugins = loader.load_from_entrypoint("fraud_detection.scorers")
    loader.hot_reload("velocity_scorer")
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import sys
import threading
import types
from pathlib import Path
from typing import Any

import structlog

from scoring.plugins.base import ScorerPlugin
from scoring.plugins.registry import PluginRegistry

__all__ = [
    "PluginLoadError",
    "PluginLoader",
]

logger = structlog.get_logger(__name__)


class PluginLoadError(Exception):
    """Raised when a plugin cannot be loaded or instantiated."""


class PluginLoader:
    """Discover and dynamically load :class:`~scoring.plugins.base.ScorerPlugin` subclasses.

    Args:
        registry: The :class:`~scoring.plugins.registry.PluginRegistry` that
            loaded plugins will be registered into.
        default_config: A mapping of ``{plugin_name: config_dict}`` applied
            automatically when a plugin is registered through this loader.
    """

    def __init__(
        self,
        registry: PluginRegistry,
        default_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._registry = registry
        self._default_config: dict[str, dict[str, Any]] = default_config or {}
        # Maps plugin_name -> (module_dotted_path, class_name) for hot-reload.
        self._source_map: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()
        self._log = logger.bind(loader=repr(registry))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_module(
        self,
        module_path: str,
        *,
        class_name: str | None = None,
        config: dict[str, Any] | None = None,
        auto_register: bool = True,
    ) -> list[ScorerPlugin]:
        """Import a Python module and extract concrete :class:`ScorerPlugin` subclasses.

        Two calling conventions are supported:

        * ``module_path="pkg.module.ClassName"`` — loads exactly that class.
        * ``module_path="pkg.module"`` — inspects all module members and
          returns every concrete :class:`ScorerPlugin` subclass found.

        Args:
            module_path: Dotted Python path.
            class_name: Explicit class name within the module.
            config: Configuration passed to :meth:`PluginRegistry.register`.
            auto_register: When ``True`` (default), automatically registers each plugin.

        Returns:
            List of instantiated (and optionally registered) plugins.

        Raises:
            PluginLoadError: If the module cannot be imported or no concrete
                plugin classes are found.
        """
        resolved_module, resolved_class = _split_module_class(module_path, class_name)

        try:
            mod = importlib.import_module(resolved_module)
        except ModuleNotFoundError as exc:
            raise PluginLoadError(
                f"Cannot import module {resolved_module!r}: {exc}"
            ) from exc

        plugins = _collect_plugins_from_module(mod, class_name=resolved_class)

        if not plugins:
            raise PluginLoadError(
                f"No concrete ScorerPlugin subclasses found in {resolved_module!r}"
                + (f" (class {resolved_class!r})" if resolved_class else "")
            )

        if auto_register:
            for plugin in plugins:
                plugin_config = config or self._default_config.get(plugin.name, {})
                self._registry.register(
                    plugin,
                    config=plugin_config,
                    module_path=f"{resolved_module}.{type(plugin).__name__}",
                    overwrite=False,
                )
                with self._lock:
                    self._source_map[plugin.name] = (
                        resolved_module,
                        type(plugin).__name__,
                    )
                self._log.info(
                    "loader.module.loaded",
                    plugin_name=plugin.name,
                    module=resolved_module,
                )

        return plugins

    def load_from_directory(
        self,
        dir_path: str | Path,
        *,
        glob: str = "*.py",
        recursive: bool = False,
        config_map: dict[str, dict[str, Any]] | None = None,
        auto_register: bool = True,
    ) -> list[ScorerPlugin]:
        """Scan a directory for Python files containing plugin classes.

        Args:
            dir_path: Filesystem path to the directory to scan.
            glob: File name pattern. Default ``"*.py"``.
            recursive: When ``True``, scan subdirectories as well.
            config_map: Optional ``{plugin_name: config_dict}`` overrides.
            auto_register: When ``True`` (default), register each discovered plugin.

        Returns:
            List of all successfully instantiated plugins from the directory.

        Raises:
            PluginLoadError: If *dir_path* does not exist or is not a directory.
        """
        base = Path(dir_path).expanduser().resolve()
        if not base.exists():
            raise PluginLoadError(f"Plugin directory does not exist: {base}")
        if not base.is_dir():
            raise PluginLoadError(f"Plugin path is not a directory: {base}")

        pattern = f"**/{glob}" if recursive else glob
        py_files = sorted(base.glob(pattern))

        all_plugins: list[ScorerPlugin] = []
        config_map = config_map or {}

        for py_file in py_files:
            if py_file.name.startswith("_"):
                continue
            try:
                mod = _import_from_path(py_file)
                plugins = _collect_plugins_from_module(mod)
                for plugin in plugins:
                    plugin_config = (
                        config_map.get(plugin.name)
                        or self._default_config.get(plugin.name, {})
                    )
                    if auto_register:
                        module_dotted = _path_to_dotted(py_file, base)
                        self._registry.register(
                            plugin,
                            config=plugin_config,
                            module_path=f"{module_dotted}.{type(plugin).__name__}",
                            overwrite=False,
                        )
                        with self._lock:
                            self._source_map[plugin.name] = (
                                module_dotted,
                                type(plugin).__name__,
                            )
                    all_plugins.append(plugin)
                    self._log.info(
                        "loader.directory.loaded",
                        plugin_name=plugin.name,
                        file=str(py_file),
                    )
            except Exception:
                self._log.exception(
                    "loader.directory.file_error",
                    file=str(py_file),
                )

        self._log.info(
            "loader.directory.complete",
            directory=str(base),
            plugin_count=len(all_plugins),
        )
        return all_plugins

    def load_from_entrypoint(
        self,
        group: str,
        *,
        config_map: dict[str, dict[str, Any]] | None = None,
        auto_register: bool = True,
    ) -> list[ScorerPlugin]:
        """Load plugins advertised via ``importlib.metadata`` entry-points.

        Third-party packages expose plugins by adding entries like::

            [project.entry-points."fraud_detection.scorers"]
            my_scorer = "mypackage.scorer:MyScorer"

        to their ``pyproject.toml``.

        Args:
            group: The entry-point group name (e.g. ``"fraud_detection.scorers"``).
            config_map: Optional ``{plugin_name: config_dict}`` overrides.
            auto_register: When ``True`` (default), register each plugin.

        Returns:
            List of all successfully instantiated plugins from the group.
        """
        eps = importlib.metadata.entry_points(group=group)
        all_plugins: list[ScorerPlugin] = []
        config_map = config_map or {}

        for ep in eps:
            try:
                klass = ep.load()
                if not _is_concrete_plugin(klass):
                    self._log.warning(
                        "loader.entrypoint.not_a_plugin",
                        entry_point=ep.name,
                        klass=repr(klass),
                    )
                    continue

                plugin: ScorerPlugin = klass()
                plugin_config = (
                    config_map.get(plugin.name)
                    or self._default_config.get(plugin.name, {})
                )

                if auto_register:
                    module_dotted = klass.__module__
                    self._registry.register(
                        plugin,
                        config=plugin_config,
                        module_path=f"{module_dotted}.{klass.__name__}",
                        overwrite=False,
                    )
                    with self._lock:
                        self._source_map[plugin.name] = (
                            module_dotted,
                            klass.__name__,
                        )

                all_plugins.append(plugin)
                self._log.info(
                    "loader.entrypoint.loaded",
                    entry_point=ep.name,
                    plugin_name=plugin.name,
                    group=group,
                )
            except Exception:
                self._log.exception(
                    "loader.entrypoint.error",
                    entry_point=ep.name,
                    group=group,
                )

        self._log.info(
            "loader.entrypoint.complete",
            group=group,
            plugin_count=len(all_plugins),
        )
        return all_plugins

    def hot_reload(self, name: str, *, overwrite: bool = True) -> ScorerPlugin:
        """Force a re-import of the module that hosts *name* and re-register it.

        Hot-reload sequence:
        1. Look up the source ``(module, class_name)`` stored at original load time.
        2. Remove the module from ``sys.modules`` so the next import is fresh.
        3. Re-import the module and instantiate a new plugin instance.
        4. Call :meth:`~PluginRegistry.register` with ``overwrite=True``.

        Args:
            name: Plugin name.
            overwrite: Passed to :meth:`PluginRegistry.register`.

        Returns:
            The newly instantiated plugin.

        Raises:
            KeyError: If *name* was not loaded through this loader.
            PluginLoadError: If re-import or instantiation fails.
        """
        with self._lock:
            if name not in self._source_map:
                raise KeyError(
                    f"No source mapping for plugin {name!r}. "
                    "Only plugins loaded via this loader can be hot-reloaded."
                )
            module_dotted, class_name = self._source_map[name]

        self._log.info(
            "loader.hot_reload.start",
            plugin_name=name,
            module=module_dotted,
            class_name=class_name,
        )

        _evict_module(module_dotted)

        try:
            mod = importlib.import_module(module_dotted)
        except ModuleNotFoundError as exc:
            raise PluginLoadError(
                f"Hot-reload failed: cannot re-import {module_dotted!r}: {exc}"
            ) from exc

        klass = getattr(mod, class_name, None)
        if klass is None or not _is_concrete_plugin(klass):
            raise PluginLoadError(
                f"Hot-reload failed: class {class_name!r} not found or "
                f"is not a concrete ScorerPlugin in {module_dotted!r}"
            )

        plugin: ScorerPlugin = klass()
        existing_entry = self._registry._entries.get(name)
        plugin_config = (
            existing_entry.config if existing_entry else self._default_config.get(name, {})
        )

        self._registry.register(
            plugin,
            config=plugin_config,
            module_path=f"{module_dotted}.{class_name}",
            overwrite=overwrite,
        )

        with self._lock:
            self._source_map[plugin.name] = (module_dotted, class_name)

        self._log.info(
            "loader.hot_reload.complete",
            plugin_name=plugin.name,
            module=module_dotted,
        )
        return plugin

    def source_map(self) -> dict[str, tuple[str, str]]:
        """Return a copy of the ``{plugin_name: (module, class_name)}`` mapping."""
        with self._lock:
            return dict(self._source_map)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_module_class(
    path: str,
    explicit_class: str | None,
) -> tuple[str, str | None]:
    """Split a dotted path into ``(module_path, class_name_or_None)``."""
    if explicit_class:
        return path, explicit_class

    parts = path.rsplit(".", 1)
    if len(parts) == 2 and parts[1][0].isupper():
        return parts[0], parts[1]

    return path, None


def _collect_plugins_from_module(
    mod: types.ModuleType,
    *,
    class_name: str | None = None,
) -> list[ScorerPlugin]:
    """Return all concrete ScorerPlugin subclasses from *mod*."""
    plugins: list[ScorerPlugin] = []

    members = inspect.getmembers(mod, inspect.isclass)
    for attr_name, klass in members:
        if class_name and attr_name != class_name:
            continue
        if klass.__module__ != mod.__name__:
            continue
        if _is_concrete_plugin(klass):
            try:
                instance = klass()
                plugins.append(instance)
            except Exception:
                logger.exception(
                    "loader.instantiate_error",
                    module=mod.__name__,
                    class_name=attr_name,
                )

    return plugins


def _is_concrete_plugin(obj: Any) -> bool:
    """Return ``True`` if *obj* is a non-abstract ScorerPlugin subclass."""
    return (
        inspect.isclass(obj)
        and issubclass(obj, ScorerPlugin)
        and obj is not ScorerPlugin
        and not inspect.isabstract(obj)
    )


def _import_from_path(py_file: Path) -> types.ModuleType:
    """Import a Python file as a module using its filesystem path."""
    module_name = f"_fd_plugin_{py_file.stem}_{abs(hash(str(py_file)))}"
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"Cannot build import spec for {py_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _path_to_dotted(py_file: Path, base: Path) -> str:
    """Convert a filesystem path relative to *base* to a dotted module name."""
    relative = py_file.relative_to(base)
    parts = list(relative.parts)
    parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def _evict_module(module_dotted: str) -> None:
    """Remove *module_dotted* (and any sub-modules) from ``sys.modules``."""
    to_remove = [
        key for key in sys.modules
        if key == module_dotted or key.startswith(module_dotted + ".")
    ]
    for key in to_remove:
        del sys.modules[key]
