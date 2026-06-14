"""Configuration hot-reload with validation, rollback, and change notifications.

Watches YAML and JSON configuration files for changes and atomically swaps
the active configuration after validating the new values. Supports
environment-specific overrides, config versioning, diff logging, and
thread-safe access.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

ChangeCallback = Callable[["ConfigSnapshot", "ConfigSnapshot"], None]
"""Signature: callback(old_snapshot, new_snapshot) -> None"""


class ConfigValidator(Protocol):
    """Protocol for config validators.

    A validator receives the merged configuration dict and must raise
    ``ValueError`` (or any ``Exception`` subclass) if the config is invalid.
    """

    def __call__(self, config: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# ConfigSnapshot – immutable, versioned view of a configuration
# ---------------------------------------------------------------------------


class ConfigSnapshot:
    """Immutable, versioned snapshot of a configuration dictionary."""

    __slots__ = ("_data", "_version", "_timestamp")

    def __init__(self, data: dict[str, Any], version: int, timestamp: float | None = None):
        self._data = copy.deepcopy(data)
        self._version = version
        self._timestamp = timestamp or time.time()

    # -- read helpers -------------------------------------------------------

    @property
    def data(self) -> dict[str, Any]:
        """Return a deep copy so callers cannot mutate internal state."""
        return copy.deepcopy(self._data)

    @property
    def version(self) -> int:
        return self._version

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def get(self, key: str, default: Any = None) -> Any:
        """Lookup a top-level key."""
        return copy.deepcopy(self._data.get(key, default))

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"ConfigSnapshot(version={self._version}, keys={list(self._data.keys())})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_file(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON file and return a dict.

    Raises ``ValueError`` for unsupported extensions and parse errors.
    """
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        parsed = yaml.safe_load(text)
    elif suffix == ".json":
        parsed = json.loads(text)
    else:
        raise ValueError(f"Unsupported config file extension: {suffix}")

    if not isinstance(parsed, dict):
        raise ValueError(f"Config file must be a mapping, got {type(parsed).__name__}")
    return parsed


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def compute_diff(
    old: dict[str, Any],
    new: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    """Return a list of human-readable change descriptions between two dicts."""
    changes: list[str] = []
    all_keys = set(old.keys()) | set(new.keys())
    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in old:
            changes.append(f"added {full_key}={new[key]!r}")
        elif key not in new:
            changes.append(f"removed {full_key} (was {old[key]!r})")
        elif isinstance(old[key], dict) and isinstance(new[key], dict):
            changes.extend(compute_diff(old[key], new[key], prefix=full_key))
        elif old[key] != new[key]:
            changes.append(f"changed {full_key}: {old[key]!r} -> {new[key]!r}")
    return changes


# ---------------------------------------------------------------------------
# HotReloader
# ---------------------------------------------------------------------------


class HotReloader:
    """Watches config files and atomically hot-reloads on change.

    Parameters
    ----------
    config_path:
        Primary configuration file (YAML or JSON).
    env:
        Optional environment name (e.g. ``"production"``).  When set, the
        reloader also loads ``<stem>.<env>.<ext>`` (if it exists) and deep-
        merges it on top of the base config.
    validators:
        Sequence of callables.  Each receives the merged config dict and
        should raise on invalid data.
    callbacks:
        Sequence of ``(old_snapshot, new_snapshot)`` callables invoked after
        a successful reload.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        env: str | None = None,
        validators: list[ConfigValidator] | None = None,
        callbacks: list[ChangeCallback] | None = None,
    ) -> None:
        self._config_path = Path(config_path).resolve()
        self._env = env
        self._validators: list[ConfigValidator] = list(validators or [])
        self._callbacks: list[ChangeCallback] = list(callbacks or [])

        self._lock = threading.RLock()
        self._version = 0
        self._snapshot: ConfigSnapshot | None = None
        self._stop_event = threading.Event()
        self._watcher_thread: threading.Thread | None = None

        # Perform the initial load (raises on failure – intentional).
        self._reload()

    # -- public API ---------------------------------------------------------

    @property
    def snapshot(self) -> ConfigSnapshot:
        """Return the current config snapshot (thread-safe)."""
        with self._lock:
            assert self._snapshot is not None
            return self._snapshot

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def get(self, key: str, default: Any = None) -> Any:
        """Shortcut to ``snapshot.get(key, default)``."""
        return self.snapshot.get(key, default)

    def add_validator(self, validator: ConfigValidator) -> None:
        with self._lock:
            self._validators.append(validator)

    def add_callback(self, callback: ChangeCallback) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def reload(self) -> ConfigSnapshot:
        """Manually trigger a reload.  Returns the new snapshot.

        Raises ``ConfigReloadError`` if validation fails (config is rolled
        back to the previous version).
        """
        return self._reload()

    # -- watcher control ----------------------------------------------------

    def start_watching(self, poll_interval: int = 1000) -> None:
        """Start a background thread that watches config files for changes.

        Parameters
        ----------
        poll_interval:
            Polling interval in **milliseconds** passed to *watchfiles*.
        """
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            return  # already watching

        self._stop_event.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            args=(poll_interval,),
            daemon=True,
            name="config-hot-reloader",
        )
        self._watcher_thread.start()

    def stop_watching(self) -> None:
        """Signal the watcher thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=5)
            self._watcher_thread = None

    @property
    def watching(self) -> bool:
        return self._watcher_thread is not None and self._watcher_thread.is_alive()

    # -- internals ----------------------------------------------------------

    def _env_override_path(self) -> Path | None:
        if not self._env:
            return None
        stem = self._config_path.stem
        suffix = self._config_path.suffix
        parent = self._config_path.parent
        override = parent / f"{stem}.{self._env}{suffix}"
        return override if override.is_file() else None

    def _load_merged(self) -> dict[str, Any]:
        """Load base config, optionally merge env override, return dict."""
        base = _load_file(self._config_path)
        override_path = self._env_override_path()
        if override_path:
            override = _load_file(override_path)
            base = _deep_merge(base, override)
            logger.info("Merged environment override from %s", override_path)
        return base

    def _validate(self, config: dict[str, Any]) -> None:
        for validator in self._validators:
            validator(config)

    def _reload(self) -> ConfigSnapshot:
        with self._lock:
            old_snapshot = self._snapshot
            try:
                merged = self._load_merged()
                self._validate(merged)
            except Exception as exc:
                logger.error(
                    "Config reload failed (keeping v%s): %s",
                    self._version,
                    exc,
                )
                raise ConfigReloadError(str(exc)) from exc

            self._version += 1
            new_snapshot = ConfigSnapshot(merged, self._version)
            # Atomic swap
            self._snapshot = new_snapshot

            # Diff logging
            if old_snapshot is not None:
                diffs = compute_diff(old_snapshot.data, new_snapshot.data)
                if diffs:
                    logger.info(
                        "Config updated to v%s – changes: %s",
                        self._version,
                        "; ".join(diffs),
                    )
                else:
                    logger.debug("Config reloaded to v%s with no effective changes", self._version)

            # Notify callbacks
            callbacks = list(self._callbacks)

        # Run callbacks outside the lock to avoid deadlocks.
        if old_snapshot is not None:
            for cb in callbacks:
                try:
                    cb(old_snapshot, new_snapshot)
                except Exception:
                    logger.exception("Error in config change callback %s", cb)

        return new_snapshot

    def _watch_loop(self, poll_interval: int) -> None:
        """Background loop driven by *watchfiles*."""
        from watchfiles import watch

        paths_to_watch: list[Path] = [self._config_path]
        env_path = self._env_override_path()
        if env_path:
            paths_to_watch.append(env_path)

        # watchfiles.watch() blocks until changes are detected or the
        # stop_event is set.
        for _changes in watch(
            *paths_to_watch,
            stop_event=self._stop_event,
            step=poll_interval,
        ):
            if self._stop_event.is_set():
                break
            try:
                self._reload()
            except ConfigReloadError:
                pass  # already logged inside _reload


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ConfigReloadError(Exception):
    """Raised when a configuration reload fails validation."""


__all__ = [
    "ChangeCallback",
    "ConfigReloadError",
    "ConfigSnapshot",
    "ConfigValidator",
    "HotReloader",
    "compute_diff",
]
