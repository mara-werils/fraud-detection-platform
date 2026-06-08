"""Tests for configuration hot-reload."""

from __future__ import annotations

import json
import textwrap
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from shared.hot_reload import (
    ConfigReloadError,
    ConfigSnapshot,
    HotReloader,
    _deep_merge,
    _load_file,
    compute_diff,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def yaml_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"database": {"host": "localhost", "port": 5432}, "debug": False}))
    return cfg


@pytest.fixture()
def json_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"database": {"host": "localhost", "port": 5432}, "debug": False}))
    return cfg


# ---------------------------------------------------------------------------
# ConfigSnapshot
# ---------------------------------------------------------------------------


class TestConfigSnapshot:
    def test_data_returns_deep_copy(self):
        snap = ConfigSnapshot({"a": [1, 2]}, version=1)
        d1 = snap.data
        d1["a"].append(3)
        assert snap.data == {"a": [1, 2]}, "Internal data must not be mutated"

    def test_version_and_timestamp(self):
        snap = ConfigSnapshot({}, version=42, timestamp=100.0)
        assert snap.version == 42
        assert snap.timestamp == 100.0

    def test_get_and_contains(self):
        snap = ConfigSnapshot({"x": 1}, version=1)
        assert snap.get("x") == 1
        assert snap.get("missing", "default") == "default"
        assert "x" in snap
        assert "missing" not in snap

    def test_getitem(self):
        snap = ConfigSnapshot({"key": "value"}, version=1)
        assert snap["key"] == "value"
        with pytest.raises(KeyError):
            _ = snap["nope"]

    def test_repr(self):
        snap = ConfigSnapshot({"a": 1, "b": 2}, version=3)
        assert "version=3" in repr(snap)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestLoadFile:
    def test_load_yaml(self, yaml_config: Path):
        data = _load_file(yaml_config)
        assert data["database"]["host"] == "localhost"

    def test_load_json(self, json_config: Path):
        data = _load_file(json_config)
        assert data["database"]["port"] == 5432

    def test_unsupported_extension(self, tmp_path: Path):
        p = tmp_path / "config.toml"
        p.write_text("[x]\ny=1")
        with pytest.raises(ValueError, match="Unsupported"):
            _load_file(p)

    def test_non_dict_raises(self, tmp_path: Path):
        p = tmp_path / "list.yaml"
        p.write_text("- 1\n- 2\n")
        with pytest.raises(ValueError, match="mapping"):
            _load_file(p)


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        base = {"db": {"host": "localhost", "port": 5432}}
        override = {"db": {"host": "prod-db"}}
        result = _deep_merge(base, override)
        assert result == {"db": {"host": "prod-db", "port": 5432}}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}


class TestComputeDiff:
    def test_added_key(self):
        diffs = compute_diff({}, {"a": 1})
        assert any("added" in d for d in diffs)

    def test_removed_key(self):
        diffs = compute_diff({"a": 1}, {})
        assert any("removed" in d for d in diffs)

    def test_changed_key(self):
        diffs = compute_diff({"a": 1}, {"a": 2})
        assert any("changed" in d for d in diffs)

    def test_nested_diff(self):
        diffs = compute_diff({"db": {"host": "a"}}, {"db": {"host": "b"}})
        assert any("db.host" in d for d in diffs)

    def test_no_diff(self):
        assert compute_diff({"a": 1}, {"a": 1}) == []


# ---------------------------------------------------------------------------
# HotReloader – core behaviour
# ---------------------------------------------------------------------------


class TestHotReloaderInit:
    def test_loads_yaml_on_init(self, yaml_config: Path):
        hr = HotReloader(yaml_config)
        assert hr.snapshot.version == 1
        assert hr.get("debug") is False

    def test_loads_json_on_init(self, json_config: Path):
        hr = HotReloader(json_config)
        assert hr.snapshot["database"]["host"] == "localhost"

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(Exception):
            HotReloader(tmp_path / "does_not_exist.yaml")

    def test_raises_on_invalid_initial_config(self, yaml_config: Path):
        def reject_all(cfg: dict[str, Any]) -> None:
            raise ValueError("nope")

        with pytest.raises(ConfigReloadError):
            HotReloader(yaml_config, validators=[reject_all])


class TestHotReloaderReload:
    def test_manual_reload_bumps_version(self, yaml_config: Path):
        hr = HotReloader(yaml_config)
        assert hr.version == 1
        hr.reload()
        assert hr.version == 2

    def test_reload_picks_up_changes(self, yaml_config: Path):
        hr = HotReloader(yaml_config)
        yaml_config.write_text(yaml.dump({"database": {"host": "new-host"}, "debug": True}))
        snap = hr.reload()
        assert snap["debug"] is True
        assert snap["database"]["host"] == "new-host"

    def test_rollback_on_invalid_config(self, yaml_config: Path):
        hr = HotReloader(yaml_config)
        original = hr.snapshot.data

        # Write invalid YAML content (not a mapping)
        yaml_config.write_text("- just\n- a\n- list\n")
        with pytest.raises(ConfigReloadError):
            hr.reload()

        # Config should remain unchanged
        assert hr.snapshot.data == original
        assert hr.version == 1

    def test_rollback_on_validation_failure(self, yaml_config: Path):
        def require_debug(cfg: dict[str, Any]) -> None:
            if "debug" not in cfg:
                raise ValueError("missing debug")

        hr = HotReloader(yaml_config, validators=[require_debug])
        yaml_config.write_text(yaml.dump({"database": {"host": "localhost"}}))

        with pytest.raises(ConfigReloadError):
            hr.reload()

        assert hr.version == 1
        assert "debug" in hr.snapshot


class TestHotReloaderCallbacks:
    def test_callback_invoked_on_change(self, yaml_config: Path):
        callback = MagicMock()
        hr = HotReloader(yaml_config, callbacks=[callback])

        yaml_config.write_text(yaml.dump({"debug": True}))
        hr.reload()

        callback.assert_called_once()
        old_snap, new_snap = callback.call_args[0]
        assert old_snap.version == 1
        assert new_snap.version == 2

    def test_callback_not_called_on_first_load(self, yaml_config: Path):
        callback = MagicMock()
        HotReloader(yaml_config, callbacks=[callback])
        callback.assert_not_called()

    def test_failing_callback_does_not_block(self, yaml_config: Path):
        def bad_callback(old: Any, new: Any) -> None:
            raise RuntimeError("boom")

        good_callback = MagicMock()
        hr = HotReloader(yaml_config, callbacks=[bad_callback, good_callback])
        yaml_config.write_text(yaml.dump({"debug": True}))
        hr.reload()

        # Good callback still called despite bad one raising
        good_callback.assert_called_once()

    def test_add_callback_dynamically(self, yaml_config: Path):
        hr = HotReloader(yaml_config)
        cb = MagicMock()
        hr.add_callback(cb)
        yaml_config.write_text(yaml.dump({"debug": True}))
        hr.reload()
        cb.assert_called_once()


class TestHotReloaderValidators:
    def test_add_validator_dynamically(self, yaml_config: Path):
        hr = HotReloader(yaml_config)
        hr.add_validator(lambda cfg: (_ for _ in ()).throw(ValueError("nope")))

        # Actually write a simple rejector
        hr._validators.clear()

        def reject(cfg: dict[str, Any]) -> None:
            raise ValueError("always fail")

        hr.add_validator(reject)

        with pytest.raises(ConfigReloadError):
            hr.reload()
        assert hr.version == 1


# ---------------------------------------------------------------------------
# Environment-specific overrides
# ---------------------------------------------------------------------------


class TestEnvironmentOverrides:
    def test_env_override_merges(self, tmp_path: Path):
        base = tmp_path / "config.yaml"
        base.write_text(yaml.dump({"database": {"host": "localhost", "port": 5432}, "debug": False}))

        prod = tmp_path / "config.production.yaml"
        prod.write_text(yaml.dump({"database": {"host": "prod-db"}, "debug": False}))

        hr = HotReloader(base, env="production")
        assert hr.get("database")["host"] == "prod-db"
        assert hr.get("database")["port"] == 5432  # kept from base

    def test_missing_env_override_ignored(self, yaml_config: Path):
        hr = HotReloader(yaml_config, env="staging")
        assert hr.snapshot.version == 1  # loaded fine without override


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_reads_during_reload(self, yaml_config: Path):
        hr = HotReloader(yaml_config)
        errors: list[Exception] = []

        def reader():
            for _ in range(50):
                try:
                    _ = hr.snapshot.data
                    _ = hr.get("debug")
                except Exception as exc:
                    errors.append(exc)

        def writer():
            for i in range(20):
                yaml_config.write_text(
                    yaml.dump({"database": {"host": f"host-{i}"}, "debug": True})
                )
                try:
                    hr.reload()
                except ConfigReloadError:
                    pass

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Got errors during concurrent access: {errors}"


# ---------------------------------------------------------------------------
# Watcher (start / stop)
# ---------------------------------------------------------------------------


class TestWatcher:
    def test_start_and_stop(self, yaml_config: Path):
        hr = HotReloader(yaml_config)
        assert not hr.watching

        with patch("watchfiles.watch") as mock_watch:
            # Make watch return immediately
            mock_watch.return_value = iter([])
            hr.start_watching(poll_interval=500)
            time.sleep(0.1)
            # Thread was started
            hr.stop_watching()
            assert not hr.watching

    def test_start_watching_idempotent(self, yaml_config: Path):
        hr = HotReloader(yaml_config)
        with patch("watchfiles.watch") as mock_watch:
            mock_watch.return_value = iter([])
            hr.start_watching()
            time.sleep(0.1)
            hr.start_watching()
            hr.stop_watching()


# ---------------------------------------------------------------------------
# Diff logging
# ---------------------------------------------------------------------------


class TestDiffLogging:
    def test_diff_logged_on_reload(self, yaml_config: Path, caplog):
        hr = HotReloader(yaml_config)
        yaml_config.write_text(yaml.dump({"database": {"host": "new-host"}, "debug": True}))

        import logging

        with caplog.at_level(logging.INFO, logger="shared.hot_reload"):
            hr.reload()

        assert any("Config updated" in r.message for r in caplog.records)
