from __future__ import annotations

import importlib


def test_hot_reload_module_publishes_expected_exports() -> None:
    hot_reload = importlib.import_module("shared.hot_reload")

    for name in {
        "ConfigReloadError",
        "ConfigSnapshot",
        "ConfigValidator",
        "HotReloader",
        "compute_diff",
    }:
        assert name in hot_reload.__all__
        assert hasattr(hot_reload, name)


def test_shared_package_reexports_hot_reload_primitives() -> None:
    shared = importlib.import_module("shared")

    for name in {"ConfigReloadError", "ConfigSnapshot", "HotReloader"}:
        assert name in shared.__all__
        assert hasattr(shared, name)
