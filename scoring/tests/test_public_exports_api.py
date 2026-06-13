from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "expected_names"),
    [
        ("scoring.api.health", {"ComponentStatus", "HealthResponse", "router"}),
        ("scoring.api.batch", {"BatchScoreRequest", "BatchScoreResponse", "router"}),
        ("scoring.api.jwt_auth", {"CurrentUser", "create_access_token", "router"}),
        ("scoring.api.versioning", {"APIVersion", "VersionNegotiator", "version_transform"}),
    ],
)
def test_api_modules_publish_expected_exports(module_name: str, expected_names: set[str]) -> None:
    module = importlib.import_module(module_name)

    assert hasattr(module, "__all__")
    assert expected_names.issubset(set(module.__all__))

    for name in expected_names:
        assert hasattr(module, name)
