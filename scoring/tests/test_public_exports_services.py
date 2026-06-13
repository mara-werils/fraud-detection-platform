from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "expected_names"),
    [
        ("scoring.services.feature_cache", {"LRUCache", "TwoLevelFeatureCache"}),
        ("scoring.services.redis_pool", {"CircuitBreaker", "RedisPoolManager"}),
        ("scoring.services.case_manager", {"CaseManager", "FraudCase"}),
        ("scoring.services.webhook_retry", {"RetryPolicy", "WebhookRetryService"}),
        ("scoring.services.ip_intelligence", {"GeoLocation", "IPIntelligenceService"}),
    ],
)
def test_service_modules_publish_expected_exports(module_name: str, expected_names: set[str]) -> None:
    module = importlib.import_module(module_name)

    assert hasattr(module, "__all__")
    assert expected_names.issubset(set(module.__all__))

    for name in expected_names:
        assert hasattr(module, name)


def test_services_package_reexports_curated_surface() -> None:
    services = importlib.import_module("scoring.services")

    for name in {"CaseManager", "DriftDetector", "FeedbackStore", "TransactionStore"}:
        assert name in services.__all__
        assert hasattr(services, name)
