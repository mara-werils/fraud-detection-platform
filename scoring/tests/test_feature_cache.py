from __future__ import annotations

import json

from scoring.services.feature_cache import LRUCache, _deserialize_cached_value


def test_lru_cache_clamps_non_positive_ttl() -> None:
    cache = LRUCache(max_size=10, default_ttl=30)
    cache.set("k", "v", ttl=0)
    assert cache.get("k") == "v"


def test_deserialize_cached_value_supports_bytes() -> None:
    payload = {"risk": 0.42, "country": "US"}
    raw = json.dumps(payload).encode("utf-8")
    assert _deserialize_cached_value(raw) == payload
