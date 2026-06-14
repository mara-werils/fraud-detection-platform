"""Tests for API key management module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from scoring.api.api_keys import (
    APIKey,
    APIKeyStore,
    Permission,
    _generate_raw_key,
    _hash_key,
    cleanup_expired_keys,
    create_api_key,
    expand_permissions,
    get_store,
    get_usage_analytics,
    record_usage,
    revoke_key,
    rotate_key,
    router,
    validate_key,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_store():
    """Ensure a clean store for every test."""
    store = get_store()
    store.clear()
    yield
    store.clear()


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Key generation and hashing ───────────────────────────────────────────────


class TestKeyGeneration:
    def test_generate_raw_key_default_prefix(self):
        key = _generate_raw_key()
        assert key.startswith("sk_live_")

    def test_generate_raw_key_custom_prefix(self):
        key = _generate_raw_key(prefix="sk_test")
        assert key.startswith("sk_test_")

    def test_generated_keys_are_unique(self):
        keys = {_generate_raw_key() for _ in range(100)}
        assert len(keys) == 100

    def test_hash_key_is_deterministic(self):
        assert _hash_key("hello") == _hash_key("hello")

    def test_hash_key_differs_for_different_inputs(self):
        assert _hash_key("key_a") != _hash_key("key_b")

    def test_hash_key_returns_hex_string(self):
        h = _hash_key("test")
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)


# ── Permission expansion ────────────────────────────────────────────────────


class TestPermissions:
    def test_read_scope_expands_to_read_only(self):
        assert expand_permissions({"read"}) == frozenset({"read"})

    def test_write_scope_implies_read(self):
        expanded = expand_permissions({"write"})
        assert "read" in expanded
        assert "write" in expanded

    def test_admin_scope_implies_all(self):
        expanded = expand_permissions({"admin"})
        assert expanded == frozenset({"read", "write", "admin"})

    def test_unknown_scopes_passed_through(self):
        expanded = expand_permissions({"read", "custom_scope"})
        assert "custom_scope" in expanded
        assert "read" in expanded


# ── APIKeyStore ──────────────────────────────────────────────────────────────


class TestAPIKeyStore:
    def test_add_and_get_by_id(self):
        store = APIKeyStore()
        key = APIKey(
            id="k1", name="test", key_hash="hash1", prefix="sk_...",
            scopes={"read"}, rate_limit_rpm=100,
            created_at=datetime.now(UTC), expires_at=None,
        )
        store.add(key)
        assert store.get_by_id("k1") is key

    def test_get_by_hash(self):
        store = APIKeyStore()
        key = APIKey(
            id="k1", name="test", key_hash="hash1", prefix="sk_...",
            scopes={"read"}, rate_limit_rpm=100,
            created_at=datetime.now(UTC), expires_at=None,
        )
        store.add(key)
        assert store.get_by_hash("hash1") is key

    def test_get_missing_returns_none(self):
        store = APIKeyStore()
        assert store.get_by_id("nope") is None
        assert store.get_by_hash("nope") is None

    def test_remove(self):
        store = APIKeyStore()
        key = APIKey(
            id="k1", name="test", key_hash="hash1", prefix="sk_...",
            scopes={"read"}, rate_limit_rpm=100,
            created_at=datetime.now(UTC), expires_at=None,
        )
        store.add(key)
        assert store.remove("k1") is True
        assert store.get_by_id("k1") is None
        assert store.get_by_hash("hash1") is None

    def test_remove_missing_returns_false(self):
        store = APIKeyStore()
        assert store.remove("nope") is False

    def test_list_all(self):
        store = APIKeyStore()
        for i in range(3):
            store.add(APIKey(
                id=f"k{i}", name=f"test{i}", key_hash=f"hash{i}", prefix="sk_...",
                scopes={"read"}, rate_limit_rpm=100,
                created_at=datetime.now(UTC), expires_at=None,
            ))
        assert len(store.list_all()) == 3

    def test_clear(self):
        store = APIKeyStore()
        store.add(APIKey(
            id="k1", name="test", key_hash="hash1", prefix="sk_...",
            scopes={"read"}, rate_limit_rpm=100,
            created_at=datetime.now(UTC), expires_at=None,
        ))
        store.clear()
        assert len(store.list_all()) == 0


# ── Core operations ──────────────────────────────────────────────────────────


class TestCreateAPIKey:
    def test_create_returns_raw_key_and_record(self):
        raw_key, record = create_api_key(name="my-key")
        assert raw_key.startswith("sk_live_")
        assert record.name == "my-key"
        assert record.is_active is True

    def test_create_default_scopes(self):
        _, record = create_api_key(name="default")
        assert record.scopes == {Permission.READ}

    def test_create_custom_scopes(self):
        _, record = create_api_key(name="rw", scopes={"read", "write"})
        assert record.scopes == {"read", "write"}

    def test_create_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="Invalid scopes"):
            create_api_key(name="bad", scopes={"read", "destroy"})

    def test_create_with_expiration(self):
        _, record = create_api_key(name="expiring", expires_in_days=30)
        assert record.expires_at is not None
        delta = record.expires_at - record.created_at
        assert 29 <= delta.days <= 30

    def test_create_without_expiration(self):
        _, record = create_api_key(name="forever")
        assert record.expires_at is None

    def test_create_custom_rate_limit(self):
        _, record = create_api_key(name="fast", rate_limit_rpm=5000)
        assert record.rate_limit_rpm == 5000

    def test_created_key_is_in_store(self):
        _, record = create_api_key(name="stored")
        assert get_store().get_by_id(record.id) is record


class TestValidateKey:
    def test_valid_key_returns_record(self):
        raw, record = create_api_key(name="valid")
        assert validate_key(raw) is record

    def test_invalid_key_returns_none(self):
        assert validate_key("sk_live_nonexistent") is None

    def test_revoked_key_returns_none(self):
        raw, record = create_api_key(name="revoked")
        revoke_key(record.id)
        assert validate_key(raw) is None

    def test_expired_key_returns_none_and_auto_revokes(self):
        raw, record = create_api_key(name="expired", expires_in_days=1)
        # Manually set expiration to the past
        record.expires_at = datetime.now(UTC) - timedelta(hours=1)
        assert validate_key(raw) is None
        assert record.is_active is False
        assert record.revoked_at is not None

    def test_grace_period_key_still_valid(self):
        raw, record = create_api_key(name="grace")
        record.is_active = False
        record.grace_period_end = datetime.now(UTC) + timedelta(hours=12)
        assert validate_key(raw) is record

    def test_grace_period_expired_returns_none(self):
        raw, record = create_api_key(name="past-grace")
        record.is_active = False
        record.grace_period_end = datetime.now(UTC) - timedelta(hours=1)
        assert validate_key(raw) is None


class TestRevokeKey:
    def test_revoke_existing_key(self):
        _, record = create_api_key(name="to-revoke")
        assert revoke_key(record.id) is True
        assert record.is_active is False
        assert record.revoked_at is not None

    def test_revoke_missing_key(self):
        assert revoke_key("nonexistent") is False


class TestRotateKey:
    def test_rotate_returns_new_key(self):
        raw_old, old_record = create_api_key(name="rotate-me", scopes={"read", "write"})
        result = rotate_key(old_record.id, grace_period_hours=48)
        assert result is not None
        new_raw, new_record = result

        assert new_raw != raw_old
        assert new_record.name == old_record.name
        assert new_record.scopes == old_record.scopes
        assert new_record.rotated_from == old_record.id

    def test_rotate_old_key_has_grace_period(self):
        _, old_record = create_api_key(name="rotate-grace")
        rotate_key(old_record.id, grace_period_hours=24)

        assert old_record.is_active is False
        assert old_record.grace_period_end is not None
        assert old_record.grace_period_end > datetime.now(UTC)

    def test_rotate_preserves_rate_limit(self):
        _, old_record = create_api_key(name="r", rate_limit_rpm=5000)
        _, new_record = rotate_key(old_record.id)
        assert new_record.rate_limit_rpm == 5000

    def test_rotate_missing_key_returns_none(self):
        assert rotate_key("nonexistent") is None

    def test_old_key_valid_during_grace_period(self):
        raw_old, old_record = create_api_key(name="grace-test")
        rotate_key(old_record.id, grace_period_hours=24)
        # Old key should still validate during grace period
        assert validate_key(raw_old) is not None


# ── Usage analytics ──────────────────────────────────────────────────────────


class TestUsageAnalytics:
    def test_record_usage_increments(self):
        _, record = create_api_key(name="usage")
        record_usage(record, "/score")
        record_usage(record, "/score")
        record_usage(record, "/batch")
        assert record.usage.total_requests == 3
        assert record.usage.requests_by_endpoint["/score"] == 2
        assert record.usage.requests_by_endpoint["/batch"] == 1

    def test_record_usage_tracks_errors(self):
        _, record = create_api_key(name="errors")
        record_usage(record, "/score", is_error=False)
        record_usage(record, "/score", is_error=True)
        assert record.usage.errors == 1
        assert record.usage.total_requests == 2

    def test_record_usage_updates_last_used(self):
        _, record = create_api_key(name="ts")
        assert record.usage.last_used is None
        record_usage(record, "/score")
        assert record.usage.last_used is not None

    def test_get_usage_analytics(self):
        _, record = create_api_key(name="analytics")
        record_usage(record, "/score")
        record_usage(record, "/score", is_error=True)

        analytics = get_usage_analytics(record.id)
        assert analytics is not None
        assert analytics["total_requests"] == 2
        assert analytics["errors"] == 1
        assert analytics["error_rate"] == 0.5
        assert analytics["requests_by_endpoint"] == {"/score": 2}

    def test_get_usage_analytics_missing_key(self):
        assert get_usage_analytics("nope") is None

    def test_error_rate_zero_when_no_requests(self):
        _, record = create_api_key(name="no-usage")
        analytics = get_usage_analytics(record.id)
        assert analytics["error_rate"] == 0.0


# ── Cleanup ──────────────────────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_revokes_expired_keys(self):
        _, record = create_api_key(name="expired", expires_in_days=1)
        record.expires_at = datetime.now(UTC) - timedelta(hours=1)
        cleaned = cleanup_expired_keys()
        assert cleaned == 1
        assert record.is_active is False

    def test_cleanup_removes_past_grace_period_keys(self):
        _, record = create_api_key(name="past-grace")
        record.is_active = False
        record.grace_period_end = datetime.now(UTC) - timedelta(hours=1)
        cleaned = cleanup_expired_keys()
        assert cleaned == 1
        assert get_store().get_by_id(record.id) is None

    def test_cleanup_ignores_active_keys(self):
        create_api_key(name="active")
        cleaned = cleanup_expired_keys()
        assert cleaned == 0


# ── APIKey model properties ──────────────────────────────────────────────────


class TestAPIKeyProperties:
    def test_is_expired_false_when_no_expiry(self):
        _, record = create_api_key(name="no-exp")
        assert record.is_expired is False

    def test_is_expired_true_when_past(self):
        _, record = create_api_key(name="exp", expires_in_days=1)
        record.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert record.is_expired is True

    def test_is_in_grace_period(self):
        _, record = create_api_key(name="gp")
        record.grace_period_end = datetime.now(UTC) + timedelta(hours=1)
        assert record.is_in_grace_period is True

    def test_not_in_grace_period_when_none(self):
        _, record = create_api_key(name="no-gp")
        assert record.is_in_grace_period is False

    def test_has_permission_admin_has_all(self):
        _, record = create_api_key(name="admin", scopes={"admin"})
        assert record.has_permission("read")
        assert record.has_permission("write")
        assert record.has_permission("admin")

    def test_has_permission_read_only(self):
        _, record = create_api_key(name="ro", scopes={"read"})
        assert record.has_permission("read")
        assert not record.has_permission("write")
        assert not record.has_permission("admin")


# ── FastAPI endpoint tests ───────────────────────────────────────────────────


class TestCreateKeyEndpoint:
    def test_create_key_201(self, client: TestClient):
        resp = client.post("/api-keys", json={"name": "test-key"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["raw_key"].startswith("sk_live_")
        assert data["name"] == "test-key"
        assert "read" in data["scopes"]

    def test_create_key_custom_scopes(self, client: TestClient):
        resp = client.post("/api-keys", json={
            "name": "admin-key",
            "scopes": ["admin"],
            "rate_limit_rpm": 5000,
        })
        assert resp.status_code == 201
        assert "admin" in resp.json()["scopes"]

    def test_create_key_invalid_scope_400(self, client: TestClient):
        resp = client.post("/api-keys", json={
            "name": "bad",
            "scopes": ["read", "destroy"],
        })
        assert resp.status_code == 400

    def test_create_key_with_expiration(self, client: TestClient):
        resp = client.post("/api-keys", json={
            "name": "expiring",
            "expires_in_days": 90,
        })
        assert resp.status_code == 201
        assert resp.json()["expires_at"] is not None


class TestListKeysEndpoint:
    def test_list_empty(self, client: TestClient):
        resp = client.get("/api-keys")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_created_keys(self, client: TestClient):
        client.post("/api-keys", json={"name": "a"})
        client.post("/api-keys", json={"name": "b"})
        resp = client.get("/api-keys")
        assert len(resp.json()) == 2

    def test_list_active_only(self, client: TestClient):
        r = client.post("/api-keys", json={"name": "active"})
        key_id = r.json()["key_id"]
        client.post("/api-keys", json={"name": "revoked"})
        revoked_id = client.get("/api-keys").json()[1]["key_id"]
        client.delete(f"/api-keys/{revoked_id}")

        resp = client.get("/api-keys", params={"active_only": True})
        ids = [k["key_id"] for k in resp.json()]
        assert key_id in ids


class TestGetKeyEndpoint:
    def test_get_existing_key(self, client: TestClient):
        r = client.post("/api-keys", json={"name": "lookup"})
        key_id = r.json()["key_id"]
        resp = client.get(f"/api-keys/{key_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "lookup"

    def test_get_missing_key_404(self, client: TestClient):
        resp = client.get("/api-keys/nonexistent")
        assert resp.status_code == 404


class TestUpdateKeyEndpoint:
    def test_update_name(self, client: TestClient):
        r = client.post("/api-keys", json={"name": "old-name"})
        key_id = r.json()["key_id"]
        resp = client.patch(f"/api-keys/{key_id}", json={"name": "new-name"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "new-name"

    def test_update_scopes(self, client: TestClient):
        r = client.post("/api-keys", json={"name": "scope-test"})
        key_id = r.json()["key_id"]
        resp = client.patch(f"/api-keys/{key_id}", json={"scopes": ["admin"]})
        assert resp.status_code == 200
        assert "admin" in resp.json()["scopes"]

    def test_update_invalid_scope_400(self, client: TestClient):
        r = client.post("/api-keys", json={"name": "s"})
        key_id = r.json()["key_id"]
        resp = client.patch(f"/api-keys/{key_id}", json={"scopes": ["nuke"]})
        assert resp.status_code == 400

    def test_update_missing_key_404(self, client: TestClient):
        resp = client.patch("/api-keys/missing", json={"name": "x"})
        assert resp.status_code == 404

    def test_update_rate_limit(self, client: TestClient):
        r = client.post("/api-keys", json={"name": "rl"})
        key_id = r.json()["key_id"]
        resp = client.patch(f"/api-keys/{key_id}", json={"rate_limit_rpm": 9999})
        assert resp.status_code == 200
        assert resp.json()["rate_limit_rpm"] == 9999


class TestDeleteKeyEndpoint:
    def test_delete_existing_key(self, client: TestClient):
        r = client.post("/api-keys", json={"name": "delete-me"})
        key_id = r.json()["key_id"]
        resp = client.delete(f"/api-keys/{key_id}")
        assert resp.status_code == 204

        resp = client.get(f"/api-keys/{key_id}")
        assert resp.status_code == 404

    def test_delete_missing_key_404(self, client: TestClient):
        resp = client.delete("/api-keys/nonexistent")
        assert resp.status_code == 404


class TestRotateKeyEndpoint:
    def test_rotate_key(self, client: TestClient):
        r = client.post("/api-keys", json={"name": "rotate"})
        key_id = r.json()["key_id"]
        resp = client.post(f"/api-keys/{key_id}/rotate", json={"grace_period_hours": 48})
        assert resp.status_code == 200
        data = resp.json()
        assert data["old_key_id"] == key_id
        assert data["new_key_id"] != key_id
        assert data["new_raw_key"].startswith("sk_")

    def test_rotate_missing_key_404(self, client: TestClient):
        resp = client.post("/api-keys/missing/rotate", json={})
        assert resp.status_code == 404


class TestUsageEndpoint:
    def test_usage_for_new_key(self, client: TestClient):
        r = client.post("/api-keys", json={"name": "usage"})
        key_id = r.json()["key_id"]
        resp = client.get(f"/api-keys/{key_id}/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 0
        assert data["error_rate"] == 0.0

    def test_usage_missing_key_404(self, client: TestClient):
        resp = client.get("/api-keys/nonexistent/usage")
        assert resp.status_code == 404


class TestCleanupEndpoint:
    def test_cleanup_returns_count(self, client: TestClient):
        resp = client.post("/api-keys/cleanup")
        assert resp.status_code == 200
        assert resp.json()["cleaned_up"] == 0
