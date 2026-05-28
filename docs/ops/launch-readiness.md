# Launch Readiness Checklist

This checklist tracks practical launch hardening tasks for the fraud detection platform.

- [ ] Confirm `docker-compose.yml` and `docker-compose.infra.yml` boot on a clean machine.
- [ ] Verify scoring API `/health` responds `ok` within SLO after cold start.
- [ ] Run critical scoring tests before release.
- [ ] Validate DB migrations apply and rollback paths are documented.
- [ ] Confirm Kafka topics exist with correct retention and partition settings.
- [ ] Validate Redis connectivity and eviction policy for scoring cache.
- [ ] Check feature store online/offline parity for required features.
- [ ] Ensure model artifact version used in production is pinned and traceable.
- [ ] Verify model registry metadata includes owner and rollback version.
- [ ] Run canary traffic through shadow mode and compare score drift.
- [ ] Confirm rate limiter thresholds for public APIs are sane for launch volume.
- [ ] Validate JWT auth configuration and token expiry settings.
- [ ] Ensure RBAC policies block unauthorized access paths.
- [ ] Verify sanctions and compliance endpoints return deterministic responses.
- [ ] Check webhook retries and dead-letter behavior under downstream failures.
- [ ] Confirm audit logs contain actor, action, and correlation id.
- [ ] Validate PII masking rules in logs and exports.
- [ ] Run batch queue smoke test with success/failure callbacks.
- [ ] Verify anomaly detector fallback behavior when model dependency is absent.
- [ ] Confirm rule engine v1/v2 compatibility for legacy tenants.
- [ ] Validate entity list import/export and deduplication paths.
- [ ] Confirm case management workflow updates status transitions correctly.
- [ ] Ensure analytics endpoints stay within acceptable response times.
- [ ] Validate dashboard API integration against current schema.
- [ ] Confirm TypeScript SDK builds and basic client call works.
- [ ] Check Python SDK install and sample request/response flow.
- [ ] Review alert routing escalation rules for high-risk events.
- [ ] Verify observability stack collects app, DB, and queue metrics.
- [ ] Confirm alert thresholds for latency/error budget burn are tuned.
- [ ] Document release rollback runbook and owner on-call rotation.
