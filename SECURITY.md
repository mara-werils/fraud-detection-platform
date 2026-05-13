# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to the maintainers. You should receive a response within 48 hours. If the issue is confirmed, we will release a patch as soon as possible.

### What to Include

- Type of vulnerability (e.g., SQL injection, XSS, authentication bypass)
- Full path of the affected source file(s)
- Step-by-step instructions to reproduce the issue
- Impact assessment
- Any suggested fix (optional)

## Security Measures

This platform implements the following security measures:

- **API authentication** — All scoring endpoints require API key authentication
- **Rate limiting** — Configurable per-key rate limits to prevent abuse
- **Input validation** — Pydantic v2 strict validation on all inputs
- **Request ID tracking** — Every request gets a unique ID for audit trails
- **Structured logging** — All actions are logged with structured JSON format
- **No secrets in code** — All credentials via environment variables
- **CORS configuration** — Configurable allowed origins

## Best Practices for Deployment

1. Always change default credentials (Redis, ClickHouse, MinIO, Grafana)
2. Use TLS for all external-facing services
3. Restrict network access to internal services (Kafka, Redis, ClickHouse)
4. Rotate API keys regularly
5. Enable Kubernetes network policies in production
6. Use secrets management (e.g., Vault) for production credentials
7. Review Prometheus/Grafana access controls
