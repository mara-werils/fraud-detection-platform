# Contributing to Fraud Detection Platform

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose v2
- Make
- Node.js 18+ (for dashboard)

### Local Development

```bash
# Clone and setup
git clone https://github.com/mara-werils/fraud-detection-platform.git
cd fraud-detection-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,gnn]"

# Install pre-commit hooks
pre-commit install

# Start infrastructure
make up-infra

# Run tests
make test
```

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/mara-werils/fraud-detection-platform/issues) first
2. Use the bug report template
3. Include steps to reproduce, expected vs actual behavior, and environment details

### Suggesting Features

1. Open a feature request issue
2. Describe the use case and expected behavior
3. Explain why this feature would benefit the platform

### Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes following the code style guidelines below
4. Write or update tests as needed
5. Run the test suite and linting:
   ```bash
   make test
   make lint
   ```
6. Commit with a clear message (see commit conventions below)
7. Push and open a pull request

## Code Style

### Python

- **Formatter**: Ruff (line length 100)
- **Linter**: Ruff with rules E, F, W, I, N, UP, B, A, SIM
- **Type checking**: mypy in strict mode
- **Docstrings**: Google style
- All new code must have type annotations

```bash
# Format code
make format

# Run linter and type checker
make lint
```

### TypeScript (Dashboard)

- Follow existing Next.js patterns in `dashboard/`
- Use TypeScript strict mode
- Prefer server components where possible

### Commit Conventions

Use clear, descriptive commit messages:

```
<type>: <short description>

<optional body explaining why>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

Examples:
```
feat: add batch scoring API endpoint
fix: resolve race condition in feature store writer
docs: update API reference with new endpoints
test: add integration tests for case management
```

## Project Structure

| Directory | Responsibility | Language |
|-----------|---------------|----------|
| `scoring/` | ML scoring service (FastAPI) | Python |
| `scoring/api/` | Routes, middleware, auth | Python |
| `scoring/models/` | ML models, ensemble, rules | Python |
| `scoring/services/` | Business logic (cases, search) | Python |
| `feature_store/` | Feature computation and serving | Python |
| `streaming/` | Kafka streaming pipeline | Python |
| `alert_service/` | Alert routing and notifications | Python |
| `dashboard/` | Real-time UI | TypeScript |
| `sdk/` | Python SDK client | Python |
| `cli/` | CLI tool | Python |
| `ml_pipeline/` | Model training | Python |
| `shared/` | Common schemas, utils | Python |
| `infra/` | Infrastructure as code | HCL/YAML |

## Testing

### Running Tests

```bash
# All tests
make test

# Specific test file
python -m pytest scoring/tests/test_scoring.py -v

# With coverage report
python -m pytest --cov=scoring --cov-report=html

# Only unit tests (skip integration)
python -m pytest -m "not integration"
```

### Writing Tests

- Place tests in `<service>/tests/` directories
- Use `pytest` and `pytest-asyncio` for async tests
- Use `fakeredis` for Redis-dependent tests
- Mark integration tests with `@pytest.mark.integration`
- Mark slow tests with `@pytest.mark.slow`

## Pull Request Process

1. Ensure all tests pass and linting is clean
2. Update documentation if needed
3. Add a description of changes in the PR
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested

## Architecture Decisions

For significant changes, please open an Architecture Decision Record (ADR):

1. Copy `docs/adr/template.md`
2. Number it sequentially (e.g., `004-your-decision.md`)
3. Include context, decision, consequences, and alternatives considered
4. Reference the ADR in your PR

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Review existing ADRs for architectural context

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
