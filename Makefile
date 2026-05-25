.PHONY: up down up-infra down-infra test test-unit test-integration lint format simulate logs clean \
       generate-data migrate migrate-down benchmark

# ── Docker ──────────────────────────────────────────────────

up:
	docker compose up -d --build

down:
	docker compose down -v

up-infra:
	docker compose -f docker-compose.infra.yml up -d

down-infra:
	docker compose -f docker-compose.infra.yml down -v

logs:
	docker compose logs -f

# ── Testing ─────────────────────────────────────────────────

test:
	python -m pytest --cov=shared --cov=scoring --cov=feature_store --cov=alert_service --cov=simulator --cov-report=term-missing -x

test-unit:
	python -m pytest scoring/tests/ feature_store/tests/ -x -q --ignore=scoring/tests/test_integration.py

test-integration:
	python -m pytest scoring/tests/test_integration.py -x -v -m integration

# ── Code Quality ────────────────────────────────────────────

lint:
	ruff check .
	mypy shared/ scoring/ feature_store/ alert_service/ simulator/

format:
	ruff format .
	ruff check --fix .

# ── Data & ML ───────────────────────────────────────────────

simulate:
	python -m simulator

generate-data:
	python scripts/generate_synthetic_data.py --num-users 1000 --num-merchants 200 --num-transactions 50000 --fraud-rate 0.03 --output-path data/synthetic/

# ── Database ────────────────────────────────────────────────

migrate:
	alembic -c migrations/alembic.ini upgrade head

migrate-down:
	alembic -c migrations/alembic.ini downgrade -1

migrate-history:
	alembic -c migrations/alembic.ini history --verbose

# ── Load Testing ────────────────────────────────────────────

benchmark:
	k6 run load_tests/k6_scoring.js

benchmark-e2e:
	k6 run load_tests/k6_e2e.js

benchmark-scale:
	k6 run load_tests/scale_test.js

# ── Cleanup ─────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/
