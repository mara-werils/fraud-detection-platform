.PHONY: up down up-infra down-infra test lint format simulate logs clean

up:
	docker compose up -d --build

down:
	docker compose down -v

up-infra:
	docker compose -f docker-compose.infra.yml up -d

down-infra:
	docker compose -f docker-compose.infra.yml down -v

test:
	python -m pytest --cov=shared --cov=scoring --cov=feature_store --cov=alert_service --cov=simulator --cov-report=term-missing tests/

lint:
	ruff check .
	mypy shared/ scoring/ feature_store/ alert_service/ simulator/

format:
	ruff format .

simulate:
	python -m simulator

logs:
	docker compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/
