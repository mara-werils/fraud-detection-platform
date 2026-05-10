## ---------------------------------------------------------------------------
## Stage 1: build dependencies
## ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install .

## ---------------------------------------------------------------------------
## Stage 2: runtime
## ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="fraud-detection-team"

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY shared/ ./shared/
COPY feature_store/ ./feature_store/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-m", "feature_store.main"]
