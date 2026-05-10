## ---------------------------------------------------------------------------
## Stage 1: build dependencies
## ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY shared/ ./shared/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

## ---------------------------------------------------------------------------
## Stage 2: runtime
## ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="fraud-detection-team"

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY shared/ ./shared/
COPY feature_store/ ./feature_store/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-m", "feature_store.main"]
