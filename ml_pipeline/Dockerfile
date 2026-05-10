FROM python:3.12-slim AS builder

LABEL maintainer="fraud-detection-platform"

WORKDIR /app

# System dependencies for XGBoost, scipy, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY shared/ ./shared/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY shared/ shared/
COPY feature_store/ feature_store/
COPY ml_pipeline/ ml_pipeline/
COPY scripts/ scripts/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

# Default: run XGBoost training
ENTRYPOINT ["python", "-m"]
CMD ["ml_pipeline.train_xgboost", "--data", "data/training.parquet", "--output", "models/xgboost_v1.json"]
