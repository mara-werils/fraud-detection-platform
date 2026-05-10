FROM python:3.11-slim AS base

LABEL maintainer="fraud-detection-platform"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies for XGBoost, scipy, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    xgboost>=2.0.0 \
    scikit-learn>=1.4.0 \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    pyarrow>=15.0.0 \
    structlog>=24.1.0 \
    orjson>=3.9.0 \
    pydantic>=2.0 \
    optuna>=3.5.0 \
    mlflow>=2.11.0 \
    shap>=0.45.0 \
    imbalanced-learn>=0.12.0 \
    plotly>=5.20.0

# Copy source code
COPY shared/ shared/
COPY feature_store/ feature_store/
COPY ml_pipeline/ ml_pipeline/
COPY scripts/ scripts/

# Default: run XGBoost training
ENTRYPOINT ["python", "-m"]
CMD ["ml_pipeline.train_xgboost", "--data", "data/training.parquet", "--output", "models/xgboost_v1.json"]
