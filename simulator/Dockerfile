# ---------- builder ----------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libsnappy-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY shared/ ./shared/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libsnappy1v5 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY shared/ ./shared/
COPY simulator/ ./simulator/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "simulator"]
