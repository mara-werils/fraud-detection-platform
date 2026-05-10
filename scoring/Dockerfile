FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY shared/ ./shared/
COPY scoring/ ./scoring/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install \
    fastapi uvicorn[standard] pydantic pydantic-settings \
    aiokafka orjson structlog redis[hiredis] \
    prometheus-client python-snappy

FROM python:3.12-slim

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY shared/ ./shared/
COPY scoring/ ./scoring/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER app

EXPOSE 8000

CMD ["uvicorn", "scoring.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
