FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY shared/ ./shared/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

FROM python:3.12-slim

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY shared/ ./shared/
COPY scoring/ ./scoring/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER app

EXPOSE 8000

CMD ["uvicorn", "scoring.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
