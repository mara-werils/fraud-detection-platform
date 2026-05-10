# Real-Time AI Fraud Detection Platform

## Цель проекта
Построить production-grade платформу обнаружения финансового мошенничества в реальном времени. Система принимает поток транзакций, обогащает их фичами, прогоняет через ML-модели (XGBoost + GNN + LLM explainer) и выдаёт решение за <100ms. Аналог систем Kaspi, Stripe, Revolut.

**GitHub repo:** `mara-werils/fraud-detection-platform`

---

## Архитектура

```
                         ┌──────────────┐
                         │  Transaction │
                         │  Simulator   │
                         │  (Python)    │
                         └──────┬───────┘
                                │ JSON events
                                ▼
                         ┌──────────────┐
                         │    Kafka     │
                         │  (3 topics)  │
                         │  - raw_txn   │
                         │  - scored    │
                         │  - alerts    │
                         └──────┬───────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
             ┌────────────┐ ┌────────┐ ┌────────────┐
             │  Flink /   │ │ Feature│ │  Archiver  │
             │  Spark     │ │ Store  │ │  (to Data  │
             │  Streaming │ │ Writer │ │   Lake)    │
             └─────┬──────┘ └───┬────┘ └─────┬──────┘
                   │            │             │
                   ▼            ▼             ▼
             ┌────────────┐ ┌────────┐ ┌────────────┐
             │  Scoring   │ │ Redis  │ │ MinIO (S3) │
             │  Service   │ │ (online│ │ + Delta    │
             │  (FastAPI) │ │ feat.) │ │   Lake     │
             │            │ └────────┘ └────────────┘
             │ - XGBoost  │ ┌────────┐
             │ - GNN      │ │ Click- │
             │ - LLM      │ │ House  │
             │   explainer│ │(offline│
             └─────┬──────┘ │ feat.) │
                   │        └────────┘
          ┌────────┼────────┐
          ▼        ▼        ▼
    ┌──────────┐ ┌──────┐ ┌──────────┐
    │ Alert    │ │Kafka │ │ Dashboard│
    │ Service  │ │scored│ │ (Next.js)│
    │(Telegram)│ │topic │ │ WebSocket│
    └──────────┘ └──────┘ └──────────┘

    ┌─────────────────────────────────────────┐
    │           INFRASTRUCTURE                │
    │  Airflow (orchestration)                │
    │  dbt (data transforms)                  │
    │  MLflow (experiment tracking)           │
    │  Prometheus + Grafana (monitoring)      │
    │  Jaeger (distributed tracing)           │
    │  K8s + Terraform + ArgoCD (deploy)      │
    │  GitHub Actions (CI/CD)                 │
    └─────────────────────────────────────────┘
```

---

## Структура репозитория

```
fraud-detection-platform/
├── PROJECT_PLAN.md                    # Этот файл
├── README.md
├── docker-compose.yml                 # Локальный запуск всех сервисов
├── docker-compose.infra.yml           # Kafka, Redis, ClickHouse, MinIO, Prometheus, Grafana
├── Makefile                           # make up, make down, make test, make train, make simulate
├── pyproject.toml                     # Monorepo Python deps
├── .github/
│   └── workflows/
│       ├── ci.yml                     # lint + test + build
│       ├── cd.yml                     # deploy to K8s via ArgoCD
│       └── model-training.yml         # scheduled model retraining
│
├── simulator/                         # Сервис 1: Генератор транзакций
│   ├── Dockerfile
│   ├── main.py                        # Kafka producer, генерирует поток транзакций
│   ├── schemas.py                     # Pydantic модель транзакции
│   ├── fraud_patterns.py              # Паттерны мошенничества (для симуляции)
│   └── config.py
│
├── streaming/                         # Сервис 2: Потоковая обработка
│   ├── Dockerfile
│   ├── flink_job.py                   # Flink/Spark Structured Streaming job
│   ├── enrichment.py                  # Обогащение транзакций (IP geolocation, device fingerprint)
│   ├── windowed_aggregates.py         # Скользящие окна (txn count/sum за 1m, 5m, 1h, 24h)
│   └── config.py
│
├── feature_store/                     # Сервис 3: Feature Store
│   ├── Dockerfile
│   ├── online_store.py                # Redis — real-time фичи (last N txn, velocity, etc.)
│   ├── offline_store.py               # ClickHouse — исторические фичи
│   ├── feature_registry.py            # Реестр фич с версионированием
│   └── config.py
│
├── scoring/                           # Сервис 4: ML Scoring (ядро системы)
│   ├── Dockerfile
│   ├── main.py                        # FastAPI app
│   ├── api/
│   │   ├── routes.py                  # POST /score, GET /health, GET /model/info
│   │   └── middleware.py              # Tracing, latency metrics
│   ├── models/
│   │   ├── xgboost_scorer.py          # XGBoost — основная модель (<10ms)
│   │   ├── gnn_scorer.py              # GNN — граф связей (PyTorch Geometric)
│   │   ├── llm_explainer.py           # LLM — объяснение решения (async, не блокирует)
│   │   └── ensemble.py                # Взвешенное объединение скоров
│   ├── ab_testing.py                  # A/B тестирование моделей
│   ├── config.py
│   └── tests/
│       ├── test_scoring.py
│       ├── test_ensemble.py
│       └── conftest.py
│
├── alert_service/                     # Сервис 5: Алерты
│   ├── Dockerfile
│   ├── main.py                        # Kafka consumer → Telegram/webhook
│   ├── notifiers/
│   │   ├── telegram.py
│   │   ├── webhook.py
│   │   └── email.py
│   └── config.py
│
├── dashboard/                         # Сервис 6: Real-time UI
│   ├── Dockerfile
│   ├── package.json
│   ├── next.config.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx               # Главная: live fraud map + метрики
│   │   │   ├── transactions/page.tsx  # Таблица транзакций (real-time)
│   │   │   ├── analytics/page.tsx     # Графики и аналитика
│   │   │   └── models/page.tsx        # Статус моделей, A/B результаты
│   │   ├── components/
│   │   │   ├── FraudMap.tsx           # Карта мошенничества (leaflet/mapbox)
│   │   │   ├── TransactionFeed.tsx    # WebSocket live feed
│   │   │   ├── ScoreGauge.tsx         # Визуализация скора
│   │   │   ├── MetricsCards.tsx       # KPI карточки
│   │   │   └── ModelComparison.tsx    # A/B dashboard
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts        # WebSocket hook для real-time данных
│   │   └── lib/
│   │       └── api.ts                 # API client
│   └── tsconfig.json
│
├── data_lake/                         # Сервис 7: Data Lake + Warehouse
│   ├── archiver/
│   │   ├── Dockerfile
│   │   └── main.py                    # Kafka → MinIO (Parquet + Delta Lake)
│   ├── dbt_project/
│   │   ├── dbt_project.yml
│   │   ├── profiles.yml
│   │   ├── models/
│   │   │   ├── staging/
│   │   │   │   ├── stg_transactions.sql
│   │   │   │   └── stg_fraud_labels.sql
│   │   │   ├── intermediate/
│   │   │   │   ├── int_daily_aggregates.sql
│   │   │   │   └── int_user_profiles.sql
│   │   │   └── marts/
│   │   │       ├── fraud_analytics.sql
│   │   │       └── model_performance.sql
│   │   └── tests/
│   │       └── assert_no_null_scores.sql
│   └── clickhouse/
│       └── init.sql                   # DDL для ClickHouse таблиц
│
├── ml_pipeline/                       # ML Training Pipeline
│   ├── Dockerfile
│   ├── train_xgboost.py               # Обучение XGBoost
│   ├── train_gnn.py                   # Обучение GNN (PyTorch Geometric)
│   ├── evaluate.py                    # Метрики: precision, recall, F1, AUC-ROC, PR-AUC
│   ├── data_prep.py                   # Подготовка данных из ClickHouse / Delta Lake
│   ├── feature_engineering.py         # Feature engineering pipeline
│   ├── hyperopt_search.py             # Optuna гиперпараметры
│   └── mlflow_registry.py            # Регистрация модели в MLflow
│
├── orchestration/                     # Airflow DAGs
│   ├── dags/
│   │   ├── daily_retrain.py           # Ежедневное дообучение моделей
│   │   ├── feature_backfill.py        # Бэкфилл фич в ClickHouse
│   │   ├── dbt_run.py                 # dbt transforms
│   │   └── data_quality_check.py      # Great Expectations проверки
│   └── plugins/
│
├── infra/                             # Infrastructure as Code
│   ├── terraform/
│   │   ├── main.tf                    # Провайдер + modules
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   ├── modules/
│   │   │   ├── k8s_cluster/           # Managed K8s (EKS/GKE)
│   │   │   ├── kafka/                 # MSK / Confluent
│   │   │   ├── redis/                 # ElastiCache
│   │   │   ├── s3/                    # MinIO / S3
│   │   │   ├── clickhouse/            # ClickHouse Cloud
│   │   │   └── monitoring/            # Prometheus + Grafana
│   │   └── environments/
│   │       ├── dev.tfvars
│   │       └── prod.tfvars
│   ├── k8s/
│   │   ├── namespaces.yaml
│   │   ├── scoring/
│   │   │   ├── deployment.yaml        # 3 replicas, resource limits
│   │   │   ├── service.yaml
│   │   │   ├── hpa.yaml               # Autoscaling по CPU/latency
│   │   │   └── pdb.yaml               # Pod disruption budget
│   │   ├── streaming/
│   │   │   ├── deployment.yaml
│   │   │   └── service.yaml
│   │   ├── dashboard/
│   │   │   ├── deployment.yaml
│   │   │   └── ingress.yaml
│   │   └── monitoring/
│   │       ├── prometheus-values.yaml
│   │       ├── grafana-dashboards/
│   │       │   ├── fraud-overview.json
│   │       │   ├── model-performance.json
│   │       │   └── system-health.json
│   │       └── alertmanager-rules.yaml
│   └── argocd/
│       ├── application.yaml           # ArgoCD Application
│       └── appset.yaml                # ApplicationSet для multi-env
│
├── observability/                     # Мониторинг и трейсинг
│   ├── prometheus/
│   │   └── prometheus.yml
│   ├── grafana/
│   │   └── provisioning/
│   │       ├── datasources.yaml
│   │       └── dashboards.yaml
│   ├── jaeger/
│   │   └── jaeger.yml
│   └── custom_metrics.py             # Prometheus metrics для scoring service
│
├── load_tests/                        # Нагрузочное тестирование
│   ├── k6_scoring.js                  # k6 сценарий: 10K RPS на /score
│   ├── k6_e2e.js                      # E2E: simulator → kafka → score → alert
│   └── results/
│
├── scripts/
│   ├── seed_data.py                   # Загрузка тестовых данных
│   ├── generate_dataset.py            # Генерация обучающего датасета
│   └── migrate_db.py                  # Миграции ClickHouse
│
└── docs/
    ├── architecture.md                # C4 диаграмма архитектуры
    ├── data_model.md                  # Схема данных
    ├── ml_models.md                   # Описание моделей и метрик
    ├── runbook.md                     # Operational runbook
    └── adr/                           # Architecture Decision Records
        ├── 001-kafka-over-rabbitmq.md
        ├── 002-clickhouse-for-olap.md
        └── 003-gnn-for-graph-fraud.md
```

---

## Модель данных

### Transaction (Kafka raw_txn topic)
```json
{
  "txn_id": "uuid",
  "user_id": "uuid",
  "merchant_id": "uuid",
  "amount": 45000.00,
  "currency": "KZT",
  "category": "electronics",
  "channel": "mobile_app",
  "ip_address": "185.23.xx.xx",
  "device_fingerprint": "abc123",
  "geo_lat": 51.1694,
  "geo_lon": 71.4491,
  "timestamp": "2026-05-10T14:23:01Z",
  "card_type": "visa",
  "is_international": false
}
```

### Scored Transaction (Kafka scored topic)
```json
{
  "txn_id": "uuid",
  "fraud_score": 0.87,
  "xgboost_score": 0.82,
  "gnn_score": 0.91,
  "decision": "BLOCK",
  "explanation": "Unusual high-value transaction from new device in different city, connected to known fraud cluster",
  "features_used": {
    "txn_velocity_1h": 12,
    "avg_amount_30d": 5000,
    "amount_deviation": 8.5,
    "new_device": true,
    "geo_distance_km": 1200
  },
  "latency_ms": 47,
  "model_version": "v2.3.1",
  "ab_group": "challenger"
}
```

### Feature Store Schema (Redis — online)
```
user:{user_id}:txn_count_1h     → int
user:{user_id}:txn_count_24h    → int
user:{user_id}:txn_sum_1h       → float
user:{user_id}:avg_amount_30d   → float
user:{user_id}:last_geo         → "lat,lon"
user:{user_id}:devices          → Set[str]
user:{user_id}:merchants_7d     → Set[str]
```

### ClickHouse Tables (offline)
```sql
CREATE TABLE transactions (
    txn_id        UUID,
    user_id       UUID,
    merchant_id   UUID,
    amount        Decimal(18,2),
    currency      LowCardinality(String),
    category      LowCardinality(String),
    channel       LowCardinality(String),
    geo_lat       Float64,
    geo_lon       Float64,
    fraud_score   Float32,
    decision      LowCardinality(String),  -- ALLOW / REVIEW / BLOCK
    is_fraud      UInt8,                   -- label (0/1)
    created_at    DateTime64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (user_id, created_at);

CREATE TABLE user_profiles (
    user_id              UUID,
    total_txn_count      UInt64,
    total_txn_amount     Decimal(18,2),
    avg_txn_amount       Decimal(18,2),
    fraud_count          UInt32,
    fraud_rate           Float32,
    first_txn_date       Date,
    last_txn_date        Date,
    unique_merchants     UInt32,
    unique_devices       UInt32,
    primary_geo          String,
    risk_tier            LowCardinality(String),  -- LOW / MEDIUM / HIGH
    updated_at           DateTime64(3)
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY user_id;
```

---

## ML модели — детальный план

### Модель 1: XGBoost (primary scorer)
- **Задача:** Бинарная классификация (fraud / legit)
- **Латентность:** <10ms
- **Фичи (30+):**
  - Transaction: amount, currency, category, channel, is_international
  - Velocity: txn_count_1m, txn_count_5m, txn_count_1h, txn_count_24h
  - Amount: amount_deviation (от среднего за 30 дней), max_amount_7d, sum_1h
  - Geo: distance_from_last_txn, distance_from_home, is_new_city
  - Device: is_new_device, device_age_days, device_txn_count
  - Merchant: is_new_merchant, merchant_fraud_rate, merchant_category_risk
  - Time: hour_of_day, day_of_week, is_weekend, is_night
  - Behavioral: time_since_last_txn, avg_time_between_txn
- **Метрики целевые:** AUC-ROC > 0.95, Precision@95%Recall > 0.80
- **Обучение:** Optuna hyperparameter search, 5-fold CV
- **Данные:** Генерируем синтетический датасет (100K txn, 2% fraud rate)

### Модель 2: GNN (graph fraud detection)
- **Задача:** Node classification на графе транзакций
- **Библиотека:** PyTorch Geometric
- **Граф:**
  - Nodes: users + merchants
  - Edges: transactions (amount, timestamp как edge features)
  - Node features: user_profile + merchant_profile
- **Архитектура:** GraphSAGE (2-3 layers) → MLP classifier
- **Зачем:** Ловит fraud rings — группы связанных аккаунтов
- **Латентность:** <50ms (batch inference, кэширование embeddings)

### Модель 3: LLM Explainer (async)
- **Задача:** Генерация человеко-читаемого объяснения решения
- **Модель:** Claude API (или local Llama через Ollama для dev)
- **Input:** Transaction + features + scores → structured prompt
- **Output:** "Транзакция заблокирована: необычно высокая сумма (45K ₸ при среднем 5K ₸), новое устройство, геолокация отличается на 1200 км от обычной"
- **Не блокирует scoring** — запускается async после решения

### A/B тестирование моделей
- **Механизм:** Hash(user_id) % 100 → control (0-49) / challenger (50-99)
- **Метрики:** fraud_catch_rate, false_positive_rate, latency_p99
- **Хранение:** MLflow experiments + ClickHouse для ad-hoc анализа

---

## Поэтапный план реализации

### Фаза 1: Фундамент (Дни 1-3)
**Цель:** Рабочий пайплайн от генерации транзакций до скоринга

```
Задачи:
├── 1.1 Инициализация репозитория
│   ├── pyproject.toml (monorepo с workspace deps)
│   ├── Makefile (make up/down/test/lint)
│   ├── docker-compose.infra.yml (Kafka + Redis + ClickHouse + MinIO)
│   ├── docker-compose.yml (все сервисы)
│   ├── .github/workflows/ci.yml (lint + test)
│   └── pre-commit config (ruff, mypy, black)
│
├── 1.2 Transaction Simulator
│   ├── Pydantic модель транзакции (schemas.py)
│   ├── Kafka producer (aiokafka)
│   ├── Реалистичные паттерны: нормальные + 5 типов фрода
│   │   ├── Card testing (много мелких txn)
│   │   ├── Account takeover (новый device + большая сумма)
│   │   ├── Geo anomaly (транзакция из другой страны)
│   │   ├── Velocity abuse (>10 txn за минуту)
│   │   └── Merchant collusion (круговые переводы)
│   └── CLI: `python -m simulator --rate 100 --fraud-ratio 0.02`
│
├── 1.3 Scoring Service (MVP — rule-based)
│   ├── FastAPI app с POST /score
│   ├── Kafka consumer (raw_txn) → score → produce (scored)
│   ├── Пока простые правила (amount > 3*avg → suspicious)
│   ├── Health check, OpenAPI docs
│   └── Prometheus metrics (request_count, latency_histogram)
│
└── 1.4 E2E smoke test
    ├── docker compose up → simulator produces → scoring consumes → scored topic
    └── pytest: produce txn → assert scored message appears
```

**Результат:** `make up` поднимает Kafka + Simulator + Scoring. Транзакции текут, скоринг работает.

---

### Фаза 2: Feature Store + ML (Дни 4-7)
**Цель:** Реальные ML модели вместо правил

```
Задачи:
├── 2.1 Feature Store — Online (Redis)
│   ├── Redis writer: Kafka consumer обновляет фичи при каждой транзакции
│   ├── Sliding window counters (txn_count_1h, sum_5m, etc.)
│   ├── Feature retrieval API (sync, <5ms)
│   └── TTL политики (30 дней для user profiles)
│
├── 2.2 Feature Store — Offline (ClickHouse)
│   ├── ClickHouse DDL (transactions, user_profiles, merchant_profiles)
│   ├── Kafka → ClickHouse sink (archiver service)
│   ├── Materialized views для агрегатов
│   └── SQL-based feature engineering
│
├── 2.3 Генерация обучающего датасета
│   ├── scripts/generate_dataset.py
│   ├── 100K+ транзакций с labels (is_fraud)
│   ├── Реалистичное распределение (98% legit, 2% fraud)
│   ├── 5 типов фрода с разными паттернами
│   └── Сохранение в Parquet → MinIO
│
├── 2.4 XGBoost модель
│   ├── ml_pipeline/train_xgboost.py
│   ├── Feature engineering (30+ фич)
│   ├── Optuna hyperparameter tuning
│   ├── MLflow experiment tracking
│   ├── Evaluation: AUC-ROC, PR-AUC, confusion matrix
│   ├── Model export → MLflow Model Registry
│   └── Scoring service загружает модель из registry
│
├── 2.5 Обновление Scoring Service
│   ├── XGBoost inference (<10ms)
│   ├── Feature fetch из Redis (<5ms)
│   ├── Decision logic: score > 0.8 → BLOCK, 0.5-0.8 → REVIEW, < 0.5 → ALLOW
│   └── Produce scored txn в Kafka
│
└── 2.6 Тесты
    ├── Unit: feature engineering, scoring logic
    ├── Integration: Redis ↔ Scoring, Kafka flow
    └── ML: model quality assertions (AUC > 0.90)
```

**Результат:** Работающий ML scoring с Feature Store. Модель обучена, метрики трекаются в MLflow.

---

### Фаза 3: GNN + LLM Explainer (Дни 8-10)
**Цель:** Продвинутые модели для графового анализа и объяснений

```
Задачи:
├── 3.1 GNN модель (PyTorch Geometric)
│   ├── Построение графа: users ↔ merchants через transactions
│   ├── Node features из ClickHouse (user_profiles, merchant_profiles)
│   ├── Edge features: amount, frequency, time_pattern
│   ├── GraphSAGE архитектура (2 layers, 128 hidden dim)
│   ├── Training: fraud node classification
│   ├── Inference: batch (каждые 5 мин), кэширование embeddings в Redis
│   └── MLflow tracking
│
├── 3.2 LLM Explainer
│   ├── Structured prompt: txn data + features + scores → explanation
│   ├── Claude API (prod) / Ollama Llama (dev)
│   ├── Async — не блокирует основной scoring
│   ├── Кэширование похожих объяснений
│   └── Fallback: template-based explanation если LLM недоступен
│
├── 3.3 Ensemble
│   ├── Weighted average: 0.6*xgboost + 0.3*gnn + 0.1*rules
│   ├── Автоматическая калибровка весов (на validation set)
│   └── A/B testing framework (hash-based split)
│
└── 3.4 Тесты
    ├── GNN: node classification accuracy > 0.85
    ├── LLM: explanation quality (automated eval)
    └── Ensemble: AUC improvement over single model
```

**Результат:** 3 модели работают в ensemble. LLM объясняет решения на человеческом языке.

---

### Фаза 4: Streaming Pipeline (Дни 11-13)
**Цель:** Полноценная потоковая обработка с Flink/Spark

```
Задачи:
├── 4.1 Spark Structured Streaming (или Flink)
│   ├── Kafka source → enrichment → feature computation → Kafka sink
│   ├── Windowed aggregations (tumbling + sliding windows)
│   │   ├── 1-minute: txn_count, txn_sum
│   │   ├── 5-minute: unique_merchants, unique_devices
│   │   ├── 1-hour: velocity, amount_std
│   │   └── 24-hour: daily_pattern_deviation
│   ├── IP enrichment (GeoIP database — MaxMind GeoLite2)
│   ├── Device fingerprint matching
│   └── Output: enriched transactions → Redis + scoring
│
├── 4.2 Data Lake Archiver
│   ├── Kafka → MinIO (S3-compatible)
│   ├── Parquet формат, partitioned by date
│   ├── Delta Lake for ACID (updates, time travel)
│   └── Retention policy: raw 90 дней, aggregated бессрочно
│
├── 4.3 dbt Transforms
│   ├── Staging: stg_transactions, stg_fraud_labels
│   ├── Intermediate: int_daily_aggregates, int_user_profiles
│   ├── Marts: fraud_analytics, model_performance
│   ├── dbt tests: not_null, unique, accepted_values
│   └── dbt docs generate
│
└── 4.4 Airflow DAGs
    ├── daily_retrain: trigger model retraining on new data
    ├── feature_backfill: recompute offline features
    ├── dbt_run: run dbt models
    └── data_quality_check: Great Expectations validation
```

**Результат:** End-to-end streaming pipeline. Данные обогащаются, агрегируются, архивируются. dbt трансформирует аналитику.

---

### Фаза 5: Dashboard + Alerting (Дни 14-16)
**Цель:** Real-time UI и система оповещений

```
Задачи:
├── 5.1 Dashboard (Next.js 14 + TypeScript)
│   ├── Layout: sidebar navigation, dark theme
│   ├── Главная страница:
│   │   ├── KPI карточки (total txn, fraud rate, avg latency, blocked amount)
│   │   ├── Live transaction feed (WebSocket)
│   │   ├── Fraud score distribution chart (recharts)
│   │   └── Fraud map (leaflet — geo visualization)
│   ├── Transactions page:
│   │   ├── Таблица с фильтрами (date, score, decision, category)
│   │   ├── Детальная карточка транзакции (фичи + скоры + explanation)
│   │   └── Pagination + search
│   ├── Analytics page:
│   │   ├── Fraud rate over time (line chart)
│   │   ├── Top fraud categories (bar chart)
│   │   ├── Amount distribution (histogram)
│   │   └── Hourly heatmap
│   ├── Models page:
│   │   ├── Model versions + metrics (from MLflow API)
│   │   ├── A/B test results (control vs challenger)
│   │   └── Feature importance (XGBoost SHAP values)
│   └── WebSocket server (FastAPI → browser push)
│
├── 5.2 Alert Service
│   ├── Kafka consumer (alerts topic)
│   ├── Rules: score > 0.9 → instant alert, > 0.7 → batch (5 min)
│   ├── Telegram bot notification
│   ├── Webhook (для интеграции с PagerDuty, Slack)
│   └── Alert deduplication (same user, 10 min window)
│
└── 5.3 API Gateway
    ├── FastAPI: REST API для dashboard
    ├── GET /api/transactions (paginated, filterable)
    ├── GET /api/stats (real-time KPIs)
    ├── GET /api/models (model info from MLflow)
    ├── WS /api/ws/feed (live transaction stream)
    └── Auth: API key middleware
```

**Результат:** Красивый live dashboard. Транзакции текут в реальном времени, алерты приходят в Telegram.

---

### Фаза 6: Infrastructure + DevOps (Дни 17-20)
**Цель:** Production-ready инфраструктура

```
Задачи:
├── 6.1 Kubernetes manifests
│   ├── Deployments для всех 7 сервисов
│   ├── HPA (Horizontal Pod Autoscaler) для scoring service
│   │   └── Scale on: CPU > 70% OR p99_latency > 100ms
│   ├── PodDisruptionBudget (min 2 replicas for scoring)
│   ├── Resource limits/requests для каждого pod
│   ├── ConfigMaps + Secrets
│   ├── Ingress (nginx) для dashboard + API
│   └── NetworkPolicies (scoring ↔ redis only, etc.)
│
├── 6.2 Terraform
│   ├── Модули: k8s_cluster, kafka, redis, s3, clickhouse, monitoring
│   ├── Environments: dev (minikube) + prod
│   ├── State backend: S3 + DynamoDB lock
│   └── Outputs: cluster endpoint, service URLs
│
├── 6.3 ArgoCD (GitOps)
│   ├── Application: fraud-detection-platform
│   ├── Sync policy: automated (dev), manual (prod)
│   ├── Health checks для каждого сервиса
│   └── Rollback strategy
│
├── 6.4 CI/CD (GitHub Actions)
│   ├── ci.yml:
│   │   ├── Lint (ruff) + Type check (mypy) + Test (pytest)
│   │   ├── Build Docker images
│   │   ├── Push to GHCR (GitHub Container Registry)
│   │   └── Run on: push to main, PR
│   ├── cd.yml:
│   │   ├── Trigger: ci.yml success on main
│   │   ├── Update K8s manifests with new image tag
│   │   ├── ArgoCD sync
│   │   └── Smoke test after deploy
│   └── model-training.yml:
│       ├── Trigger: weekly schedule OR manual
│       ├── Run training pipeline
│       ├── Evaluate + compare with current model
│       └── Auto-promote if metrics improve
│
└── 6.5 Observability
    ├── Prometheus:
    │   ├── Scrape configs для всех сервисов
    │   ├── Custom metrics: fraud_score_histogram, scoring_latency, kafka_lag
    │   └── Alert rules: high_latency, kafka_consumer_lag, model_drift
    ├── Grafana dashboards (JSON provisioning):
    │   ├── System Health: CPU, memory, pod count, error rate
    │   ├── Fraud Overview: fraud rate, blocked amount, score distribution
    │   ├── Model Performance: AUC over time, A/B comparison
    │   └── Kafka: throughput, consumer lag, partition balance
    └── Jaeger:
        ├── Distributed tracing: txn → kafka → enrichment → scoring → alert
        ├── OpenTelemetry instrumentation
        └── Trace sampling: 10% normal, 100% errors
```

**Результат:** Полная инфраструктура. GitOps деплой, автоскейлинг, мониторинг, трейсинг.

---

### Фаза 7: Load Testing + Polish (Дни 21-23)
**Цель:** Доказать что система держит нагрузку. Полировка.

```
Задачи:
├── 7.1 Load Testing (k6)
│   ├── Сценарий 1: 10K RPS на /score endpoint
│   ├── Сценарий 2: E2E — 5K txn/sec через весь pipeline
│   ├── Сценарий 3: Spike test — 0 → 50K RPS за 30 секунд
│   ├── Метрики: p50, p95, p99 latency, error rate, throughput
│   └── Отчёт с графиками в docs/
│
├── 7.2 README.md
│   ├── Architecture diagram (Mermaid)
│   ├── Quick start: `make up` → working system in 2 minutes
│   ├── Tech stack table
│   ├── Screenshots dashboard
│   ├── Performance benchmarks
│   ├── API documentation link
│   └── Architecture Decision Records
│
├── 7.3 Documentation
│   ├── docs/architecture.md — C4 diagrams
│   ├── docs/data_model.md — schemas + ER diagram
│   ├── docs/ml_models.md — model cards (architecture, metrics, limitations)
│   ├── docs/runbook.md — operational procedures
│   └── docs/adr/ — Architecture Decision Records
│
└── 7.4 Final polish
    ├── Все Dockerfile multi-stage (builder + runtime)
    ├── .dockerignore, .gitignore
    ├── Type hints везде (mypy strict)
    ├── Docstrings для public API
    └── GitHub: topics, description, social preview image
```

**Результат:** Production-ready проект. Документация, тесты, бенчмарки — всё на месте.

---

## Полный стек проекта

```
Languages:        Python, TypeScript, SQL, HCL (Terraform)
ML/AI:            XGBoost, PyTorch, PyTorch Geometric (GNN), Optuna, SHAP
LLM:              Claude API, Ollama, structured prompts
Data:             Kafka, Spark Structured Streaming, ClickHouse, Redis, MinIO (S3), Delta Lake
Transforms:       dbt, Airflow, Great Expectations
Backend:          FastAPI, aiokafka, Pydantic, SQLAlchemy
Frontend:         Next.js 14, TypeScript, Tailwind, Recharts, Leaflet, WebSocket
MLOps:            MLflow, model registry, A/B testing, automated retraining
DevOps:           Docker, Kubernetes, Terraform, ArgoCD, GitHub Actions
Monitoring:       Prometheus, Grafana, Jaeger (OpenTelemetry)
Testing:          pytest, k6, Great Expectations
Architecture:     Microservices, event-driven, feature store pattern, GitOps
```

## Ежедневный чеклист для агента

При старте работы:
1. Прочитай этот файл целиком
2. Посмотри текущее состояние: `git log --oneline -10` + `ls` по сервисам
3. Определи текущую фазу по наличию реализованных компонентов
4. Работай по задачам текущей фазы последовательно
5. Каждый сервис = отдельный коммит
6. После каждого сервиса — `make up` для проверки что всё поднимается
7. Тесты пишутся сразу, не откладываются

## Принципы реализации

- **Docker-first:** Каждый сервис работает в контейнере. `docker compose up` поднимает всё.
- **Config через env:** Все настройки через `.env`, никаких хардкодов.
- **Async everywhere:** asyncio, aiokafka, httpx — всё асинхронное.
- **Type hints:** mypy strict mode, Pydantic для валидации.
- **Graceful degradation:** Если Redis недоступен — fallback на default фичи. Если LLM timeout — template explanation.
- **Idempotency:** Повторная обработка той же транзакции не создаёт дубликатов.
- **12-factor app:** Stateless сервисы, конфиг через env, логи в stdout.
