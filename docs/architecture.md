# Architecture

## System Overview

The Fraud Detection Platform is a real-time, event-driven microservices system that processes financial transactions, enriches them with features, scores them through an ML ensemble (XGBoost + GNN + LLM explainer), and delivers decisions in under 100 ms.

```mermaid
C4Context
  title System Context — Fraud Detection Platform

  Person(analyst, "Fraud Analyst", "Monitors transactions and reviews flagged cases")
  Person(ops, "Operations", "Manages platform health and model deployments")

  System(fdp, "Fraud Detection Platform", "Real-time ML-based transaction scoring")

  System_Ext(bank, "Core Banking", "Sends transaction events")
  System_Ext(telegram, "Telegram", "Alert notifications")
  System_Ext(mlflow, "MLflow Registry", "Model versioning and tracking")

  Rel(bank, fdp, "Sends transactions", "Kafka")
  Rel(fdp, telegram, "Sends fraud alerts", "HTTPS")
  Rel(fdp, mlflow, "Registers models", "HTTPS")
  Rel(analyst, fdp, "Reviews flagged transactions", "HTTPS / WebSocket")
  Rel(ops, fdp, "Monitors dashboards", "HTTPS")
```

## Component Architecture

```mermaid
graph TB
  subgraph Ingestion
    SIM[Transaction Simulator]
  end

  subgraph Messaging
    K1[Kafka: raw_txn]
    K2[Kafka: scored]
    K3[Kafka: alerts]
  end

  subgraph Processing
    STR[Streaming Pipeline<br/>Spark Structured Streaming]
    FS[Feature Store Writer]
    ARC[Data Lake Archiver]
  end

  subgraph Storage
    REDIS[(Redis<br/>Online Features)]
    CH[(ClickHouse<br/>Offline Features)]
    MINIO[(MinIO / S3<br/>Delta Lake)]
  end

  subgraph Scoring
    API[Scoring Service<br/>FastAPI]
    XGB[XGBoost Scorer]
    GNN[GNN Scorer<br/>GraphSAGE]
    LLM[LLM Explainer]
    ENS[Ensemble]
  end

  subgraph Presentation
    DASH[Dashboard<br/>Next.js]
    ALERT[Alert Service<br/>Telegram / Webhook]
  end

  subgraph Orchestration
    AIR[Airflow DAGs]
    DBT[dbt Transforms]
  end

  subgraph Observability
    PROM[Prometheus]
    GRAF[Grafana]
    JAEG[Jaeger]
  end

  SIM -->|produce| K1
  K1 --> STR
  K1 --> FS
  K1 --> ARC
  STR --> API
  FS --> REDIS
  ARC --> MINIO
  ARC --> CH
  API --> XGB
  API --> GNN
  API --> LLM
  XGB --> ENS
  GNN --> ENS
  ENS --> K2
  K2 --> DASH
  K2 --> ALERT
  ALERT --> K3
  REDIS -.->|feature lookup| API
  CH -.->|offline features| DBT
  AIR --> DBT
  API -.->|metrics| PROM
  PROM --> GRAF
  API -.->|traces| JAEG
```

## Component Descriptions

| # | Service | Technology | Responsibility |
|---|---------|-----------|----------------|
| 1 | **Transaction Simulator** | Python, aiokafka | Generates realistic transaction events with configurable fraud patterns (card testing, account takeover, geo anomaly, velocity abuse, merchant collusion). Produces to `raw_txn` Kafka topic. |
| 2 | **Streaming Pipeline** | Spark Structured Streaming | Consumes raw transactions, computes windowed aggregations (1m, 5m, 1h, 24h), enriches with IP geolocation and device fingerprinting. Outputs enriched events for scoring. |
| 3 | **Feature Store** | Redis (online) + ClickHouse (offline) | Online store provides sub-5ms feature retrieval for real-time scoring. Offline store supports batch feature engineering and model training. Feature registry tracks versions. |
| 4 | **Scoring Service** | FastAPI, XGBoost, PyTorch Geometric | Core service. Fetches features, runs ML ensemble (XGBoost + GNN), applies decision logic (ALLOW / REVIEW / BLOCK), produces scored events. Supports A/B testing. |
| 5 | **Alert Service** | Python, aiokafka | Consumes scored events above threshold, sends notifications via Telegram, webhook, and email. Supports deduplication and batching. |
| 6 | **Dashboard** | Next.js 14, TypeScript, Tailwind | Real-time fraud monitoring UI with live transaction feed (WebSocket), fraud map, score distribution charts, model comparison, and analytics. |
| 7 | **Data Lake** | MinIO (S3), Delta Lake, dbt | Archives transactions to Parquet/Delta format. dbt transforms raw data into staging, intermediate, and mart layers for analytics and retraining. |

## Data Flow

### Transaction Lifecycle

```mermaid
sequenceDiagram
  participant SIM as Simulator
  participant KR as Kafka (raw_txn)
  participant STR as Streaming
  participant FS as Feature Store
  participant SC as Scoring Service
  participant KS as Kafka (scored)
  participant AL as Alert Service
  participant DB as Dashboard

  SIM->>KR: Produce transaction event
  KR->>STR: Consume & enrich
  KR->>FS: Update online features (Redis)
  STR->>SC: Forward enriched transaction
  SC->>FS: Fetch features (<5ms)
  SC->>SC: XGBoost score (<10ms)
  SC->>SC: GNN score (<50ms)
  SC->>SC: Ensemble decision
  SC->>KS: Produce scored event
  SC-->>SC: LLM explain (async)
  KS->>DB: WebSocket push
  KS->>AL: Consume high-score events
  AL->>AL: Telegram / Webhook alert
```

### Latency Budget

| Stage | Target | Notes |
|-------|--------|-------|
| Kafka produce → consume | < 5 ms | Single-partition, local broker |
| Feature retrieval (Redis) | < 5 ms | Pipeline get, connection pool |
| XGBoost inference | < 10 ms | Pre-loaded model, numpy arrays |
| GNN inference | < 50 ms | Cached embeddings, batch updates |
| Ensemble + decision | < 2 ms | Weighted average |
| Kafka produce (scored) | < 5 ms | Async, non-blocking |
| **Total (p95)** | **< 100 ms** | |
| LLM explanation (async) | 500–2000 ms | Does not block scoring path |

## Technology Choices

| Category | Technology | Rationale |
|----------|-----------|-----------|
| Message Broker | Apache Kafka | High throughput, durable, replay capability. See [ADR-001](adr/001-kafka-over-rabbitmq.md). |
| Online Store | Redis 7 | Sub-millisecond reads, native data structures (sorted sets, hashes). |
| OLAP Database | ClickHouse | Column-oriented, fast aggregations on billions of rows. See [ADR-002](adr/002-clickhouse-for-olap.md). |
| Object Storage | MinIO (S3-compatible) | Delta Lake support, cost-effective long-term storage. |
| ML Framework | XGBoost + PyTorch Geometric | XGBoost for tabular speed; GNN for graph-based fraud ring detection. See [ADR-003](adr/003-gnn-for-graph-fraud.md). |
| API Framework | FastAPI | Async-native, auto-generated OpenAPI docs, Pydantic validation. |
| Frontend | Next.js 14 | React-based SSR, excellent TypeScript support, built-in routing. |
| Orchestration | Apache Airflow | Industry-standard DAG orchestration for batch pipelines. |
| Data Transforms | dbt | SQL-based transforms, testing, documentation, lineage. |
| Experiment Tracking | MLflow | Model registry, experiment comparison, artifact storage. |
| Container Orchestration | Kubernetes | Auto-scaling, self-healing, declarative infrastructure. |
| IaC | Terraform | Multi-cloud support, modular, state management. |
| GitOps | ArgoCD | Automated sync from Git to K8s, rollback support. |
| CI/CD | GitHub Actions | Native integration, matrix builds, reusable workflows. |
| Monitoring | Prometheus + Grafana | Pull-based metrics, rich dashboarding, alerting. |
| Tracing | Jaeger (OpenTelemetry) | Distributed trace visualization, root cause analysis. |

## Deployment Architecture

```mermaid
graph TB
  subgraph "CI/CD"
    GH[GitHub Actions] -->|build + push| GHCR[GHCR<br/>Container Registry]
    GH -->|update manifests| REPO[Git Repository]
  end

  subgraph "GitOps"
    REPO --> ARGO[ArgoCD]
    ARGO -->|sync| K8S
  end

  subgraph K8S["Kubernetes Cluster"]
    direction TB
    subgraph ns-fraud["namespace: fraud-detection"]
      SC_D[Scoring<br/>3 replicas<br/>HPA]
      STR_D[Streaming<br/>2 replicas]
      SIM_D[Simulator<br/>1 replica]
      ALERT_D[Alert Service<br/>2 replicas]
      DASH_D[Dashboard<br/>2 replicas]
    end
    subgraph ns-data["namespace: data"]
      KAFKA_D[Kafka<br/>3 brokers]
      REDIS_D[Redis<br/>Sentinel]
      CH_D[ClickHouse<br/>2 shards]
      MINIO_D[MinIO<br/>4 nodes]
    end
    subgraph ns-obs["namespace: observability"]
      PROM_D[Prometheus]
      GRAF_D[Grafana]
      JAEG_D[Jaeger]
    end
  end

  ING[Ingress<br/>nginx] --> DASH_D
  ING --> SC_D
```

## Scalability Considerations

- **Scoring Service**: Horizontal Pod Autoscaler scales from 3 to 20 replicas based on CPU utilization (> 70%) and p99 latency (> 100 ms). Stateless design enables linear scaling.
- **Kafka**: 3-broker cluster with configurable partitions per topic. Consumer groups enable parallel processing. Partition count can be increased without downtime.
- **Feature Store (Redis)**: Redis Sentinel for HA. Read replicas for scaling reads. Pipeline batching reduces round trips.
- **ClickHouse**: MergeTree engine with monthly partitions. Horizontal sharding for write-heavy workloads. Materialized views for pre-computed aggregations.
- **Data Lake**: MinIO with erasure coding for durability. Delta Lake enables ACID transactions and time travel on object storage.
- **Dashboard**: Server-side rendering with CDN caching. WebSocket connections load-balanced across replicas.
