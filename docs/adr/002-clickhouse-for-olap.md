# ADR-002: ClickHouse for OLAP Storage

## Status

Accepted

## Date

2026-05-01

## Context

The platform needs an analytical database for:
- Storing scored transactions for offline analysis (billions of rows over time)
- Computing aggregate features for model training (user profiles, merchant statistics)
- Powering dashboard analytics (fraud rate trends, category breakdowns, model performance)
- Running ad-hoc queries for incident investigation

Requirements:
- Sub-second query performance on time-range aggregations
- Efficient storage for high-cardinality time-series data
- Support for materialized views and pre-computed aggregations
- Ingestion rate: 5,000+ rows/second sustained
- SQL interface for compatibility with dbt and analytics tools

Candidates evaluated: ClickHouse, PostgreSQL (with TimescaleDB), Apache Druid, and Amazon Redshift.

## Decision

We chose **ClickHouse** as the OLAP database.

## Rationale

### Comparison

| Criterion | ClickHouse | PostgreSQL + TimescaleDB | Druid | Redshift |
|-----------|-----------|------------------------|-------|----------|
| **Query speed (aggregations)** | Excellent (columnar, vectorized) | Good (hypertables) | Excellent | Good |
| **Ingestion rate** | 500K+ rows/sec | 50K rows/sec | 100K+ rows/sec | 10K rows/sec |
| **Compression** | 10–40x | 3–5x | 5–10x | 5–10x |
| **SQL support** | Full (with extensions) | Full (standard) | Limited | Full |
| **dbt support** | Yes (dbt-clickhouse) | Yes (native) | No | Yes |
| **Materialized views** | Native, real-time | Yes | Pre-aggregation | Yes (manual) |
| **Self-hosted** | Yes, lightweight | Yes | Complex (ZooKeeper) | No (AWS only) |
| **Cost** | Open source | Open source | Open source | $$$ |

### Why ClickHouse Wins

1. **Columnar storage**: Fraud analytics queries scan specific columns (amount, fraud_score, decision) across millions of rows. ClickHouse's columnar format reads only needed columns, delivering 10–100x speedup over row-based databases.

2. **Compression**: `LowCardinality(String)` for columns like `currency`, `category`, `decision` achieves 20–40x compression. This is critical for storing billions of transactions cost-effectively.

3. **MergeTree engine**: Partition by month (`toYYYYMM(created_at)`), order by `(user_id, created_at)`. This layout optimizes both time-range scans and user-specific queries.

4. **ReplacingMergeTree**: Perfect for `user_profiles` — upsert semantics without complex CDC pipelines. The `updated_at` column controls deduplication during background merges.

5. **Real-time materialized views**: Pre-compute hourly fraud rates, daily aggregates, and model performance metrics automatically as data arrives. No batch job needed.

6. **dbt compatibility**: The `dbt-clickhouse` adapter supports staging → intermediate → mart layer transforms with full testing and documentation.

### Why Not PostgreSQL

PostgreSQL (even with TimescaleDB) struggles at the scale and query patterns of this platform:
- Analytical queries on 100M+ rows are 10–50x slower than ClickHouse
- Compression is limited compared to columnar storage
- No native `LowCardinality` optimization for enum-like columns
- Ingestion rate is constrained by row-based MVCC overhead

PostgreSQL remains excellent for OLTP workloads, but the fraud platform's analytical queries are purely OLAP.

## Consequences

### Positive
- Sub-second aggregation queries on billions of rows
- Excellent compression reduces storage costs by 10–40x
- Materialized views provide real-time analytics without batch jobs
- Strong SQL support integrates with dbt and existing analytics tools
- Open source with active community and commercial support (ClickHouse Cloud)

### Negative
- No ACID transactions (eventual consistency on merges)
- Updates and deletes are expensive (mutations)
- Joins are less efficient than in PostgreSQL (denormalization preferred)
- Smaller talent pool compared to PostgreSQL
- Client library ecosystem is less mature

### Mitigations
- Use `ReplacingMergeTree` for upsert patterns (user/merchant profiles)
- Denormalize data at ingestion time to avoid joins
- Use materialized views instead of complex join queries
- Document ClickHouse-specific patterns in team runbook
