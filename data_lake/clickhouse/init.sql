-- ============================================================================
-- ClickHouse DDL for Fraud Detection Platform
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Transactions table
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions
(
    txn_id             UUID,
    user_id            UUID,
    merchant_id        UUID,
    amount             Decimal(18, 2),
    currency           LowCardinality(String),
    category           LowCardinality(String),
    channel            LowCardinality(String),
    ip_address         String,
    device_fingerprint String,
    geo_lat            Float64,
    geo_lon            Float64,
    fraud_score        Float32,
    xgboost_score      Float32,
    gnn_score          Float32,
    decision           LowCardinality(String),  -- ALLOW / REVIEW / BLOCK
    explanation        String,
    is_fraud           UInt8,                    -- label (0/1)
    features           Map(String, Float64),
    created_at         DateTime64(3),
    scored_at          DateTime64(3)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (user_id, created_at);


-- ----------------------------------------------------------------------------
-- 2. User profiles table
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_profiles
(
    user_id            UUID,
    total_txn_count    UInt64,
    total_txn_amount   Decimal(18, 2),
    avg_txn_amount     Decimal(18, 2),
    fraud_count        UInt32,
    fraud_rate         Float32,
    first_txn_date     Date,
    last_txn_date      Date,
    unique_merchants   UInt32,
    unique_devices     UInt32,
    primary_geo        String,
    risk_tier          LowCardinality(String),  -- LOW / MEDIUM / HIGH
    updated_at         DateTime64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY user_id;


-- ----------------------------------------------------------------------------
-- 3. Merchant profiles table
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS merchant_profiles
(
    merchant_id        UUID,
    name               String,
    category           LowCardinality(String),
    total_txn          UInt64,
    fraud_count        UInt32,
    fraud_rate         Float32,
    avg_txn_amount     Decimal(18, 2),
    risk_tier          LowCardinality(String),  -- LOW / MEDIUM / HIGH
    updated_at         DateTime64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY merchant_id;


-- ----------------------------------------------------------------------------
-- 4. Alerts table
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts
(
    alert_id           UUID,
    txn_id             UUID,
    user_id            UUID,
    fraud_score        Float32,
    decision           LowCardinality(String),  -- ALLOW / REVIEW / BLOCK
    severity           LowCardinality(String),  -- low / medium / high / critical
    channel            LowCardinality(String),  -- telegram / webhook / email
    sent_at            DateTime64(3),
    acknowledged       UInt8                    -- 0/1
)
ENGINE = MergeTree()
ORDER BY (sent_at);


-- ----------------------------------------------------------------------------
-- 5. Materialized view: Daily fraud stats
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_fraud_stats
(
    day                Date,
    total_txn          UInt64,
    total_amount       Decimal(38, 2),
    fraud_txn          UInt64,
    fraud_amount       Decimal(38, 2),
    avg_fraud_score    Float64,
    block_count        UInt64,
    review_count       UInt64,
    allow_count        UInt64
)
ENGINE = SummingMergeTree()
ORDER BY day;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_fraud_stats
TO daily_fraud_stats
AS
SELECT
    toDate(created_at)                          AS day,
    count()                                     AS total_txn,
    sum(amount)                                 AS total_amount,
    countIf(is_fraud = 1)                       AS fraud_txn,
    sumIf(amount, is_fraud = 1)                 AS fraud_amount,
    avg(fraud_score)                            AS avg_fraud_score,
    countIf(decision = 'BLOCK')                 AS block_count,
    countIf(decision = 'REVIEW')                AS review_count,
    countIf(decision = 'ALLOW')                 AS allow_count
FROM transactions
GROUP BY day;


-- ----------------------------------------------------------------------------
-- 6. Materialized view: Hourly transaction volume
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hourly_volume
(
    hour               DateTime,
    total_txn          UInt64,
    total_amount       Decimal(38, 2),
    unique_users       UInt64,
    avg_score          Float64,
    p95_score          Float32,
    fraud_count        UInt64
)
ENGINE = SummingMergeTree()
ORDER BY hour;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_hourly_volume
TO hourly_volume
AS
SELECT
    toStartOfHour(created_at)                   AS hour,
    count()                                     AS total_txn,
    sum(amount)                                 AS total_amount,
    uniq(user_id)                               AS unique_users,
    avg(fraud_score)                            AS avg_score,
    quantile(0.95)(fraud_score)                 AS p95_score,
    countIf(is_fraud = 1)                       AS fraud_count
FROM transactions
GROUP BY hour;
