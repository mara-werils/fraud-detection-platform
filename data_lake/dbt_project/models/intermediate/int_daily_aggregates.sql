-- Intermediate model: daily transaction aggregates per user.

{{ config(materialized='table') }}

SELECT
    user_id,
    transaction_date,
    count()                                                     AS daily_txn_count,
    sum(amount)                                                 AS daily_txn_sum,
    avg(amount)                                                 AS daily_txn_avg,
    max(amount)                                                 AS daily_txn_max,
    min(amount)                                                 AS daily_txn_min,
    stddevPop(amount)                                           AS daily_txn_std,
    uniqExact(merchant_id)                                      AS daily_unique_merchants,
    uniqExact(device_id)                                        AS daily_unique_devices,
    uniqExact(ip_address)                                       AS daily_unique_ips,
    avg(fraud_score)                                            AS daily_avg_fraud_score,
    max(fraud_score)                                            AS daily_max_fraud_score,
    countIf(is_flagged = 1)                                     AS daily_flagged_count,
    countIf(fraud_score >= 0.75)                                AS daily_high_risk_count,
    min(transaction_ts)                                         AS first_txn_ts,
    max(transaction_ts)                                         AS last_txn_ts
FROM {{ ref('stg_transactions') }}
GROUP BY user_id, transaction_date
