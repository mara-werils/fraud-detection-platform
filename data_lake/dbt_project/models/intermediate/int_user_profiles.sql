-- Intermediate model: lifetime user profile aggregates.

{{ config(materialized='table') }}

SELECT
    user_id,
    count()                                                     AS lifetime_txn_count,
    sum(amount)                                                 AS lifetime_txn_sum,
    avg(amount)                                                 AS lifetime_txn_avg,
    max(amount)                                                 AS lifetime_txn_max,
    stddevPop(amount)                                           AS lifetime_txn_std,
    uniqExact(merchant_id)                                      AS lifetime_unique_merchants,
    uniqExact(device_id)                                        AS lifetime_unique_devices,
    uniqExact(ip_address)                                       AS lifetime_unique_ips,
    avg(fraud_score)                                            AS lifetime_avg_fraud_score,
    max(fraud_score)                                            AS lifetime_max_fraud_score,
    countIf(is_flagged = 1)                                     AS lifetime_flagged_count,
    countIf(is_flagged = 1) / count()                           AS fraud_flag_rate,
    min(transaction_date)                                       AS first_seen_date,
    max(transaction_date)                                       AS last_seen_date,
    dateDiff('day', min(transaction_date), max(transaction_date)) AS account_age_days,
    count() / greatest(dateDiff('day', min(transaction_date), max(transaction_date)), 1) AS txn_per_day
FROM {{ ref('stg_transactions') }}
GROUP BY user_id
