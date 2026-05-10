-- Staging model: clean and type-cast raw scored transactions.

{{ config(materialized='view') }}

SELECT
    transaction_id,
    user_id,
    toDecimal64(amount, 2)                              AS amount,
    currency,
    transaction_type,
    merchant_id,
    merchant_category,
    ip_address,
    device_id,
    latitude,
    longitude,
    timestamp                                           AS transaction_ts,
    toFloat64(fraud_score)                              AS fraud_score,
    model_version,
    toFloat64(scoring_latency_ms)                       AS scoring_latency_ms,
    scored_at,
    is_flagged,
    flag_reason,
    toDate(timestamp)                                   AS transaction_date,
    toHour(timestamp)                                   AS transaction_hour,
    toDayOfWeek(timestamp)                              AS transaction_dow
FROM {{ source('fraud', 'scored_transactions') }}
WHERE transaction_id IS NOT NULL
  AND user_id IS NOT NULL
  AND amount >= 0
