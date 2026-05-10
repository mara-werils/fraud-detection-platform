-- Mart: fraud analytics summary joining transactions with labels.

{{ config(materialized='table') }}

SELECT
    t.user_id,
    t.transaction_date,
    t.daily_txn_count,
    t.daily_txn_sum,
    t.daily_txn_avg,
    t.daily_txn_max,
    t.daily_txn_std,
    t.daily_unique_merchants,
    t.daily_unique_devices,
    t.daily_avg_fraud_score,
    t.daily_max_fraud_score,
    t.daily_flagged_count,
    t.daily_high_risk_count,

    -- Confirmed fraud from labels
    countIf(fl.is_fraud = 1)                                    AS confirmed_fraud_count,
    countIf(fl.is_fraud = 0)                                    AS confirmed_legit_count,

    -- Precision proxy: of flagged transactions, how many were actually fraud
    CASE
        WHEN t.daily_flagged_count > 0
        THEN countIf(fl.is_fraud = 1) / t.daily_flagged_count
        ELSE 0
    END                                                         AS daily_precision,

    -- User profile context
    u.lifetime_txn_count,
    u.lifetime_avg_fraud_score,
    u.fraud_flag_rate,
    u.account_age_days,
    u.txn_per_day

FROM {{ ref('int_daily_aggregates') }} AS t

LEFT JOIN {{ ref('stg_transactions') }} AS st
    ON t.user_id = st.user_id AND t.transaction_date = st.transaction_date

LEFT JOIN {{ ref('stg_fraud_labels') }} AS fl
    ON st.transaction_id = fl.transaction_id

LEFT JOIN {{ ref('int_user_profiles') }} AS u
    ON t.user_id = u.user_id

GROUP BY
    t.user_id,
    t.transaction_date,
    t.daily_txn_count,
    t.daily_txn_sum,
    t.daily_txn_avg,
    t.daily_txn_max,
    t.daily_txn_std,
    t.daily_unique_merchants,
    t.daily_unique_devices,
    t.daily_avg_fraud_score,
    t.daily_max_fraud_score,
    t.daily_flagged_count,
    t.daily_high_risk_count,
    u.lifetime_txn_count,
    u.lifetime_avg_fraud_score,
    u.fraud_flag_rate,
    u.account_age_days,
    u.txn_per_day
