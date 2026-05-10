-- Mart: model performance tracking over time.

{{ config(materialized='table') }}

WITH scored AS (
    SELECT
        transaction_date,
        model_version,
        fraud_score,
        is_flagged,
        transaction_id
    FROM {{ ref('stg_transactions') }}
),

labeled AS (
    SELECT
        transaction_id,
        is_fraud
    FROM {{ ref('stg_fraud_labels') }}
),

joined AS (
    SELECT
        s.transaction_date,
        s.model_version,
        s.fraud_score,
        s.is_flagged,
        COALESCE(l.is_fraud, 0)                                 AS is_fraud
    FROM scored AS s
    LEFT JOIN labeled AS l ON s.transaction_id = l.transaction_id
)

SELECT
    transaction_date,
    model_version,
    count()                                                     AS total_predictions,

    -- Confusion matrix components
    countIf(is_flagged = 1 AND is_fraud = 1)                    AS true_positives,
    countIf(is_flagged = 1 AND is_fraud = 0)                    AS false_positives,
    countIf(is_flagged = 0 AND is_fraud = 1)                    AS false_negatives,
    countIf(is_flagged = 0 AND is_fraud = 0)                    AS true_negatives,

    -- Rates
    countIf(is_flagged = 1 AND is_fraud = 1) /
        greatest(countIf(is_flagged = 1), 1)                    AS precision,
    countIf(is_flagged = 1 AND is_fraud = 1) /
        greatest(countIf(is_fraud = 1), 1)                      AS recall,

    -- Score distribution
    avg(fraud_score)                                            AS avg_score,
    quantile(0.5)(fraud_score)                                  AS median_score,
    quantile(0.95)(fraud_score)                                 AS p95_score,
    quantile(0.99)(fraud_score)                                 AS p99_score,

    -- Volume
    countIf(is_flagged = 1)                                     AS flagged_count,
    countIf(is_fraud = 1)                                       AS actual_fraud_count,
    countIf(is_flagged = 1) / count()                           AS flag_rate

FROM joined
GROUP BY transaction_date, model_version
ORDER BY transaction_date DESC, model_version
