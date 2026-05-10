-- Test: ensure no null fraud scores exist in staged transactions.

SELECT
    transaction_id,
    fraud_score
FROM {{ ref('stg_transactions') }}
WHERE fraud_score IS NULL
