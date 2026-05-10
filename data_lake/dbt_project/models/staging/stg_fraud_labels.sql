-- Staging model: fraud labels for supervised learning.

{{ config(materialized='view') }}

SELECT
    transaction_id,
    label                                               AS is_fraud,
    labeled_at,
    labeled_by,
    label_source,
    toDate(labeled_at)                                  AS label_date
FROM {{ source('fraud', 'fraud_labels') }}
WHERE transaction_id IS NOT NULL
  AND label IN (0, 1)
