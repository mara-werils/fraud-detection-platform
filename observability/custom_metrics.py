"""
Custom Prometheus metrics for the fraud detection platform.

All services import from this module to ensure consistent metric naming
and labeling across the platform.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# ---------------------------------------------------------------------------
# Scoring Service Metrics
# ---------------------------------------------------------------------------

scoring_request_duration = Histogram(
    "scoring_request_duration_seconds",
    "Time spent processing a scoring request end-to-end",
    labelnames=["model_name", "model_version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)

scoring_score = Histogram(
    "scoring_score",
    "Distribution of fraud scores produced by the scoring service",
    labelnames=["model_name"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

transactions_scored_total = Counter(
    "transactions_scored_total",
    "Total number of transactions scored",
    labelnames=["model_name", "model_version"],
)

transactions_flagged_total = Counter(
    "transactions_flagged_total",
    "Total number of transactions flagged by result type",
    labelnames=["result", "model_name"],
)

blocked_transaction_amount_total = Counter(
    "blocked_transaction_amount_total",
    "Total monetary amount of blocked transactions (USD)",
    labelnames=["model_name"],
)

scoring_batch_size = Histogram(
    "scoring_batch_size",
    "Number of transactions in each scoring batch",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)

# ---------------------------------------------------------------------------
# Model Performance Metrics
# ---------------------------------------------------------------------------

model_auc_roc = Gauge(
    "model_auc_roc",
    "Current AUC-ROC score of the model",
    labelnames=["model_name", "model_version"],
)

model_precision = Gauge(
    "model_precision",
    "Current precision of the model",
    labelnames=["model_name", "model_version"],
)

model_recall = Gauge(
    "model_recall",
    "Current recall of the model",
    labelnames=["model_name", "model_version"],
)

model_f1_score = Gauge(
    "model_f1_score",
    "Current F1 score of the model",
    labelnames=["model_name", "model_version"],
)

model_inference_duration = Histogram(
    "model_inference_duration_seconds",
    "Time spent on model inference only (excluding feature retrieval)",
    labelnames=["model_name", "model_version"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

model_info = Info(
    "model",
    "Information about the currently loaded model",
)

# ---------------------------------------------------------------------------
# Feature Store Metrics
# ---------------------------------------------------------------------------

feature_retrieval_duration = Histogram(
    "feature_retrieval_duration_seconds",
    "Time spent retrieving features from the feature store",
    labelnames=["store_type"],  # "online" or "offline"
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

feature_cache_hits_total = Counter(
    "feature_cache_hits_total",
    "Number of feature cache hits",
    labelnames=["feature_group"],
)

feature_cache_misses_total = Counter(
    "feature_cache_misses_total",
    "Number of feature cache misses",
    labelnames=["feature_group"],
)

feature_drift_score = Gauge(
    "feature_drift_score",
    "Population Stability Index (PSI) for feature drift detection",
    labelnames=["feature", "model_name"],
)

# ---------------------------------------------------------------------------
# Streaming / Kafka Metrics
# ---------------------------------------------------------------------------

kafka_messages_consumed_total = Counter(
    "kafka_messages_consumed_total",
    "Total number of Kafka messages consumed",
    labelnames=["topic", "consumer_group"],
)

kafka_messages_produced_total = Counter(
    "kafka_messages_produced_total",
    "Total number of Kafka messages produced",
    labelnames=["topic"],
)

kafka_consumer_lag = Gauge(
    "kafka_consumer_group_lag",
    "Kafka consumer group lag per partition",
    labelnames=["group", "topic", "partition"],
)

kafka_processing_duration = Histogram(
    "kafka_processing_duration_seconds",
    "Time spent processing a single Kafka message",
    labelnames=["topic", "consumer_group"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# ---------------------------------------------------------------------------
# Alert Service Metrics
# ---------------------------------------------------------------------------

alerts_sent_total = Counter(
    "alerts_sent_total",
    "Total number of alerts sent",
    labelnames=["channel", "severity"],  # channel: email, telegram, webhook
)

alerts_deduplicated_total = Counter(
    "alerts_deduplicated_total",
    "Total number of alerts suppressed by deduplication",
)

alert_processing_duration = Histogram(
    "alert_processing_duration_seconds",
    "Time spent processing and dispatching an alert",
    labelnames=["channel"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# ---------------------------------------------------------------------------
# Simulator Metrics
# ---------------------------------------------------------------------------

simulated_transactions_total = Counter(
    "simulated_transactions_total",
    "Total number of simulated transactions generated",
    labelnames=["transaction_type"],  # "legitimate" or "fraud"
)

simulation_rate = Gauge(
    "simulation_rate_per_second",
    "Current rate of simulated transactions per second",
)

# ---------------------------------------------------------------------------
# HTTP / General Service Metrics
# ---------------------------------------------------------------------------

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "endpoint", "status", "app"],
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    labelnames=["method", "endpoint", "app"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ---------------------------------------------------------------------------
# Data Lake / ClickHouse Metrics
# ---------------------------------------------------------------------------

clickhouse_query_duration = Histogram(
    "clickhouse_query_duration_seconds",
    "Duration of ClickHouse queries",
    labelnames=["query_type"],  # "insert", "select"
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

clickhouse_rows_inserted_total = Counter(
    "clickhouse_rows_inserted_total",
    "Total number of rows inserted into ClickHouse",
    labelnames=["table"],
)
