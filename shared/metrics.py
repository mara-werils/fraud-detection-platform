"""Prometheus metrics helpers for the fraud detection platform."""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Histogram, Info, start_http_server

__all__ = [
    "MetricsRegistry",
    "create_metrics",
    "start_metrics_server",
    "track_latency",
]


class MetricsRegistry:
    """Central registry for Prometheus metrics."""

    def __init__(self, service_name: str) -> None:
        self._service_name = service_name
        self._prefix = service_name.replace("-", "_")

        # Service info
        self.service_info: Info = Info(
            f"{self._prefix}_service",
            "Service information",
        )

        # Transaction counters
        self.transactions_received: Counter = Counter(
            f"{self._prefix}_transactions_received_total",
            "Total number of transactions received",
            ["transaction_type"],
        )
        self.transactions_scored: Counter = Counter(
            f"{self._prefix}_transactions_scored_total",
            "Total number of transactions scored",
            ["model_version"],
        )
        self.transactions_flagged: Counter = Counter(
            f"{self._prefix}_transactions_flagged_total",
            "Total number of transactions flagged as fraudulent",
            ["severity"],
        )
        self.transaction_errors: Counter = Counter(
            f"{self._prefix}_transaction_errors_total",
            "Total number of transaction processing errors",
            ["error_type"],
        )

        # Latency histograms
        self.scoring_latency: Histogram = Histogram(
            f"{self._prefix}_scoring_latency_seconds",
            "Time taken to score a transaction",
            ["model_version"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )
        self.feature_computation_latency: Histogram = Histogram(
            f"{self._prefix}_feature_computation_latency_seconds",
            "Time taken to compute feature vectors",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        self.kafka_publish_latency: Histogram = Histogram(
            f"{self._prefix}_kafka_publish_latency_seconds",
            "Time taken to publish messages to Kafka",
            ["topic"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
        )
        self.end_to_end_latency: Histogram = Histogram(
            f"{self._prefix}_end_to_end_latency_seconds",
            "End-to-end latency from transaction ingestion to scoring",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # Kafka consumer metrics
        self.kafka_messages_consumed: Counter = Counter(
            f"{self._prefix}_kafka_messages_consumed_total",
            "Total Kafka messages consumed",
            ["topic"],
        )
        self.kafka_consumer_lag: Histogram = Histogram(
            f"{self._prefix}_kafka_consumer_lag",
            "Kafka consumer lag in messages",
            ["topic", "partition"],
            buckets=(0, 10, 50, 100, 500, 1000, 5000, 10000),
        )

        # Alert metrics
        self.alerts_generated: Counter = Counter(
            f"{self._prefix}_alerts_generated_total",
            "Total alerts generated",
            ["severity"],
        )

    def set_service_info(self, version: str, **kwargs: str) -> None:
        """Set service metadata info.

        Args:
            version: Service version string.
            **kwargs: Additional key-value pairs for service info.
        """
        self.service_info.info({"version": version, "service": self._service_name, **kwargs})


def create_metrics(service_name: str) -> MetricsRegistry:
    """Create a metrics registry for a service.

    Args:
        service_name: Name of the service.

    Returns:
        A configured MetricsRegistry instance.
    """
    return MetricsRegistry(service_name)


def start_metrics_server(port: int = 8080) -> None:
    """Start a Prometheus metrics HTTP server.

    Args:
        port: Port to expose metrics on.
    """
    start_http_server(port)


def track_latency(histogram: Histogram, labels: dict[str, str] | None = None) -> Any:
    """Context manager to track operation latency.

    Args:
        histogram: The Histogram metric to record to.
        labels: Optional labels for the histogram.

    Returns:
        A context manager that records elapsed time.
    """
    if labels:
        return histogram.labels(**labels).time()
    return histogram.time()
