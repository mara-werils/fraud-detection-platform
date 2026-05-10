# ADR-001: Kafka over RabbitMQ for Event Streaming

## Status

Accepted

## Date

2026-05-01

## Context

The fraud detection platform requires a message broker to handle the flow of transaction events between services. The system must process thousands of transactions per second with low latency while supporting event replay for model retraining and audit purposes.

Key requirements:
- High throughput: 10,000+ events/second sustained
- Low latency: < 5 ms broker transit time
- Durability: events must survive broker restarts
- Replay capability: reprocess historical events for model retraining
- Consumer groups: multiple independent consumers per topic
- Ordering: per-user ordering guarantees

The two primary candidates were Apache Kafka and RabbitMQ.

## Decision

We chose **Apache Kafka** as the event streaming platform.

## Rationale

### Kafka Advantages for This Use Case

| Criterion | Kafka | RabbitMQ |
|-----------|-------|----------|
| **Throughput** | 100K+ msg/sec per partition | 20K–50K msg/sec per queue |
| **Message retention** | Configurable (days/weeks), log-based | Consumed messages are deleted |
| **Replay** | Native — seek to any offset | Not supported without plugins |
| **Consumer groups** | Native — independent consumer groups | Requires exchange/queue bindings |
| **Ordering** | Per-partition ordering guaranteed | Per-queue ordering only |
| **Backpressure** | Consumers read at their own pace | Requires prefetch tuning |
| **Ecosystem** | Kafka Connect, Kafka Streams, ksqlDB | Limited streaming ecosystem |
| **Durability** | Replicated log, ISR-based | Mirrored queues, less battle-tested at scale |

### Why Replay Matters

The fraud detection pipeline requires event replay for:
1. **Model retraining**: Replay historical transactions through updated feature engineering
2. **Backfill**: When new features are added, reprocess past events to compute feature values
3. **Debugging**: Replay specific time windows to diagnose scoring anomalies
4. **Disaster recovery**: Rebuild downstream state from the event log

RabbitMQ deletes messages after consumption, making replay impossible without a separate archival layer.

### Why Not RabbitMQ

RabbitMQ excels at task queuing and request/reply patterns, but the fraud detection platform is fundamentally a **streaming** system, not a task queue. Transactions flow continuously, multiple consumers process the same events independently, and historical data must remain accessible.

## Consequences

### Positive
- Native support for event replay enables model retraining from historical data
- Consumer groups allow independent scaling of scoring, archiving, and alerting
- Per-partition ordering guarantees correct per-user event sequencing
- High throughput supports growth to 50K+ transactions/second

### Negative
- Higher operational complexity than RabbitMQ (ZooKeeper/KRaft, partition management)
- Larger resource footprint (disk I/O for log segments)
- Consumer offset management requires careful handling
- Team needs Kafka-specific expertise

### Mitigations
- Use KRaft mode (no ZooKeeper dependency) in Kafka 3.x+
- Containerized deployment with managed configuration
- Use `aiokafka` library with built-in offset management
- Monitoring via Prometheus JMX exporter and Grafana dashboards
