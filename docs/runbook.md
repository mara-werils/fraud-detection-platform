# Operational Runbook

## Service Health Checks

### Endpoints

| Service | Health Endpoint | Expected Response | Port |
|---------|----------------|-------------------|------|
| Scoring Service | `GET /health` | `{"status": "healthy", "models_loaded": true}` | 8000 |
| Simulator | `GET /health` | `{"status": "healthy", "producing": true}` | 8001 |
| Alert Service | `GET /health` | `{"status": "healthy", "consumers_active": true}` | 8002 |
| Feature Store | `GET /health` | `{"status": "healthy", "redis": "connected", "clickhouse": "connected"}` | 8003 |
| Dashboard | `GET /api/health` | `{"status": "ok"}` | 3000 |
| Kafka | Broker metadata | N/A (use `kafka-broker-api-versions.sh`) | 9092 |
| Redis | `PING` | `PONG` | 6379 |
| ClickHouse | `GET /ping` | `Ok.\n` | 8123 |

### Quick Health Check Script

```bash
# Check all services
for svc in scoring:8000 simulator:8001 alert-service:8002 feature-store:8003; do
  name=$(echo $svc | cut -d: -f1)
  port=$(echo $svc | cut -d: -f2)
  status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health)
  echo "$name: $status"
done

# Check infrastructure
redis-cli ping
curl -s http://localhost:8123/ping
docker compose exec kafka kafka-broker-api-versions.sh --bootstrap-server localhost:9092 | head -1
```

### Kubernetes Health Checks

```bash
# Pod status
kubectl get pods -n fraud-detection -o wide

# Service endpoints
kubectl get endpoints -n fraud-detection

# Recent events
kubectl get events -n fraud-detection --sort-by='.lastTimestamp' | tail -20

# HPA status
kubectl get hpa -n fraud-detection
```

---

## Common Issues and Troubleshooting

### 1. High Scoring Latency (p99 > 100ms)

**Symptoms**: Dashboard shows latency spikes, Grafana alerts fire.

**Diagnosis**:
```bash
# Check scoring service metrics
curl -s http://localhost:8000/metrics | grep scoring_latency

# Check Redis latency
redis-cli --latency -h localhost

# Check pod resource usage
kubectl top pods -n fraud-detection -l app=scoring
```

**Resolution**:
1. Check if Redis is responding slowly (> 5 ms) — may need more memory or connection pooling
2. Check if the XGBoost model file is corrupted — re-download from MLflow
3. Check if GNN embeddings are being recomputed instead of cached
4. Scale up scoring replicas: `kubectl scale deployment scoring --replicas=5 -n fraud-detection`
5. Check for CPU throttling in pod resource limits

### 2. Kafka Consumer Lag

**Symptoms**: Transactions are delayed, dashboard shows stale data.

**Diagnosis**:
```bash
# Check consumer group lag
docker compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group scoring-group

# Check partition assignment
docker compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group scoring-group --members
```

**Resolution**:
1. If lag is growing steadily → scale consumers (increase replicas)
2. If lag is in a single partition → check for a poison message (skip or DLQ)
3. If all partitions lagging → check consumer processing time, may need to optimize scoring
4. Temporary: increase `max.poll.records` to process more messages per batch
5. If consumer is restarting → check logs for OOM or connection errors

### 3. Redis Connection Failures

**Symptoms**: Feature retrieval falls back to defaults, scoring accuracy drops.

**Diagnosis**:
```bash
# Check Redis connectivity
redis-cli -h localhost ping

# Check memory usage
redis-cli info memory | grep used_memory_human

# Check connection count
redis-cli info clients | grep connected_clients

# Check slow log
redis-cli slowlog get 10
```

**Resolution**:
1. If OOM → increase `maxmemory` or review TTL policies
2. If max connections reached → increase `maxclients`, review connection pooling in services
3. If network timeout → check network policies, DNS resolution
4. Restart Redis if unresponsive: `kubectl rollout restart statefulset redis -n data`

### 4. Model Loading Failure

**Symptoms**: Scoring service starts but returns errors or uses fallback rules.

**Diagnosis**:
```bash
# Check scoring service logs
kubectl logs -n fraud-detection -l app=scoring --tail=100 | grep -i "model\|error"

# Check MLflow connectivity
curl -s http://mlflow:5000/api/2.0/mlflow/registered-models/list

# Check model artifacts
kubectl exec -n fraud-detection deployment/scoring -- ls -la /app/models/
```

**Resolution**:
1. Verify MLflow is accessible and the model version exists
2. Re-download model: restart scoring pod to trigger model fetch
3. Check disk space in the scoring pod
4. If model is corrupted → roll back to previous version in MLflow registry
5. Fallback: scoring service automatically uses rule-based scoring when models fail to load

### 5. Dashboard WebSocket Disconnections

**Symptoms**: Live feed stops updating, users report stale data.

**Diagnosis**:
```bash
# Check WebSocket connections
curl -s http://localhost:3000/api/health

# Check Kafka consumer for dashboard
docker compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group dashboard-group
```

**Resolution**:
1. Check if the backend WebSocket server is running
2. Verify Kafka consumer for the dashboard is not lagging
3. Check ingress/load balancer WebSocket timeout settings (should be > 60s)
4. Client-side: the dashboard auto-reconnects — check browser console for errors

---

## Scaling Procedures

### Horizontal Scaling — Scoring Service

```bash
# Manual scale
kubectl scale deployment scoring --replicas=10 -n fraud-detection

# Update HPA limits
kubectl patch hpa scoring-hpa -n fraud-detection \
  --type merge -p '{"spec":{"maxReplicas":20}}'

# Verify scaling
kubectl get hpa scoring-hpa -n fraud-detection -w
```

### Kafka Partition Scaling

> Partitions can only be increased, never decreased.

```bash
# Increase partitions for raw_txn topic
docker compose exec kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --alter --topic raw_txn \
  --partitions 24

# Verify
docker compose exec kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --describe --topic raw_txn
```

After increasing partitions, restart consumers to rebalance.

### ClickHouse Scaling

```bash
# Add a shard (update Terraform)
cd infra/terraform
terraform plan -var-file=environments/prod.tfvars
terraform apply -var-file=environments/prod.tfvars

# Rebalance data
clickhouse-client --query "SYSTEM RESHARD TABLE fraud_detection.transactions"
```

---

## Model Deployment Process

### Standard Deployment (Automated)

1. Airflow `daily_retrain` DAG triggers model training
2. `ml_pipeline/train_xgboost.py` trains on latest data
3. `ml_pipeline/evaluate.py` computes metrics on holdout set
4. If metrics improve → model registered in MLflow as `Staging`
5. A/B test runs for 7 days (challenger group)
6. If A/B criteria met → model promoted to `Production` in MLflow
7. Scoring service detects new production model version
8. Rolling restart picks up new model (zero downtime)

### Manual / Emergency Deployment

```bash
# 1. List available model versions
mlflow models list --name fraud-xgboost

# 2. Promote a specific version
mlflow models transition-stage \
  --name fraud-xgboost \
  --version 15 \
  --stage Production

# 3. Force scoring service to reload
kubectl rollout restart deployment scoring -n fraud-detection

# 4. Verify
curl -s http://localhost:8000/model/info | jq .
```

### Rollback

```bash
# 1. Identify the previous version
mlflow models list --name fraud-xgboost | head -5

# 2. Transition current to Archived
mlflow models transition-stage \
  --name fraud-xgboost --version 16 --stage Archived

# 3. Promote previous version back to Production
mlflow models transition-stage \
  --name fraud-xgboost --version 15 --stage Production

# 4. Restart scoring
kubectl rollout restart deployment scoring -n fraud-detection
```

---

## Incident Response: High Fraud Rate

### Alert Trigger

Prometheus alert: `fraud_rate_5m > 5%` (normal baseline: ~2%)

### Response Steps

1. **Acknowledge** the alert in PagerDuty / Grafana
2. **Assess scope**:
   ```bash
   # Check fraud rate over last hour
   clickhouse-client --query "
     SELECT
       toStartOfMinute(created_at) AS minute,
       countIf(decision = 'BLOCK') AS blocked,
       count() AS total,
       blocked / total AS fraud_rate
     FROM fraud_detection.transactions
     WHERE created_at > now() - INTERVAL 1 HOUR
     GROUP BY minute
     ORDER BY minute DESC
     LIMIT 20
   "
   ```
3. **Check for attack patterns**:
   ```bash
   # Top users by blocked transactions
   clickhouse-client --query "
     SELECT user_id, count() AS blocked_count, sum(amount) AS total_amount
     FROM fraud_detection.transactions
     WHERE decision = 'BLOCK' AND created_at > now() - INTERVAL 1 HOUR
     GROUP BY user_id
     ORDER BY blocked_count DESC
     LIMIT 20
   "
   ```
4. **If targeted attack**: Consider temporarily lowering block threshold from 0.80 to 0.70
5. **If model drift**: Trigger emergency model retraining with recent data
6. **If false positive spike**: Check if feature store is returning stale data
7. **Post-incident**: Update fraud patterns in simulator, retrain model, write post-mortem

---

## Kafka Consumer Lag Remediation

### Monitoring

```bash
# Grafana dashboard: "Kafka Overview" → Consumer Lag panel
# Or manual check:
docker compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --all-groups
```

### Thresholds

| Lag Level | Threshold | Action |
|-----------|-----------|--------|
| Normal | < 1,000 messages | No action |
| Warning | 1,000–10,000 | Monitor, prepare to scale |
| Critical | > 10,000 | Scale consumers immediately |
| Emergency | > 100,000 | Scale + consider skipping old messages |

### Remediation Steps

1. **Scale consumers**:
   ```bash
   kubectl scale deployment scoring --replicas=10 -n fraud-detection
   ```

2. **Increase batch size** (temporary):
   ```bash
   kubectl set env deployment/scoring \
     KAFKA_MAX_POLL_RECORDS=1000 \
     -n fraud-detection
   ```

3. **Skip stale messages** (emergency only):
   ```bash
   # Reset consumer offset to latest
   docker compose exec kafka kafka-consumer-groups.sh \
     --bootstrap-server localhost:9092 \
     --group scoring-group \
     --topic raw_txn \
     --reset-offsets --to-latest \
     --execute
   ```

4. **Post-recovery**: Verify lag returns to normal, review why lag accumulated

---

## Database Maintenance

### ClickHouse

#### Partition Management

```bash
# List partitions
clickhouse-client --query "
  SELECT partition, name, rows, bytes_on_disk
  FROM system.parts
  WHERE table = 'transactions' AND active
  ORDER BY partition
"

# Drop old partition (if retention policy requires)
clickhouse-client --query "
  ALTER TABLE fraud_detection.transactions
  DROP PARTITION '202401'
"

# Optimize table (merge parts)
clickhouse-client --query "
  OPTIMIZE TABLE fraud_detection.transactions FINAL
"
```

#### Backup

```bash
# Create backup
clickhouse-client --query "
  BACKUP TABLE fraud_detection.transactions
  TO S3('s3://backups/clickhouse/transactions/', 'access_key', 'secret_key')
"
```

### Redis

#### Memory Management

```bash
# Check memory usage per key pattern
redis-cli --scan --pattern "user:*:txn_count_*" | wc -l
redis-cli info memory

# Force eviction of expired keys
redis-cli debug sleep 0  # triggers lazy expiration

# Manual cleanup if needed
redis-cli --scan --pattern "user:*:gnn_embedding" | xargs redis-cli del
```

#### Persistence

```bash
# Trigger RDB snapshot
redis-cli bgsave

# Check last save status
redis-cli lastsave
```

### MinIO / S3

```bash
# List Delta Lake partitions
mc ls minio/fraud-detection/transactions/

# Check storage usage
mc du minio/fraud-detection/

# Clean up old partitions (> 90 days for raw data)
mc rm --recursive --older-than 90d minio/fraud-detection/raw/
```
