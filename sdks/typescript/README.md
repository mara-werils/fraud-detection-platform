# @fraud-detection/sdk

TypeScript SDK for the [Fraud Detection Platform](../../README.md).

Provides a fully-typed, production-ready HTTP client for scoring transactions, managing fraud cases, submitting analyst feedback, and monitoring model drift — with zero runtime dependencies (native `fetch` only).

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
  - [Scoring](#scoring)
  - [Batch Scoring](#batch-scoring)
  - [Transaction Search](#transaction-search)
  - [Case Management](#case-management)
  - [Analyst Feedback](#analyst-feedback)
  - [Drift Detection](#drift-detection)
  - [Model Info](#model-info)
  - [Health Checks](#health-checks)
- [Error Handling](#error-handling)
- [Retry Behaviour](#retry-behaviour)
- [Interceptors](#interceptors)
- [TypeScript Types](#typescript-types)

---

## Requirements

| Requirement | Minimum version |
|-------------|----------------|
| Node.js     | 18.0.0         |
| TypeScript  | 5.0.0          |

Node 18+ ships native `fetch`, so the SDK requires **no extra HTTP library**.

---

## Installation

```bash
# npm
npm install @fraud-detection/sdk

# yarn
yarn add @fraud-detection/sdk

# pnpm
pnpm add @fraud-detection/sdk
```

---

## Quick Start

```ts
import { FraudClient } from "@fraud-detection/sdk";

const client = new FraudClient({
  baseUrl: "https://fraud.example.com",
  apiKey: process.env.FRAUD_API_KEY,
});

// Score a single transaction
const result = await client.score({
  user_id: "user-001",
  amount: 4500,
  transaction_type: "purchase",
  merchant_id: "merch-007",
  ip_address: "203.0.113.42",
});

console.log(`Score: ${result.fraud_score.toFixed(4)} | Decision: ${result.decision}`);
// Score: 0.8723 | Decision: BLOCK
```

---

## Configuration

```ts
import { FraudClient, ClientConfig } from "@fraud-detection/sdk";

const config: ClientConfig = {
  /** Base URL of the fraud detection service */
  baseUrl: "https://fraud.example.com",   // default: http://localhost:8000

  /** API key sent as X-API-Key header */
  apiKey: "fdp_live_abc123",

  /** Request timeout in milliseconds */
  timeout: 15_000,                         // default: 30_000

  /** Max retry attempts on 429/5xx responses */
  retries: 3,                              // default: 3

  /** Base delay (ms) for exponential backoff */
  retryDelay: 300,                         // default: 300

  /** Optional request interceptor (e.g. add tracing headers) */
  onRequest: (url, init) => {
    (init.headers as Record<string, string>)["X-Request-Id"] = crypto.randomUUID();
    return init;
  },

  /** Optional response interceptor (e.g. metrics) */
  onResponse: (response, url) => {
    console.debug(`[${response.status}] ${url}`);
  },
};

const client = new FraudClient(config);
```

---

## API Reference

### Scoring

#### `client.score(transaction)`

Score a single transaction. Returns a `ScoredTransaction` enriched with a `decision` field derived by the SDK (`ALLOW` / `REVIEW` / `BLOCK`).

```ts
const result = await client.score({
  user_id: "user-001",
  amount: 450.00,
  currency: "GBP",
  transaction_type: "purchase",
  merchant_id: "merch-007",
  merchant_category: "5411",        // Grocery stores
  ip_address: "203.0.113.42",
  device_id: "device-fingerprint-xyz",
  latitude: 51.5074,
  longitude: -0.1278,
  metadata: { channel: "mobile_app" },
});

console.log(result.fraud_score);    // 0.12
console.log(result.decision);       // "ALLOW"
console.log(result.is_flagged);     // false
console.log(result.model_version);  // "xgb-v1.2.0"
```

**Decision thresholds** (derived by the SDK):

| Score range | Decision |
|-------------|----------|
| >= 0.8      | `BLOCK`  |
| 0.5 – 0.8   | `REVIEW` |
| < 0.5       | `ALLOW`  |

---

### Batch Scoring

#### `client.batchScore(transactions)`

Score up to 1,000 transactions in a single HTTP call. Individual failures don't block other transactions.

```ts
const response = await client.batchScore([
  { user_id: "u1", amount: 99, transaction_type: "purchase" },
  { user_id: "u2", amount: 50_000, transaction_type: "transfer" },
]);

console.log(`Processed: ${response.total}`);
console.log(`Failed:    ${response.failed}`);
console.log(`Latency:   ${response.latency_ms}ms`);

for (const tx of response.results) {
  console.log(`${tx.transaction_id}: ${tx.decision} (${tx.fraud_score.toFixed(4)})`);
}

// Handle partial failures
for (const err of response.errors) {
  console.error(`[${err.index}] ${err.transaction_id}: ${err.error}`);
}
```

#### `client.batchScoreStats(transactions)`

Score a batch and return only aggregate statistics — no per-transaction results.

```ts
const stats = await client.batchScoreStats(transactions);

console.log(`Blocked: ${stats.blocked}/${stats.total}`);
console.log(`Avg score: ${stats.avg_score.toFixed(4)}`);
```

---

### Transaction Search

#### `client.searchTransactions(filters?)`

Search scored transactions with filtering, pagination, and time range support.

```ts
const page = await client.searchTransactions({
  min_score: 0.8,
  is_flagged: true,
  start_time: "2025-01-01T00:00:00Z",
  end_time: "2025-01-31T23:59:59Z",
  limit: 20,
  offset: 0,
});

console.log(`${page.total} transactions found`);
console.log(`Showing ${page.transactions.length}, has_more: ${page.has_more}`);
```

**Available filters:**

| Filter       | Type      | Description                              |
|--------------|-----------|------------------------------------------|
| `user_id`    | `string`  | Filter by user UUID                      |
| `min_score`  | `number`  | Minimum fraud score (0–1)               |
| `max_score`  | `number`  | Maximum fraud score (0–1)               |
| `is_flagged` | `boolean` | Only flagged/unflagged transactions      |
| `decision`   | `string`  | `ALLOW`, `REVIEW`, or `BLOCK`           |
| `start_time` | `string`  | ISO 8601 start of time range            |
| `end_time`   | `string`  | ISO 8601 end of time range              |
| `limit`      | `number`  | Results per page (1–500, default 50)    |
| `offset`     | `number`  | Pagination offset                        |

#### `client.getTransaction(transactionId)`

Retrieve a single scored transaction by ID.

```ts
const tx = await client.getTransaction("txn-abc123");
```

#### `client.getTransactionStats()`

Get aggregate statistics across all stored transactions.

```ts
const stats = await client.getTransactionStats();
console.log(`Total: ${stats.total}, Flagged: ${stats.flagged}`);
```

---

### Case Management

#### `client.createCase(input)`

Create a fraud investigation case for a suspicious transaction.

```ts
const fraudCase = await client.createCase({
  transaction_id: "txn-abc123",
  priority: "high",
  assigned_to: "analyst-jane",
  notes: "Unusual cross-border transfer; customer unreachable.",
  tags: ["cross-border", "high-value"],
});

console.log(fraudCase.case_id);   // "case-uuid-..."
console.log(fraudCase.status);    // "open"
```

#### `client.getCases(filters?)`

List cases with optional filtering and pagination.

```ts
const { cases, total } = await client.getCases({
  status: "open",
  priority: "high",
  assigned_to: "analyst-jane",
  limit: 50,
  offset: 0,
});
```

#### `client.getCase(caseId)`

Get a single case by ID, including its full audit event log and notes.

```ts
const fraudCase = await client.getCase("case-uuid-...");
console.log(fraudCase.events);   // Array of CaseEvent
console.log(fraudCase.notes);    // Array of CaseNote
```

#### `client.updateCase(caseId, update)`

Update status, assignment, or add a note to a case.

```ts
// Escalate
await client.updateCase("case-uuid-...", {
  status: "escalated",
  actor: "analyst-jane",
  notes: "Escalating due to high customer impact.",
});

// Resolve
await client.updateCase("case-uuid-...", {
  status: "resolved_fraud",
  actor: "senior-analyst-bob",
});
```

#### `client.getCaseStats()`

```ts
const stats = await client.getCaseStats();
console.log(stats.by_status);      // { open: 12, investigating: 4, ... }
console.log(stats.avg_resolution_hours);
```

---

### Analyst Feedback

#### `client.submitFeedback(feedback)`

Submit an analyst verdict on a scored transaction. Feedback feeds the model accuracy dashboard and training data pipeline. Automatically closes any open case for the same transaction.

```ts
await client.submitFeedback({
  transaction_id: "txn-abc123",
  is_fraud: true,
  analyst: "jane.doe",
  notes: "Confirmed fraud via customer call-back at 14:32 UTC.",
});
```

#### `client.listFeedback(filters?)`

```ts
const { entries, total } = await client.listFeedback({
  analyst: "jane.doe",
  is_fraud: true,
  limit: 100,
});
```

#### `client.getFeedbackStats()`

```ts
const stats = await client.getFeedbackStats();
console.log(`False positive rate: ${(stats.false_positive_rate * 100).toFixed(1)}%`);
console.log(`Agreement rate:      ${(stats.agreement_rate * 100).toFixed(1)}%`);
```

#### `client.exportTrainingData()`

Export all labelled feedback entries for use in model retraining pipelines.

```ts
const { total, data } = await client.exportTrainingData();
console.log(`Exporting ${total} labelled samples`);
```

---

### Drift Detection

#### `client.getDriftReport()`

Get the latest feature drift report. Reports Population Stability Index (PSI) and Kolmogorov–Smirnov statistics for every monitored feature.

```ts
const report = await client.getDriftReport();

if (report.status === "insufficient_data") {
  console.log("Not enough data yet.");
} else if (report.overall_drift_detected) {
  console.warn("Drift detected — model retraining recommended.");
  for (const [feature, stats] of Object.entries(report.features)) {
    if (stats.has_drifted) {
      console.warn(`  ${feature}: PSI=${stats.psi.toFixed(3)}`);
    }
  }
}
```

#### `client.getDriftHistory(limit?)`

```ts
const history = await client.getDriftHistory(10);
for (const report of history.reports) {
  console.log(`${report.checked_at}: ${report.status}`);
}
```

---

### Model Info

#### `client.getModelInfo()`

```ts
const info = await client.getModelInfo();
console.log(`Model:    ${info.model_version}`);
console.log(`Type:     ${info.model_type}`);
console.log(`Features: ${info.feature_count}`);
console.log(`Thresholds: BLOCK >= ${info.thresholds.block}, REVIEW >= ${info.thresholds.review}`);
```

---

### Health Checks

#### `client.healthCheck()`

Readiness check — confirms critical dependencies (Redis, scorer) are healthy.

```ts
const health = await client.healthCheck();
if (health.status !== "healthy") {
  console.error("Service not ready:", health.components);
}
```

#### `client.healthStatus()`

Full dependency check including Redis, Kafka, PostgreSQL, and the scorer.

```ts
const status = await client.healthStatus();
console.log(`Overall: ${status.status}`);
for (const component of status.components) {
  console.log(`  ${component.name}: ${component.status} (${component.latency_ms}ms)`);
}
```

---

## Error Handling

The SDK exports three error classes:

| Class                | When thrown                                              |
|----------------------|----------------------------------------------------------|
| `FraudAPIError`      | API returned a 4xx or 5xx response                      |
| `FraudTimeoutError`  | Request exceeded the configured `timeout`               |
| `FraudNetworkError`  | Network-level failure (DNS, connection refused, etc.)   |

```ts
import {
  FraudClient,
  FraudAPIError,
  FraudTimeoutError,
  FraudNetworkError,
} from "@fraud-detection/sdk";

const client = new FraudClient({ baseUrl: "http://localhost:8000" });

try {
  const result = await client.score({
    user_id: "user-001",
    amount: 100,
    transaction_type: "purchase",
  });
  console.log(result.decision);
} catch (err) {
  if (err instanceof FraudAPIError) {
    console.error(`API error ${err.statusCode}: ${err.message}`);
    console.error("Response body:", err.body);
    console.error("URL:", err.url);

    if (err.statusCode === 401) {
      console.error("Check your API key.");
    } else if (err.statusCode === 422) {
      console.error("Validation error — check request payload.");
    }
  } else if (err instanceof FraudTimeoutError) {
    console.error(`Timed out: ${err.url}`);
  } else if (err instanceof FraudNetworkError) {
    console.error(`Network failure: ${err.message}`);
  } else {
    throw err; // Re-throw unexpected errors
  }
}
```

---

## Retry Behaviour

The client automatically retries on the following HTTP status codes:

`408` (Request Timeout) · `429` (Too Many Requests) · `500` · `502` · `503` · `504`

Retry strategy: **exponential backoff with full jitter**.

```
delay = random(0, min(cap, baseDelay × 2^attempt))
```

- Default `retries`: **3**
- Default `retryDelay`: **300ms**
- Maximum backoff cap: **30 seconds**

4xx errors (except 408 and 429) are **not** retried. `FraudTimeoutError` is also not retried — if a timeout occurs the error is thrown immediately.

```ts
const client = new FraudClient({
  retries: 5,
  retryDelay: 500,   // 500ms base → max ~30s per attempt
});
```

---

## Interceptors

Use interceptors to add tracing, logging, or custom authentication headers without subclassing the client.

### Request interceptor

```ts
const client = new FraudClient({
  apiKey: "fdp_...",
  onRequest: (url, init) => {
    // Add OpenTelemetry trace context
    (init.headers as Record<string, string>)["traceparent"] = getTraceparent();
    return init;
  },
});
```

### Response interceptor

```ts
const client = new FraudClient({
  onResponse: async (response, url) => {
    // Record latency in your metrics system
    const latency = response.headers.get("X-Response-Time");
    metrics.histogram("fraud_api_latency", Number(latency));

    // Log all 4xx responses without suppressing them
    if (response.status >= 400) {
      logger.warn({ url, status: response.status }, "API error");
    }
  },
});
```

---

## TypeScript Types

All types are exported from the package root. A full list:

```ts
import type {
  // Config
  ClientConfig,
  RequestInterceptor,
  ResponseInterceptor,

  // Enums
  TransactionType,
  Decision,
  CaseStatus,
  CasePriority,
  AlertSeverity,
  WebhookFormat,

  // Transactions
  TransactionInput,
  Transaction,
  ScoredTransaction,
  TransactionFilters,
  TransactionSearchResponse,
  TransactionStats,

  // Feature vector
  FeatureVector,

  // Batch scoring
  BatchScoreRequest,
  BatchScoreResponse,
  BatchScoreStats,
  BatchScoreError,

  // Cases
  CaseNote,
  CaseEvent,
  Case,
  CreateCaseInput,
  UpdateCaseInput,
  CaseFilters,
  CaseStats,

  // Feedback
  FeedbackRequest,
  FeedbackEntry,
  FeedbackFilters,
  FeedbackStats,

  // Drift
  FeatureDriftStats,
  DriftReport,
  DriftHistoryResponse,

  // Model
  ModelInfo,

  // Webhooks
  WebhookConfig,

  // Health
  ComponentStatus,
  HealthStatus,

  // Generic
  PaginatedResponse,
  APIErrorBody,
} from "@fraud-detection/sdk";
```

---

## Development

```bash
# Install dev dependencies
npm install

# Type-check without building
npm run typecheck

# Compile to dist/
npm run build

# Run tests
npm test
```

The SDK ships both ESM (`dist/index.js`) and CommonJS (`dist/index.cjs`) builds so it works in any Node.js project regardless of `"type"` setting in the consumer's `package.json`.
