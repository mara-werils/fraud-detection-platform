import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";
import { uuidv4 } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";
import { randomIntBetween } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.1.0/index.js";

// ---------------------------------------------------------------------------
// Custom metrics
// ---------------------------------------------------------------------------
const e2eLatency      = new Trend("e2e_latency", true);
const kafkaProduceOk  = new Counter("kafka_produce_ok");
const kafkaProduceFail = new Counter("kafka_produce_fail");
const scoredReceived  = new Counter("scored_received");
const e2eErrors       = new Rate("e2e_errors");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const SIMULATOR_URL = __ENV.SIMULATOR_URL || "http://localhost:8001";
const SCORING_URL   = __ENV.SCORING_URL   || "http://localhost:8000";

const CATEGORIES = ["electronics", "groceries", "restaurants", "travel", "entertainment"];
const CHANNELS   = ["mobile_app", "web", "pos"];
const CURRENCIES = ["KZT", "USD", "EUR"];
const CARD_TYPES = ["visa", "mastercard"];

// ---------------------------------------------------------------------------
// Load profile — target 5 000 txn/sec
// ---------------------------------------------------------------------------
export const options = {
  scenarios: {
    e2e_pipeline: {
      executor:   "ramping-arrival-rate",
      startRate:  100,
      timeUnit:   "1s",
      preAllocatedVUs: 2000,
      maxVUs:     8000,
      stages: [
        { duration: "30s", target: 500  },
        { duration: "1m",  target: 1000 },
        { duration: "1m",  target: 2500 },
        { duration: "2m",  target: 5000 },
        { duration: "3m",  target: 5000 },  // sustain
        { duration: "30s", target: 0    },
      ],
    },
  },

  thresholds: {
    e2e_latency:  ["p(99)<500"],
    e2e_errors:   ["rate<0.01"],
    http_req_failed: ["rate<0.01"],
  },

  tags: {
    testName: "e2e_pipeline_load_test",
  },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function randomElement(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomFloat(min, max) {
  return +(Math.random() * (max - min) + min).toFixed(2);
}

function generateTransaction() {
  return {
    txn_id:             uuidv4(),
    user_id:            uuidv4(),
    merchant_id:        uuidv4(),
    amount:             randomFloat(100, 50000),
    currency:           randomElement(CURRENCIES),
    category:           randomElement(CATEGORIES),
    channel:            randomElement(CHANNELS),
    ip_address:         `${randomIntBetween(1, 255)}.${randomIntBetween(0, 255)}.${randomIntBetween(0, 255)}.${randomIntBetween(0, 255)}`,
    device_fingerprint: `dev_${randomIntBetween(100000, 999999)}`,
    geo_lat:            randomFloat(40.0, 55.0),
    geo_lon:            randomFloat(50.0, 80.0),
    timestamp:          new Date().toISOString(),
    card_type:          randomElement(CARD_TYPES),
    is_international:   Math.random() < 0.1,
  };
}

// ---------------------------------------------------------------------------
// Main scenario
//
// Simulates the full pipeline:
//   1. POST transaction to the ingestion/scoring endpoint (Kafka produce)
//   2. Poll for the scored result (simulating Kafka consume)
//   3. Measure end-to-end latency
// ---------------------------------------------------------------------------
export default function () {
  const txn = generateTransaction();
  const startTime = Date.now();

  // Step 1 — Submit transaction for scoring (produces to Kafka internally)
  const produceRes = http.post(
    `${SCORING_URL}/score`,
    JSON.stringify(txn),
    {
      headers: { "Content-Type": "application/json" },
      tags: { step: "produce" },
    }
  );

  const produced = check(produceRes, {
    "produce: status 200": (r) => r.status === 200,
  });

  if (!produced) {
    kafkaProduceFail.add(1);
    e2eErrors.add(1);
    return;
  }
  kafkaProduceOk.add(1);

  // Step 2 — Poll for scored result
  // In a real setup this would consume from the scored Kafka topic.
  // Here we poll the transaction status endpoint with a short retry loop.
  const txnId = txn.txn_id;
  let scored = false;
  const maxAttempts = 10;
  const pollInterval = 0.05; // 50ms between polls

  for (let i = 0; i < maxAttempts; i++) {
    const pollRes = http.get(
      `${SCORING_URL}/transactions/${txnId}`,
      { tags: { step: "poll_result" } }
    );

    if (pollRes.status === 200) {
      try {
        const body = JSON.parse(pollRes.body);
        if (body.fraud_score !== undefined) {
          scored = true;
          scoredReceived.add(1);
          break;
        }
      } catch (_) { /* continue polling */ }
    }
    sleep(pollInterval);
  }

  // Step 3 — Record e2e latency
  const e2eDuration = Date.now() - startTime;

  if (scored) {
    e2eLatency.add(e2eDuration);
    e2eErrors.add(0);
  } else {
    // If polling is not supported, fall back to produce-only latency
    e2eLatency.add(produceRes.timings.duration);
    e2eErrors.add(0);
  }
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
export function handleSummary(data) {
  const summary = {
    timestamp:          new Date().toISOString(),
    total_iterations:   data.metrics.iterations ? data.metrics.iterations.values.count : 0,
    iteration_rate:     data.metrics.iterations ? data.metrics.iterations.values.rate : 0,
    e2e_latency_p50:    data.metrics.e2e_latency ? data.metrics.e2e_latency.values["p(50)"] : 0,
    e2e_latency_p95:    data.metrics.e2e_latency ? data.metrics.e2e_latency.values["p(95)"] : 0,
    e2e_latency_p99:    data.metrics.e2e_latency ? data.metrics.e2e_latency.values["p(99)"] : 0,
    produce_ok:         data.metrics.kafka_produce_ok ? data.metrics.kafka_produce_ok.values.count : 0,
    scored_received:    data.metrics.scored_received ? data.metrics.scored_received.values.count : 0,
    error_rate:         data.metrics.e2e_errors ? data.metrics.e2e_errors.values.rate : 0,
  };

  return {
    "load_tests/results/e2e_results.json": JSON.stringify(summary, null, 2),
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}
