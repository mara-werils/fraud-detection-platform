import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";
import { randomIntBetween } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";
import { uuidv4 } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";

// ---------------------------------------------------------------------------
// Custom metrics
// ---------------------------------------------------------------------------
const fraudScoreTrend = new Trend("fraud_score", true);
const decisionAllow   = new Counter("decision_allow");
const decisionReview  = new Counter("decision_review");
const decisionBlock   = new Counter("decision_block");
const scoringErrors   = new Rate("scoring_errors");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const BASE_URL = __ENV.SCORING_URL || "http://localhost:8000";

const CATEGORIES  = ["electronics", "groceries", "restaurants", "travel", "entertainment", "clothing", "healthcare", "utilities"];
const CHANNELS    = ["mobile_app", "web", "pos", "atm"];
const CURRENCIES  = ["KZT", "USD", "EUR", "RUB"];
const CARD_TYPES  = ["visa", "mastercard", "amex"];

// ---------------------------------------------------------------------------
// Load stages — ramp from 0 to 10 000 VUs
// ---------------------------------------------------------------------------
export const options = {
  stages: [
    { duration: "30s",  target: 100   },
    { duration: "1m",   target: 500   },
    { duration: "1m",   target: 1000  },
    { duration: "2m",   target: 5000  },
    { duration: "2m",   target: 10000 },
    { duration: "3m",   target: 10000 },  // sustain peak
    { duration: "1m",   target: 0     },  // ramp-down
  ],

  thresholds: {
    http_req_duration:  ["p(95)<100", "p(99)<200"],
    scoring_errors:     ["rate<0.01"],
    http_req_failed:    ["rate<0.01"],
  },

  tags: {
    testName: "scoring_load_test",
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
  const isFraudulent = Math.random() < 0.02;
  const amount = isFraudulent
    ? randomFloat(50000, 500000)
    : randomFloat(100, 30000);

  return {
    txn_id:             uuidv4(),
    user_id:            uuidv4(),
    merchant_id:        uuidv4(),
    amount:             amount,
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
// Main VU function
// ---------------------------------------------------------------------------
export default function () {
  const payload = JSON.stringify(generateTransaction());

  const params = {
    headers: {
      "Content-Type": "application/json",
    },
    tags: { endpoint: "score" },
  };

  const res = http.post(`${BASE_URL}/score`, payload, params);

  const success = check(res, {
    "status is 200":       (r) => r.status === 200,
    "response has score":  (r) => {
      try { return JSON.parse(r.body).fraud_score !== undefined; }
      catch { return false; }
    },
    "latency < 200ms":    (r) => r.timings.duration < 200,
  });

  if (!success) {
    scoringErrors.add(1);
  } else {
    scoringErrors.add(0);
    try {
      const body = JSON.parse(res.body);
      fraudScoreTrend.add(body.fraud_score);

      switch (body.decision) {
        case "ALLOW":  decisionAllow.add(1);  break;
        case "REVIEW": decisionReview.add(1); break;
        case "BLOCK":  decisionBlock.add(1);  break;
      }
    } catch (_) { /* ignore parse errors for metrics */ }
  }

  sleep(0.1);
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
export function handleSummary(data) {
  const summary = {
    timestamp:       new Date().toISOString(),
    vus_max:         data.metrics.vus_max ? data.metrics.vus_max.values.max : 0,
    requests_total:  data.metrics.http_reqs ? data.metrics.http_reqs.values.count : 0,
    rps:             data.metrics.http_reqs ? data.metrics.http_reqs.values.rate : 0,
    latency_p50:     data.metrics.http_req_duration ? data.metrics.http_req_duration.values["p(50)"] : 0,
    latency_p95:     data.metrics.http_req_duration ? data.metrics.http_req_duration.values["p(95)"] : 0,
    latency_p99:     data.metrics.http_req_duration ? data.metrics.http_req_duration.values["p(99)"] : 0,
    error_rate:      data.metrics.http_req_failed ? data.metrics.http_req_failed.values.rate : 0,
    fraud_score_avg: data.metrics.fraud_score ? data.metrics.fraud_score.values.avg : 0,
  };

  return {
    "load_tests/results/scoring_results.json": JSON.stringify(summary, null, 2),
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}

import { textSummary } from "https://jslib.k6.io/k6-summary/0.1.0/index.js";
