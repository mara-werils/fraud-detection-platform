/**
 * Smoke test — quick sanity check for CI/CD pipeline.
 * Runs 10 VUs for 30 seconds to verify core endpoints respond correctly.
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const API_KEY = __ENV.API_KEY || "fdp_dev_key_change_me_in_production";

const errorRate = new Rate("errors");
const scoreLatency = new Trend("score_latency", true);

export const options = {
  vus: 10,
  duration: "30s",
  thresholds: {
    http_req_failed: ["rate<0.01"],          // < 1% errors
    http_req_duration: ["p(95)<500"],        // 95% under 500ms
    score_latency: ["p(99)<200"],            // scoring p99 < 200ms
  },
};

function randomTransaction() {
  return JSON.stringify({
    transaction_id: `txn_smoke_${Math.random().toString(36).slice(2)}`,
    user_id: `user_${Math.floor(Math.random() * 1000)}`,
    amount: parseFloat((Math.random() * 500).toFixed(2)),
    currency: "USD",
    merchant_id: `merch_${Math.floor(Math.random() * 100)}`,
    timestamp: new Date().toISOString(),
  });
}

export default function () {
  const headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
  };

  // Health check
  const health = http.get(`${BASE_URL}/health`);
  check(health, { "health OK": (r) => r.status === 200 });
  errorRate.add(health.status !== 200);

  // Score a transaction
  const scoreRes = http.post(
    `${BASE_URL}/api/v1/score`,
    randomTransaction(),
    { headers }
  );
  check(scoreRes, {
    "score 200": (r) => r.status === 200,
    "has fraud_score": (r) => JSON.parse(r.body).fraud_score !== undefined,
    "has decision": (r) => ["BLOCK", "REVIEW", "ALLOW"].includes(JSON.parse(r.body).decision),
  });
  scoreLatency.add(scoreRes.timings.duration);
  errorRate.add(scoreRes.status !== 200);

  sleep(0.1);
}
