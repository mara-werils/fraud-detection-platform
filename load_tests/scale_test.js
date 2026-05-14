/**
 * Scale test — validates the platform under 100K users load profile.
 *
 * Stages:
 *   0–2m:  Ramp up to 500 VUs (simulates morning traffic surge)
 *   2–8m:  Hold at 500 VUs (steady state, ~5K rps across 12 partitions)
 *   8–10m: Spike to 2000 VUs (flash sale / fraud event simulation)
 *   10–12m: Ramp down
 *
 * SLO Targets validated by thresholds:
 *   - Scoring p95 < 100ms
 *   - Scoring p99 < 200ms
 *   - Error rate < 0.1%
 *   - Batch scoring p95 < 500ms
 */

import http from "k6/http";
import { check, sleep, group } from "k6";
import { Counter, Rate, Trend, Gauge } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const API_KEY = __ENV.API_KEY || "fdp_dev_key_change_me_in_production";

// Custom metrics
const scoreErrors = new Counter("score_errors");
const batchErrors = new Counter("batch_errors");
const scoreLatency = new Trend("score_latency_ms", true);
const batchLatency = new Trend("batch_latency_ms", true);
const errorRate = new Rate("error_rate");
const blockRate = new Rate("block_rate");

export const options = {
  stages: [
    { duration: "2m", target: 500 },   // Ramp up
    { duration: "6m", target: 500 },   // Steady state
    { duration: "2m", target: 2000 },  // Spike
    { duration: "2m", target: 0 },     // Ramp down
  ],
  thresholds: {
    // SLO: scoring latency
    score_latency_ms: ["p(95)<100", "p(99)<200"],
    // SLO: batch latency
    batch_latency_ms: ["p(95)<500"],
    // SLO: availability
    error_rate: ["rate<0.001"],
    http_req_failed: ["rate<0.001"],
    // Overall p99
    http_req_duration: ["p(99)<500"],
  },
};

const HEADERS = {
  "Content-Type": "application/json",
  "X-API-Key": API_KEY,
};

function randomTransaction(highRisk = false) {
  const amount = highRisk
    ? parseFloat((Math.random() * 5000 + 1000).toFixed(2))
    : parseFloat((Math.random() * 200).toFixed(2));

  return {
    transaction_id: `txn_scale_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
    user_id: `user_${Math.floor(Math.random() * 100_000)}`,
    amount,
    currency: ["USD", "EUR", "GBP"][Math.floor(Math.random() * 3)],
    merchant_id: `merch_${Math.floor(Math.random() * 10_000)}`,
    merchant_category: ["retail", "travel", "gambling", "crypto"][Math.floor(Math.random() * 4)],
    ip_address: `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.0.1`,
    device_id: `dev_${Math.floor(Math.random() * 500_000)}`,
    timestamp: new Date().toISOString(),
    latitude: parseFloat((Math.random() * 180 - 90).toFixed(4)),
    longitude: parseFloat((Math.random() * 360 - 180).toFixed(4)),
  };
}

function randomBatch(size = 20) {
  return Array.from({ length: size }, () => randomTransaction(Math.random() < 0.1));
}

export default function () {
  const scenario = Math.random();

  if (scenario < 0.70) {
    // 70%: Single transaction scoring (primary use case)
    group("single_score", () => {
      const isHighRisk = Math.random() < 0.05;
      const payload = JSON.stringify(randomTransaction(isHighRisk));
      const res = http.post(`${BASE_URL}/api/v1/score`, payload, { headers: HEADERS });

      const ok = check(res, {
        "status 200": (r) => r.status === 200,
        "has fraud_score": (r) => {
          try { return JSON.parse(r.body).fraud_score !== undefined; } catch { return false; }
        },
        "valid decision": (r) => {
          try { return ["BLOCK", "REVIEW", "ALLOW"].includes(JSON.parse(r.body).decision); } catch { return false; }
        },
        "latency < 200ms": (r) => r.timings.duration < 200,
      });

      scoreLatency.add(res.timings.duration);
      if (!ok || res.status !== 200) {
        scoreErrors.add(1);
        errorRate.add(1);
      } else {
        errorRate.add(0);
        try {
          const body = JSON.parse(res.body);
          blockRate.add(body.decision === "BLOCK" ? 1 : 0);
        } catch {}
      }
    });

  } else if (scenario < 0.85) {
    // 15%: Batch scoring
    group("batch_score", () => {
      const batchSize = Math.floor(Math.random() * 40) + 10;
      const payload = JSON.stringify({ transactions: randomBatch(batchSize) });
      const res = http.post(`${BASE_URL}/api/v1/batch/score`, payload, { headers: HEADERS });

      const ok = check(res, {
        "batch 200": (r) => r.status === 200,
        "has results": (r) => {
          try { return Array.isArray(JSON.parse(r.body).results); } catch { return false; }
        },
      });

      batchLatency.add(res.timings.duration);
      if (!ok) batchErrors.add(1);
    });

  } else if (scenario < 0.95) {
    // 10%: Read transactions
    group("read_transactions", () => {
      const minScore = parseFloat((Math.random() * 0.5).toFixed(2));
      const res = http.get(
        `${BASE_URL}/api/v1/transactions?min_score=${minScore}&limit=20`,
        { headers: HEADERS }
      );
      check(res, { "txn list 200": (r) => r.status === 200 });
    });

  } else {
    // 5%: Health check (simulates load balancer probes)
    group("health", () => {
      const res = http.get(`${BASE_URL}/health/ready`);
      check(res, { "ready": (r) => r.status === 200 });
    });
  }

  // Realistic inter-request delay (Poisson distributed)
  sleep(Math.random() * 0.5);
}

export function handleSummary(data) {
  return {
    "load_tests/results/scale_test_summary.json": JSON.stringify(data, null, 2),
  };
}
