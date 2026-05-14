/**
 * Soak test — sustained load over 1 hour to detect memory leaks,
 * connection pool exhaustion, and slow performance degradation.
 *
 * Run with: k6 run --env BASE_URL=http://your-host:8000 soak_test.js
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const API_KEY = __ENV.API_KEY || "fdp_dev_key_change_me_in_production";

const errorRate = new Rate("error_rate");
const p99Trend = new Trend("soak_p99_latency", true);

export const options = {
  stages: [
    { duration: "5m", target: 200 },   // Ramp up
    { duration: "50m", target: 200 },  // Sustained load
    { duration: "5m", target: 0 },     // Cool down
  ],
  thresholds: {
    error_rate: ["rate<0.005"],          // < 0.5% error rate during soak
    http_req_duration: ["p(99)<500"],    // p99 stays below 500ms throughout
    soak_p99_latency: ["p(99)<300"],
  },
};

const HEADERS = {
  "Content-Type": "application/json",
  "X-API-Key": API_KEY,
};

export default function () {
  const res = http.post(
    `${BASE_URL}/api/v1/score`,
    JSON.stringify({
      transaction_id: `txn_soak_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      user_id: `user_${Math.floor(Math.random() * 50_000)}`,
      amount: parseFloat((Math.random() * 300).toFixed(2)),
      currency: "USD",
      merchant_id: `merch_${Math.floor(Math.random() * 5_000)}`,
      timestamp: new Date().toISOString(),
    }),
    { headers: HEADERS }
  );

  const ok = check(res, {
    "status 200": (r) => r.status === 200,
    "no timeout": (r) => r.timings.duration < 1000,
  });

  errorRate.add(!ok);
  p99Trend.add(res.timings.duration);

  sleep(0.2);
}

export function handleSummary(data) {
  const p99 = data.metrics["soak_p99_latency"]?.values?.["p(99)"];
  const errors = data.metrics["error_rate"]?.values?.rate;

  console.log(`Soak test complete — p99: ${p99?.toFixed(1)}ms, error rate: ${(errors * 100)?.toFixed(3)}%`);

  return {
    "load_tests/results/soak_test_summary.json": JSON.stringify(data, null, 2),
  };
}
