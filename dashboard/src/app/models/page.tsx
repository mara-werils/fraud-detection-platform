"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Brain, Shield, GitBranch, Clock, CheckCircle } from "lucide-react";
import ModelComparison from "@/components/ModelComparison";
import { cn } from "@/lib/utils";
import type { ModelInfo, ABTestResult, FeatureImportance } from "@/lib/types";

const mockModels: ModelInfo[] = [
  {
    name: "XGBoost Ensemble",
    version: "v2.3.1",
    type: "xgboost",
    auc: 0.9847,
    accuracy: 0.9923,
    precision: 0.9312,
    recall: 0.8745,
    f1_score: 0.9019,
    last_trained: "2026-05-08T14:30:00Z",
    status: "active",
    latency_p50_ms: 12.3,
    latency_p99_ms: 45.8,
  },
  {
    name: "Graph Neural Network",
    version: "v1.2.0",
    type: "gnn",
    auc: 0.9712,
    accuracy: 0.9891,
    precision: 0.9145,
    recall: 0.8923,
    f1_score: 0.9033,
    last_trained: "2026-05-06T09:15:00Z",
    status: "shadow",
    latency_p50_ms: 28.7,
    latency_p99_ms: 89.4,
  },
  {
    name: "Rule Engine",
    version: "v4.1.0",
    type: "rules",
    auc: 0.8234,
    accuracy: 0.9567,
    precision: 0.7891,
    recall: 0.9234,
    f1_score: 0.8509,
    last_trained: "2026-05-01T12:00:00Z",
    status: "active",
    latency_p50_ms: 2.1,
    latency_p99_ms: 8.3,
  },
];

const mockABTests: ABTestResult[] = [
  {
    name: "XGBoost v2.3.1 vs v2.4.0-rc1",
    start_date: "2026-05-05T00:00:00Z",
    traffic_split: 10,
    status: "running",
    control: {
      model_version: "v2.3.1",
      auc: 0.9847,
      precision: 0.9312,
      recall: 0.8745,
      f1_score: 0.9019,
      avg_latency_ms: 12.3,
      false_positive_rate: 0.0068,
      transactions_scored: 1_156_230,
    },
    challenger: {
      model_version: "v2.4.0-rc1",
      auc: 0.9891,
      precision: 0.9398,
      recall: 0.8812,
      f1_score: 0.9096,
      avg_latency_ms: 13.1,
      false_positive_rate: 0.0059,
      transactions_scored: 128_470,
    },
  },
];

const mockFeatureImportance: FeatureImportance[] = [
  { feature: "amount_zscore", importance: 0.182, shap_value: 0.34 },
  { feature: "velocity_score", importance: 0.156, shap_value: 0.29 },
  { feature: "ip_risk_score", importance: 0.134, shap_value: 0.25 },
  { feature: "distance_from_home", importance: 0.098, shap_value: 0.18 },
  { feature: "user_tx_count_24h", importance: 0.087, shap_value: 0.16 },
  { feature: "time_since_last_tx", importance: 0.076, shap_value: 0.14 },
  { feature: "is_international", importance: 0.065, shap_value: 0.12 },
  { feature: "device_fingerprint_match", importance: 0.054, shap_value: 0.1 },
  { feature: "user_unique_merchants_24h", importance: 0.043, shap_value: 0.08 },
  { feature: "user_avg_amount", importance: 0.032, shap_value: 0.06 },
  { feature: "billing_shipping_match", importance: 0.028, shap_value: 0.05 },
  { feature: "hour_of_day", importance: 0.019, shap_value: 0.04 },
  { feature: "merchant_category", importance: 0.014, shap_value: 0.03 },
  { feature: "day_of_week", importance: 0.008, shap_value: 0.015 },
  { feature: "amount", importance: 0.004, shap_value: 0.008 },
];

const chartTooltipStyle = {
  backgroundColor: "#111827",
  border: "1px solid #374151",
  borderRadius: "8px",
  color: "#e5e7eb",
  fontSize: "12px",
};

const axisTickStyle = { fill: "#6b7280", fontSize: 11 };

function ModelIcon({ type }: { type: string }) {
  switch (type) {
    case "xgboost":
      return <Brain className="h-5 w-5 text-blue-400" />;
    case "gnn":
      return <GitBranch className="h-5 w-5 text-purple-400" />;
    case "rules":
      return <Shield className="h-5 w-5 text-yellow-400" />;
    default:
      return <Brain className="h-5 w-5 text-gray-400" />;
  }
}

export default function ModelsPage() {
  const latencyComparison = useMemo(
    () =>
      mockModels.map((m) => ({
        name: m.name.split(" ")[0],
        p50: m.latency_p50_ms,
        p99: m.latency_p99_ms,
      })),
    []
  );

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Models</h1>
        <p className="mt-1 text-sm text-gray-400">
          Model performance, A/B tests, and feature analysis
        </p>
      </div>

      {/* Model Info Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        {mockModels.map((model) => (
          <div
            key={model.name}
            className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 backdrop-blur-sm"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-gray-800/80 p-2.5">
                  <ModelIcon type={model.type} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white">
                    {model.name}
                  </h3>
                  <span className="text-xs font-mono text-gray-500">
                    {model.version}
                  </span>
                </div>
              </div>
              <span
                className={cn(
                  "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
                  model.status === "active"
                    ? "bg-green-500/20 text-green-400"
                    : model.status === "shadow"
                    ? "bg-blue-500/20 text-blue-400"
                    : "bg-gray-500/20 text-gray-400"
                )}
              >
                {model.status === "active" && (
                  <CheckCircle className="h-2.5 w-2.5" />
                )}
                {model.status}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg bg-gray-800/40 px-3 py-2">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider">
                  AUC
                </div>
                <div className="text-lg font-bold text-white">
                  {model.auc.toFixed(4)}
                </div>
              </div>
              <div className="rounded-lg bg-gray-800/40 px-3 py-2">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider">
                  F1 Score
                </div>
                <div className="text-lg font-bold text-white">
                  {model.f1_score.toFixed(4)}
                </div>
              </div>
              <div className="rounded-lg bg-gray-800/40 px-3 py-2">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider">
                  Precision
                </div>
                <div className="text-sm font-semibold text-gray-200">
                  {model.precision.toFixed(4)}
                </div>
              </div>
              <div className="rounded-lg bg-gray-800/40 px-3 py-2">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider">
                  Recall
                </div>
                <div className="text-sm font-semibold text-gray-200">
                  {model.recall.toFixed(4)}
                </div>
              </div>
            </div>

            <div className="mt-3 flex items-center gap-1.5 text-xs text-gray-500">
              <Clock className="h-3 w-3" />
              Last trained:{" "}
              {new Date(model.last_trained).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </div>
          </div>
        ))}
      </div>

      {/* A/B Test Results */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">
          A/B Test Results
        </h2>
        <ModelComparison tests={mockABTests} />
      </div>

      {/* Row: Feature Importance + Latency */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Feature Importance */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm">
          <div className="border-b border-gray-800 px-5 py-3.5">
            <h3 className="text-sm font-semibold text-white">
              Feature Importance (SHAP)
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Top 15 features from XGBoost
            </p>
          </div>
          <div className="p-4">
            <ResponsiveContainer width="100%" height={480}>
              <BarChart
                data={mockFeatureImportance}
                layout="vertical"
                margin={{ left: 40 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#1f2937"
                  horizontal={false}
                />
                <XAxis
                  type="number"
                  tick={axisTickStyle}
                  axisLine={{ stroke: "#374151" }}
                  tickLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={axisTickStyle}
                  axisLine={false}
                  tickLine={false}
                  width={160}
                />
                <Tooltip
                  contentStyle={chartTooltipStyle}
                  formatter={(value: number) => [
                    value.toFixed(4),
                    "Importance",
                  ]}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={16}>
                  {mockFeatureImportance.map((_, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={`hsl(${210 + index * 8}, 80%, ${55 - index * 1.5}%)`}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Latency Comparison */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm">
          <div className="border-b border-gray-800 px-5 py-3.5">
            <h3 className="text-sm font-semibold text-white">
              Model Latency Comparison
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">P50 and P99 in ms</p>
          </div>
          <div className="p-4">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={latencyComparison} barCategoryGap="30%">
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis
                  dataKey="name"
                  tick={axisTickStyle}
                  axisLine={{ stroke: "#374151" }}
                  tickLine={false}
                />
                <YAxis
                  tick={axisTickStyle}
                  axisLine={false}
                  tickLine={false}
                  label={{
                    value: "ms",
                    position: "insideTopLeft",
                    style: { fill: "#6b7280", fontSize: 11 },
                  }}
                />
                <Tooltip
                  contentStyle={chartTooltipStyle}
                  formatter={(value: number) => [`${value.toFixed(1)}ms`]}
                />
                <Bar
                  dataKey="p50"
                  name="P50"
                  fill="#3b82f6"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="p99"
                  name="P99"
                  fill="#f97316"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Latency details table */}
          <div className="border-t border-gray-800/50 px-5 py-4">
            <div className="space-y-3">
              {mockModels.map((model) => (
                <div
                  key={model.name}
                  className="flex items-center justify-between text-xs"
                >
                  <span className="text-gray-400">{model.name}</span>
                  <div className="flex items-center gap-4">
                    <span className="text-gray-300">
                      P50:{" "}
                      <span className="font-mono text-blue-400">
                        {model.latency_p50_ms}ms
                      </span>
                    </span>
                    <span className="text-gray-300">
                      P99:{" "}
                      <span className="font-mono text-orange-400">
                        {model.latency_p99_ms}ms
                      </span>
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
