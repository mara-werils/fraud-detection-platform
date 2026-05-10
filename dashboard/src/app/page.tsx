"use client";

import { useState, useEffect, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import MetricsCards from "@/components/MetricsCards";
import TransactionFeed from "@/components/TransactionFeed";
import FraudMap from "@/components/FraudMap";
import { useWebSocket } from "@/hooks/useWebSocket";
import { generateMockTransactions } from "@/lib/utils";
import type { Stats, ScoredTransaction, ConnectionStatus } from "@/lib/types";

const mockStats: Stats = {
  total_transactions: 1_284_392,
  fraud_rate: 0.0237,
  avg_latency_ms: 23,
  blocked_amount: 847_293.45,
  total_amount: 42_918_472.8,
  transactions_per_second: 142,
  active_users: 28_491,
  review_queue_size: 34,
};

function StatusBadge({ status }: { status: ConnectionStatus }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={`relative flex h-2 w-2 ${
          status === "connected" ? "" : "opacity-50"
        }`}
      >
        {status === "connected" && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
        )}
        <span
          className={`relative inline-flex h-2 w-2 rounded-full ${
            status === "connected"
              ? "bg-green-500"
              : status === "connecting"
              ? "bg-yellow-500"
              : "bg-gray-500"
          }`}
        />
      </span>
      <span className="text-xs text-gray-400 capitalize">{status}</span>
    </div>
  );
}

export default function DashboardPage() {
  const { messages, status } = useWebSocket();
  const [mockTxns] = useState<ScoredTransaction[]>(() =>
    generateMockTransactions(60) as ScoredTransaction[]
  );

  const transactions = messages.length > 0 ? messages : mockTxns;

  const scoreDistribution = useMemo(() => {
    const bins = Array.from({ length: 10 }, (_, i) => ({
      range: `${(i * 10).toString()}-${((i + 1) * 10).toString()}`,
      count: 0,
      fill:
        i >= 7
          ? "#ef4444"
          : i >= 3
          ? "#f97316"
          : "#22c55e",
    }));
    transactions.forEach((tx) => {
      const bin = Math.min(Math.floor(tx.fraud_score * 10), 9);
      bins[bin].count++;
    });
    return bins;
  }, [transactions]);

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="mt-1 text-sm text-gray-400">
            Real-time fraud monitoring overview
          </p>
        </div>
        <StatusBadge status={status} />
      </div>

      {/* KPI Cards */}
      <MetricsCards stats={mockStats} />

      {/* Main Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Transaction Feed */}
        <TransactionFeed transactions={transactions} />

        {/* Score Distribution */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm">
          <div className="border-b border-gray-800 px-5 py-3.5">
            <h3 className="text-sm font-semibold text-white">
              Fraud Score Distribution
            </h3>
          </div>
          <div className="p-4">
            <ResponsiveContainer width="100%" height={420}>
              <BarChart data={scoreDistribution} barCategoryGap="15%">
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis
                  dataKey="range"
                  tick={{ fill: "#6b7280", fontSize: 11 }}
                  axisLine={{ stroke: "#374151" }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "#6b7280", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#111827",
                    border: "1px solid #374151",
                    borderRadius: "8px",
                    color: "#e5e7eb",
                    fontSize: "12px",
                  }}
                />
                <Bar
                  dataKey="count"
                  radius={[4, 4, 0, 0]}
                  fill="#3b82f6"
                  // Use per-bar fill from data
                  isAnimationActive={true}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Map */}
      <FraudMap transactions={transactions} />
    </div>
  );
}
