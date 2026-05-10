"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { cn } from "@/lib/utils";

// Generate mock analytics data
function generateFraudRateOverTime() {
  const data = [];
  const now = new Date();
  for (let i = 6; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    data.push({
      date: date.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" }),
      rate: 0.018 + Math.random() * 0.012,
    });
  }
  return data;
}

function generateVolumeByHour() {
  return Array.from({ length: 24 }, (_, i) => ({
    hour: `${i.toString().padStart(2, "0")}:00`,
    volume: Math.floor(
      500 +
        Math.sin((i - 6) * (Math.PI / 12)) * 400 +
        Math.random() * 200
    ),
  }));
}

function generateTopFraudCategories() {
  const cats = [
    { category: "crypto", fraud_rate: 0.089 },
    { category: "gaming", fraud_rate: 0.067 },
    { category: "electronics", fraud_rate: 0.052 },
    { category: "travel", fraud_rate: 0.041 },
    { category: "payment", fraud_rate: 0.034 },
    { category: "retail", fraud_rate: 0.021 },
    { category: "food", fraud_rate: 0.012 },
    { category: "streaming", fraud_rate: 0.009 },
    { category: "transport", fraud_rate: 0.007 },
    { category: "healthcare", fraud_rate: 0.004 },
  ];
  return cats.map((c) => ({
    ...c,
    count: Math.floor(Math.random() * 5000 + 500),
    fraud_count: Math.floor(Math.random() * 200 + 20),
  }));
}

function generateAmountDistribution() {
  const ranges = [
    "$0-50",
    "$50-100",
    "$100-250",
    "$250-500",
    "$500-1K",
    "$1K-2.5K",
    "$2.5K-5K",
    "$5K+",
  ];
  return ranges.map((range, i) => ({
    range,
    count: Math.floor(Math.max(50, 3000 * Math.exp(-i * 0.5) + Math.random() * 300)),
  }));
}

function generateHeatmap() {
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const data: { day: string; hour: number; value: number; dayIdx: number }[] = [];
  for (let d = 0; d < 7; d++) {
    for (let h = 0; h < 24; h++) {
      const baseRate = 0.015;
      const timeEffect = Math.sin((h - 3) * (Math.PI / 12)) * 0.01;
      const weekendEffect = d >= 5 ? 0.005 : 0;
      data.push({
        day: days[d],
        dayIdx: d,
        hour: h,
        value: Math.max(0, baseRate + timeEffect + weekendEffect + (Math.random() - 0.5) * 0.008),
      });
    }
  }
  return data;
}

const chartTooltipStyle = {
  backgroundColor: "#111827",
  border: "1px solid #374151",
  borderRadius: "8px",
  color: "#e5e7eb",
  fontSize: "12px",
};

const axisTickStyle = { fill: "#6b7280", fontSize: 11 };

export default function AnalyticsPage() {
  const fraudRate = useMemo(generateFraudRateOverTime, []);
  const volumeByHour = useMemo(generateVolumeByHour, []);
  const topCategories = useMemo(generateTopFraudCategories, []);
  const amountDist = useMemo(generateAmountDistribution, []);
  const heatmap = useMemo(generateHeatmap, []);

  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Analytics</h1>
        <p className="mt-1 text-sm text-gray-400">
          Fraud patterns and transaction analytics
        </p>
      </div>

      {/* Row 1: Fraud Rate + Volume */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Fraud Rate Over Time */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm">
          <div className="border-b border-gray-800 px-5 py-3.5">
            <h3 className="text-sm font-semibold text-white">
              Fraud Rate Over Time
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">Last 7 days</p>
          </div>
          <div className="p-4">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={fraudRate}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis
                  dataKey="date"
                  tick={axisTickStyle}
                  axisLine={{ stroke: "#374151" }}
                  tickLine={false}
                />
                <YAxis
                  tick={axisTickStyle}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
                />
                <Tooltip
                  contentStyle={chartTooltipStyle}
                  formatter={(value: number) => [
                    `${(value * 100).toFixed(2)}%`,
                    "Fraud Rate",
                  ]}
                />
                <Line
                  type="monotone"
                  dataKey="rate"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={{ fill: "#ef4444", r: 3 }}
                  activeDot={{ r: 5, fill: "#ef4444" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Transaction Volume by Hour */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm">
          <div className="border-b border-gray-800 px-5 py-3.5">
            <h3 className="text-sm font-semibold text-white">
              Transaction Volume by Hour
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">Today</p>
          </div>
          <div className="p-4">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={volumeByHour}>
                <defs>
                  <linearGradient id="volumeGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis
                  dataKey="hour"
                  tick={axisTickStyle}
                  axisLine={{ stroke: "#374151" }}
                  tickLine={false}
                  interval={2}
                />
                <YAxis
                  tick={axisTickStyle}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip contentStyle={chartTooltipStyle} />
                <Area
                  type="monotone"
                  dataKey="volume"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  fill="url(#volumeGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Row 2: Top Categories + Amount Distribution */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Top Fraud Categories */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm">
          <div className="border-b border-gray-800 px-5 py-3.5">
            <h3 className="text-sm font-semibold text-white">
              Top Fraud Categories
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">By fraud rate</p>
          </div>
          <div className="p-4">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart
                data={topCategories}
                layout="vertical"
                margin={{ left: 20 }}
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
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <YAxis
                  type="category"
                  dataKey="category"
                  tick={axisTickStyle}
                  axisLine={false}
                  tickLine={false}
                  width={80}
                />
                <Tooltip
                  contentStyle={chartTooltipStyle}
                  formatter={(value: number) => [
                    `${(value * 100).toFixed(2)}%`,
                    "Fraud Rate",
                  ]}
                />
                <Bar
                  dataKey="fraud_rate"
                  fill="#ef4444"
                  radius={[0, 4, 4, 0]}
                  barSize={18}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Amount Distribution */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm">
          <div className="border-b border-gray-800 px-5 py-3.5">
            <h3 className="text-sm font-semibold text-white">
              Amount Distribution
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Transaction amounts histogram
            </p>
          </div>
          <div className="p-4">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={amountDist} barCategoryGap="20%">
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis
                  dataKey="range"
                  tick={axisTickStyle}
                  axisLine={{ stroke: "#374151" }}
                  tickLine={false}
                />
                <YAxis
                  tick={axisTickStyle}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip contentStyle={chartTooltipStyle} />
                <Bar
                  dataKey="count"
                  fill="#8b5cf6"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Row 3: Hourly Heatmap */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm">
        <div className="border-b border-gray-800 px-5 py-3.5">
          <h3 className="text-sm font-semibold text-white">
            Fraud Rate Heatmap
          </h3>
          <p className="text-xs text-gray-500 mt-0.5">
            Day of week vs hour of day
          </p>
        </div>
        <div className="p-5 overflow-x-auto">
          <div className="min-w-[700px]">
            {/* Hour labels */}
            <div className="flex ml-12 mb-1">
              {Array.from({ length: 24 }, (_, h) => (
                <div
                  key={h}
                  className="flex-1 text-center text-[10px] text-gray-600"
                >
                  {h % 3 === 0 ? `${h.toString().padStart(2, "0")}` : ""}
                </div>
              ))}
            </div>
            {/* Grid rows */}
            {days.map((day, dIdx) => (
              <div key={day} className="flex items-center gap-1 mb-1">
                <div className="w-10 text-right text-xs text-gray-500 pr-2">
                  {day}
                </div>
                <div className="flex flex-1 gap-0.5">
                  {Array.from({ length: 24 }, (_, h) => {
                    const cell = heatmap.find(
                      (c) => c.dayIdx === dIdx && c.hour === h
                    );
                    const val = cell?.value || 0;
                    const intensity = Math.min(val / 0.04, 1);
                    return (
                      <div
                        key={h}
                        className="flex-1 aspect-square rounded-sm transition-colors"
                        style={{
                          backgroundColor: `rgba(239, 68, 68, ${
                            0.05 + intensity * 0.8
                          })`,
                        }}
                        title={`${day} ${h}:00 - ${(val * 100).toFixed(2)}%`}
                      />
                    );
                  })}
                </div>
              </div>
            ))}
            {/* Legend */}
            <div className="flex items-center justify-end gap-2 mt-3">
              <span className="text-[10px] text-gray-500">Low</span>
              <div className="flex gap-0.5">
                {[0.1, 0.25, 0.4, 0.55, 0.7, 0.85].map((v) => (
                  <div
                    key={v}
                    className="h-3 w-5 rounded-sm"
                    style={{
                      backgroundColor: `rgba(239, 68, 68, ${0.05 + v * 0.8})`,
                    }}
                  />
                ))}
              </div>
              <span className="text-[10px] text-gray-500">High</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
