"use client";

import { cn } from "@/lib/utils";
import type { ABTestResult } from "@/lib/types";

interface ModelComparisonProps {
  tests: ABTestResult[];
}

const metricLabels: Record<string, string> = {
  auc: "AUC",
  precision: "Precision",
  recall: "Recall",
  f1_score: "F1 Score",
  avg_latency_ms: "Avg Latency (ms)",
  false_positive_rate: "False Positive Rate",
  transactions_scored: "Transactions Scored",
};

export default function ModelComparison({ tests }: ModelComparisonProps) {
  return (
    <div className="space-y-6">
      {tests.map((test) => (
        <div
          key={test.name}
          className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm"
        >
          <div className="flex items-center justify-between border-b border-gray-800 px-5 py-3.5">
            <h3 className="text-sm font-semibold text-white">{test.name}</h3>
            <div className="flex items-center gap-3">
              <span
                className={cn(
                  "inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
                  test.status === "running"
                    ? "bg-green-500/20 text-green-400"
                    : test.status === "completed"
                    ? "bg-blue-500/20 text-blue-400"
                    : "bg-gray-500/20 text-gray-400"
                )}
              >
                {test.status}
              </span>
              <span className="text-xs text-gray-500">
                {test.traffic_split}% traffic
              </span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800/50">
                  <th className="px-5 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Metric
                  </th>
                  <th className="px-5 py-2.5 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Control ({test.control.model_version})
                  </th>
                  <th className="px-5 py-2.5 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Challenger ({test.challenger.model_version})
                  </th>
                  <th className="px-5 py-2.5 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Diff
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/30">
                {Object.entries(metricLabels).map(([key, label]) => {
                  const controlVal =
                    test.control[key as keyof typeof test.control] as number;
                  const challengerVal =
                    test.challenger[key as keyof typeof test.challenger] as number;
                  const diff = challengerVal - controlVal;
                  const isLatency = key.includes("latency");
                  const isFPR = key === "false_positive_rate";
                  const isBetter = isLatency || isFPR ? diff < 0 : diff > 0;

                  return (
                    <tr key={key} className="hover:bg-gray-800/20">
                      <td className="px-5 py-2.5 text-sm text-gray-300">
                        {label}
                      </td>
                      <td className="px-5 py-2.5 text-right text-sm font-mono text-gray-300">
                        {key === "transactions_scored"
                          ? controlVal.toLocaleString()
                          : key.includes("latency")
                          ? `${controlVal.toFixed(1)}ms`
                          : controlVal.toFixed(4)}
                      </td>
                      <td className="px-5 py-2.5 text-right text-sm font-mono text-gray-300">
                        {key === "transactions_scored"
                          ? challengerVal.toLocaleString()
                          : key.includes("latency")
                          ? `${challengerVal.toFixed(1)}ms`
                          : challengerVal.toFixed(4)}
                      </td>
                      <td
                        className={cn(
                          "px-5 py-2.5 text-right text-sm font-mono font-medium",
                          isBetter ? "text-green-400" : "text-red-400"
                        )}
                      >
                        {diff > 0 ? "+" : ""}
                        {key === "transactions_scored"
                          ? diff.toLocaleString()
                          : key.includes("latency")
                          ? `${diff.toFixed(1)}ms`
                          : diff.toFixed(4)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}
