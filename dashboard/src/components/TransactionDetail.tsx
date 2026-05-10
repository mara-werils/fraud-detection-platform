"use client";

import ScoreGauge from "./ScoreGauge";
import { cn, formatCurrency, timeAgo } from "@/lib/utils";
import type { ScoredTransaction } from "@/lib/types";

interface TransactionDetailProps {
  transaction: ScoredTransaction;
}

export default function TransactionDetail({
  transaction: tx,
}: TransactionDetailProps) {
  const featureEntries = Object.entries(tx.features);

  return (
    <div className="grid gap-6 p-6 md:grid-cols-3">
      {/* Score and Decision */}
      <div className="flex flex-col items-center gap-4 rounded-lg border border-gray-700/50 bg-gray-800/40 p-5">
        <ScoreGauge score={tx.fraud_score} size={100} />
        <div className="text-center">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
            Decision
          </div>
          <span
            className={cn(
              "inline-flex rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wider",
              tx.decision === "BLOCK"
                ? "bg-red-500/20 text-red-400 border-red-500/30"
                : tx.decision === "REVIEW"
                ? "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
                : "bg-green-500/20 text-green-400 border-green-500/30"
            )}
          >
            {tx.decision}
          </span>
        </div>
        <div className="w-full space-y-2 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-500">Model</span>
            <span className="text-gray-300 font-mono">{tx.model_version}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Latency</span>
            <span className="text-gray-300">{tx.processing_time_ms}ms</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Scored</span>
            <span className="text-gray-300">{timeAgo(tx.scored_at)}</span>
          </div>
        </div>
      </div>

      {/* Transaction Details */}
      <div className="rounded-lg border border-gray-700/50 bg-gray-800/40 p-5">
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Transaction Details
        </h4>
        <div className="space-y-2.5 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-500">TxnID</span>
            <span className="text-gray-300 font-mono text-[11px]">{tx.txn_id}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">UserID</span>
            <span className="text-gray-300 font-mono">{tx.user_id}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Amount</span>
            <span className="text-gray-300 font-semibold">{formatCurrency(tx.amount)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Merchant</span>
            <span className="text-gray-300">{tx.merchant}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Category</span>
            <span className="text-gray-300">{tx.merchant_category}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">International</span>
            <span className="text-gray-300">{tx.is_international ? "Yes" : "No"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">IP Address</span>
            <span className="text-gray-300 font-mono">{tx.ip_address}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Location</span>
            <span className="text-gray-300">
              {tx.latitude.toFixed(2)}, {tx.longitude.toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Features and Explanation */}
      <div className="space-y-4">
        {tx.explanation && (
          <div className="rounded-lg border border-gray-700/50 bg-gray-800/40 p-5">
            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              LLM Explanation
            </h4>
            <p className="text-xs leading-relaxed text-gray-300">
              {tx.explanation}
            </p>
          </div>
        )}
        <div className="rounded-lg border border-gray-700/50 bg-gray-800/40 p-5">
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Feature Vector
          </h4>
          <div className="max-h-[200px] overflow-y-auto space-y-1.5 scrollbar-thin">
            {featureEntries.map(([key, value]) => (
              <div
                key={key}
                className="flex justify-between text-[11px] py-0.5"
              >
                <span className="text-gray-500 font-mono">{key}</span>
                <span className="text-gray-300 font-mono">
                  {typeof value === "boolean"
                    ? value
                      ? "true"
                      : "false"
                    : typeof value === "number"
                    ? value.toFixed(4)
                    : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
