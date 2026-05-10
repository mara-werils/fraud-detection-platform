"use client";

import { cn, truncateId, formatCurrency, getScoreColor, getDecisionColor, timeAgo } from "@/lib/utils";
import type { ScoredTransaction } from "@/lib/types";

interface TransactionFeedProps {
  transactions: ScoredTransaction[];
  maxItems?: number;
}

export default function TransactionFeed({
  transactions,
  maxItems = 50,
}: TransactionFeedProps) {
  const items = transactions.slice(0, maxItems);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm">
      <div className="flex items-center justify-between border-b border-gray-800 px-5 py-3.5">
        <h3 className="text-sm font-semibold text-white">
          Live Transaction Feed
        </h3>
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
          </span>
          <span className="text-xs text-gray-400">{items.length} recent</span>
        </div>
      </div>

      <div className="max-h-[480px] overflow-y-auto scrollbar-thin">
        {items.length === 0 ? (
          <div className="flex h-32 items-center justify-center text-sm text-gray-500">
            Waiting for transactions...
          </div>
        ) : (
          <div className="divide-y divide-gray-800/50">
            {items.map((tx) => (
              <div
                key={tx.txn_id}
                className="flex items-center gap-4 px-5 py-3 transition-colors hover:bg-gray-800/30 animate-fade-in"
              >
                <div
                  className={cn(
                    "h-2 w-2 shrink-0 rounded-full",
                    tx.fraud_score >= 0.7
                      ? "bg-red-500"
                      : tx.fraud_score >= 0.3
                      ? "bg-yellow-500"
                      : "bg-green-500"
                  )}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-200">
                      {formatCurrency(tx.amount)}
                    </span>
                    <span className="text-xs text-gray-500">
                      {truncateId(tx.txn_id)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-xs text-gray-500">
                      {tx.merchant}
                    </span>
                    <span className="text-xs text-gray-600">|</span>
                    <span className="text-xs text-gray-500">
                      {tx.merchant_category}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span
                    className={cn(
                      "text-sm font-mono font-medium",
                      getScoreColor(tx.fraud_score)
                    )}
                  >
                    {(tx.fraud_score * 100).toFixed(0)}
                  </span>
                  <span
                    className={cn(
                      "inline-flex rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
                      getDecisionColor(tx.decision)
                    )}
                  >
                    {tx.decision}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
