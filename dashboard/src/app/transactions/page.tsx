"use client";

import { useState, useMemo } from "react";
import { Search, Filter, X } from "lucide-react";
import DataTable, { type Column } from "@/components/DataTable";
import TransactionDetail from "@/components/TransactionDetail";
import {
  cn,
  truncateId,
  formatCurrency,
  getScoreColor,
  getDecisionColor,
  timeAgo,
  generateMockTransactions,
} from "@/lib/utils";
import type { ScoredTransaction } from "@/lib/types";

const categories = [
  "all",
  "retail",
  "electronics",
  "gaming",
  "streaming",
  "transport",
  "travel",
  "payment",
  "crypto",
  "food",
  "healthcare",
];

export default function TransactionsPage() {
  const [allTransactions] = useState<ScoredTransaction[]>(
    () => generateMockTransactions(200) as ScoredTransaction[]
  );
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [decision, setDecision] = useState("all");
  const [category, setCategory] = useState("all");
  const [minScore, setMinScore] = useState("");
  const [maxScore, setMaxScore] = useState("");
  const [showFilters, setShowFilters] = useState(false);

  const filtered = useMemo(() => {
    return allTransactions.filter((tx) => {
      if (
        search &&
        !tx.txn_id.toLowerCase().includes(search.toLowerCase()) &&
        !tx.user_id.toLowerCase().includes(search.toLowerCase())
      ) {
        return false;
      }
      if (decision !== "all" && tx.decision !== decision) return false;
      if (category !== "all" && tx.merchant_category !== category) return false;
      if (minScore && tx.fraud_score < parseFloat(minScore) / 100) return false;
      if (maxScore && tx.fraud_score > parseFloat(maxScore) / 100) return false;
      return true;
    });
  }, [allTransactions, search, decision, category, minScore, maxScore]);

  const columns: Column<ScoredTransaction>[] = [
    {
      key: "time",
      header: "Time",
      sortable: true,
      sortValue: (tx) => new Date(tx.timestamp).getTime(),
      render: (tx) => (
        <span className="text-gray-400 text-xs">{timeAgo(tx.timestamp)}</span>
      ),
    },
    {
      key: "txn_id",
      header: "TxnID",
      render: (tx) => (
        <span className="font-mono text-xs text-gray-300">
          {truncateId(tx.txn_id, 12)}
        </span>
      ),
    },
    {
      key: "amount",
      header: "Amount",
      sortable: true,
      sortValue: (tx) => tx.amount,
      render: (tx) => (
        <span className="font-medium text-gray-200">
          {formatCurrency(tx.amount)}
        </span>
      ),
    },
    {
      key: "user",
      header: "User",
      render: (tx) => (
        <span className="font-mono text-xs text-gray-400">{tx.user_id}</span>
      ),
    },
    {
      key: "score",
      header: "Score",
      sortable: true,
      sortValue: (tx) => tx.fraud_score,
      render: (tx) => (
        <span
          className={cn(
            "font-mono font-semibold",
            getScoreColor(tx.fraud_score)
          )}
        >
          {(tx.fraud_score * 100).toFixed(0)}
        </span>
      ),
    },
    {
      key: "decision",
      header: "Decision",
      sortable: true,
      sortValue: (tx) => tx.decision,
      render: (tx) => (
        <span
          className={cn(
            "inline-flex rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
            getDecisionColor(tx.decision)
          )}
        >
          {tx.decision}
        </span>
      ),
    },
    {
      key: "category",
      header: "Category",
      sortable: true,
      sortValue: (tx) => tx.merchant_category,
      render: (tx) => (
        <span className="text-xs text-gray-400 capitalize">
          {tx.merchant_category}
        </span>
      ),
    },
  ];

  const clearFilters = () => {
    setSearch("");
    setDecision("all");
    setCategory("all");
    setMinScore("");
    setMaxScore("");
  };

  const hasActiveFilters =
    search || decision !== "all" || category !== "all" || minScore || maxScore;

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Transactions</h1>
        <p className="mt-1 text-sm text-gray-400">
          Browse and analyze scored transactions
        </p>
      </div>

      {/* Search and Filter Bar */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by txn_id or user_id..."
              className="w-full rounded-lg border border-gray-700 bg-gray-800/60 py-2.5 pl-10 pr-4 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={cn(
              "flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-medium transition-colors",
              showFilters || hasActiveFilters
                ? "border-blue-500/50 bg-blue-500/10 text-blue-400"
                : "border-gray-700 bg-gray-800/60 text-gray-400 hover:text-gray-200"
            )}
          >
            <Filter className="h-4 w-4" />
            Filters
            {hasActiveFilters && (
              <span className="flex h-4 w-4 items-center justify-center rounded-full bg-blue-500 text-[10px] text-white">
                !
              </span>
            )}
          </button>
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-1 rounded-lg border border-gray-700 bg-gray-800/60 px-3 py-2.5 text-xs text-gray-400 hover:text-gray-200 transition-colors"
            >
              <X className="h-3 w-3" />
              Clear
            </button>
          )}
        </div>

        {showFilters && (
          <div className="grid grid-cols-2 gap-4 rounded-xl border border-gray-800 bg-gray-900/60 p-4 md:grid-cols-4 animate-fade-in">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-500">
                Decision
              </label>
              <select
                value={decision}
                onChange={(e) => setDecision(e.target.value)}
                className="w-full rounded-lg border border-gray-700 bg-gray-800 py-2 pl-3 pr-8 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
              >
                <option value="all">All</option>
                <option value="ALLOW">Allow</option>
                <option value="REVIEW">Review</option>
                <option value="BLOCK">Block</option>
              </select>
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-500">
                Category
              </label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="w-full rounded-lg border border-gray-700 bg-gray-800 py-2 pl-3 pr-8 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
              >
                {categories.map((c) => (
                  <option key={c} value={c}>
                    {c === "all" ? "All Categories" : c}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-500">
                Min Score
              </label>
              <input
                type="number"
                value={minScore}
                onChange={(e) => setMinScore(e.target.value)}
                placeholder="0"
                min="0"
                max="100"
                className="w-full rounded-lg border border-gray-700 bg-gray-800 py-2 px-3 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-500">
                Max Score
              </label>
              <input
                type="number"
                value={maxScore}
                onChange={(e) => setMaxScore(e.target.value)}
                placeholder="100"
                min="0"
                max="100"
                className="w-full rounded-lg border border-gray-700 bg-gray-800 py-2 px-3 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
        )}
      </div>

      {/* Results count */}
      <div className="text-xs text-gray-500">
        {filtered.length} transactions found
      </div>

      {/* Data Table */}
      <DataTable
        data={filtered}
        columns={columns}
        keyExtractor={(tx) => tx.txn_id}
        pageSize={50}
        onRowClick={(tx) =>
          setExpandedRow(expandedRow === tx.txn_id ? null : tx.txn_id)
        }
        expandedRow={expandedRow}
        renderExpanded={(tx) => <TransactionDetail transaction={tx} />}
        emptyMessage="No transactions match your filters"
      />
    </div>
  );
}
