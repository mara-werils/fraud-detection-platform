"use client";

import KPICard from "./KPICard";
import {
  ArrowLeftRight,
  ShieldAlert,
  Timer,
  Ban,
} from "lucide-react";
import { formatCurrency, formatCompactNumber } from "@/lib/utils";
import type { Stats } from "@/lib/types";

interface MetricsCardsProps {
  stats: Stats;
}

export default function MetricsCards({ stats }: MetricsCardsProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <KPICard
        title="Total Transactions"
        value={formatCompactNumber(stats.total_transactions)}
        change={12.3}
        changeLabel="vs yesterday"
        icon={ArrowLeftRight}
        iconColor="text-blue-400"
      />
      <KPICard
        title="Fraud Rate"
        value={`${(stats.fraud_rate * 100).toFixed(2)}%`}
        change={-0.3}
        changeLabel="vs yesterday"
        icon={ShieldAlert}
        iconColor="text-red-400"
      />
      <KPICard
        title="Avg Latency"
        value={`${stats.avg_latency_ms.toFixed(0)}ms`}
        change={-5.2}
        changeLabel="vs yesterday"
        icon={Timer}
        iconColor="text-yellow-400"
      />
      <KPICard
        title="Blocked Amount"
        value={formatCurrency(stats.blocked_amount)}
        change={8.7}
        changeLabel="today"
        icon={Ban}
        iconColor="text-orange-400"
      />
    </div>
  );
}
