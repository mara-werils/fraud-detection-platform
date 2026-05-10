import { formatDistanceToNow } from "date-fns";
import clsx from "clsx";

export function formatCurrency(amount: number, currency = "USD"): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount);
}

export function formatCompactNumber(num: number): string {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
  return num.toFixed(0);
}

export function formatScore(score: number): string {
  return (score * 100).toFixed(1);
}

export function formatPercent(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function getDecisionColor(decision: string): string {
  switch (decision) {
    case "BLOCK":
      return "bg-red-500/20 text-red-400 border-red-500/30";
    case "REVIEW":
      return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
    case "ALLOW":
      return "bg-green-500/20 text-green-400 border-green-500/30";
    default:
      return "bg-gray-500/20 text-gray-400 border-gray-500/30";
  }
}

export function getScoreColor(score: number): string {
  if (score >= 0.7) return "text-red-400";
  if (score >= 0.3) return "text-yellow-400";
  return "text-green-400";
}

export function getScoreBgColor(score: number): string {
  if (score >= 0.7) return "bg-red-500";
  if (score >= 0.3) return "bg-yellow-500";
  return "bg-green-500";
}

export function truncateId(id: string, length = 8): string {
  if (id.length <= length) return id;
  return `${id.slice(0, length)}...`;
}

export function timeAgo(dateString: string): string {
  try {
    return formatDistanceToNow(new Date(dateString), { addSuffix: true });
  } catch {
    return dateString;
  }
}

export function cn(...inputs: (string | undefined | false | null)[]): string {
  return clsx(inputs);
}

export function generateMockTransaction(index: number) {
  const merchants = [
    "Amazon", "Walmart", "Target", "Best Buy", "Apple Store",
    "Steam", "Netflix", "Uber", "Airbnb", "Stripe",
    "PayPal Transfer", "Western Union", "Coinbase", "Binance",
  ];
  const categories = [
    "retail", "electronics", "gaming", "streaming", "transport",
    "travel", "payment", "crypto", "food", "healthcare",
  ];
  const decisions: Array<"ALLOW" | "REVIEW" | "BLOCK"> = ["ALLOW", "ALLOW", "ALLOW", "REVIEW", "BLOCK"];
  const score = Math.random();
  const decision = score > 0.7 ? "BLOCK" : score > 0.3 ? "REVIEW" : "ALLOW";

  return {
    txn_id: `txn_${Date.now()}_${index.toString().padStart(4, "0")}`,
    user_id: `user_${Math.floor(Math.random() * 10000).toString().padStart(5, "0")}`,
    amount: Math.round(Math.random() * 5000 * 100) / 100,
    currency: "USD",
    merchant: merchants[Math.floor(Math.random() * merchants.length)],
    merchant_category: categories[Math.floor(Math.random() * categories.length)],
    timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
    latitude: 25 + Math.random() * 25,
    longitude: -120 + Math.random() * 60,
    ip_address: `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
    device_id: `dev_${Math.random().toString(36).slice(2, 10)}`,
    is_international: Math.random() > 0.85,
    fraud_score: score,
    decision,
    model_version: "xgb-v2.3.1",
    features: {
      amount: Math.round(Math.random() * 5000 * 100) / 100,
      merchant_category: categories[Math.floor(Math.random() * categories.length)],
      is_international: Math.random() > 0.85,
      hour_of_day: Math.floor(Math.random() * 24),
      day_of_week: Math.floor(Math.random() * 7),
      user_avg_amount: Math.round(Math.random() * 500 * 100) / 100,
      user_tx_count_24h: Math.floor(Math.random() * 20),
      user_unique_merchants_24h: Math.floor(Math.random() * 8),
      amount_zscore: (Math.random() - 0.5) * 6,
      velocity_score: Math.random(),
      distance_from_home: Math.random() * 5000,
      time_since_last_tx: Math.random() * 86400,
      device_fingerprint_match: Math.random() > 0.2,
      ip_risk_score: Math.random(),
      billing_shipping_match: Math.random() > 0.15,
    },
    explanation: score > 0.5
      ? "High-risk transaction: unusual amount for this user, international transaction from new device, velocity spike detected."
      : "Transaction appears normal: amount within user baseline, known device and location.",
    processing_time_ms: Math.round(Math.random() * 50 + 5),
    scored_at: new Date().toISOString(),
  };
}

export function generateMockTransactions(count: number) {
  return Array.from({ length: count }, (_, i) => generateMockTransaction(i));
}
