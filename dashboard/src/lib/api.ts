import type {
  ScoredTransaction,
  Stats,
  ModelInfo,
  AnalyticsData,
  ABTestResult,
  FeatureImportance,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

export interface TransactionFilters {
  page?: number;
  limit?: number;
  min_score?: number;
  max_score?: number;
  decision?: string;
  category?: string;
  search?: string;
  start_date?: string;
  end_date?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pages: number;
}

export async function fetchTransactions(
  filters: TransactionFilters = {}
): Promise<PaginatedResponse<ScoredTransaction>> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, String(value));
    }
  });
  const query = params.toString();
  return fetchJSON(`/api/transactions${query ? `?${query}` : ""}`);
}

export async function fetchTransaction(
  txnId: string
): Promise<ScoredTransaction> {
  return fetchJSON(`/api/transactions/${txnId}`);
}

export async function fetchStats(): Promise<Stats> {
  return fetchJSON("/api/stats");
}

export async function fetchModelInfo(): Promise<ModelInfo[]> {
  return fetchJSON("/api/models");
}

export async function fetchAnalytics(
  startDate?: string,
  endDate?: string
): Promise<AnalyticsData> {
  const params = new URLSearchParams();
  if (startDate) params.set("start_date", startDate);
  if (endDate) params.set("end_date", endDate);
  const query = params.toString();
  return fetchJSON(`/api/analytics${query ? `?${query}` : ""}`);
}

export async function fetchABTests(): Promise<ABTestResult[]> {
  return fetchJSON("/api/models/ab-tests");
}

export async function fetchFeatureImportance(): Promise<FeatureImportance[]> {
  return fetchJSON("/api/models/feature-importance");
}
