export interface FeatureVector {
  amount: number;
  merchant_category: string;
  is_international: boolean;
  hour_of_day: number;
  day_of_week: number;
  user_avg_amount: number;
  user_tx_count_24h: number;
  user_unique_merchants_24h: number;
  amount_zscore: number;
  velocity_score: number;
  distance_from_home: number;
  time_since_last_tx: number;
  device_fingerprint_match: boolean;
  ip_risk_score: number;
  billing_shipping_match: boolean;
}

export interface Transaction {
  txn_id: string;
  user_id: string;
  amount: number;
  currency: string;
  merchant: string;
  merchant_category: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  ip_address: string;
  device_id: string;
  is_international: boolean;
}

export interface ScoredTransaction extends Transaction {
  fraud_score: number;
  decision: "ALLOW" | "REVIEW" | "BLOCK";
  model_version: string;
  features: FeatureVector;
  explanation?: string;
  processing_time_ms: number;
  scored_at: string;
}

export interface ModelInfo {
  name: string;
  version: string;
  type: string;
  auc: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  last_trained: string;
  status: "active" | "shadow" | "retired";
  latency_p50_ms: number;
  latency_p99_ms: number;
}

export interface Stats {
  total_transactions: number;
  fraud_rate: number;
  avg_latency_ms: number;
  blocked_amount: number;
  total_amount: number;
  transactions_per_second: number;
  active_users: number;
  review_queue_size: number;
}

export interface AnalyticsData {
  fraud_rate_over_time: TimeSeriesPoint[];
  volume_by_hour: TimeSeriesPoint[];
  top_fraud_categories: CategoryCount[];
  amount_distribution: HistogramBin[];
  hourly_heatmap: HeatmapCell[];
}

export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
  label?: string;
}

export interface CategoryCount {
  category: string;
  count: number;
  fraud_count: number;
  fraud_rate: number;
}

export interface HistogramBin {
  range: string;
  count: number;
  min: number;
  max: number;
}

export interface HeatmapCell {
  day: number;
  hour: number;
  value: number;
}

export interface ABTestResult {
  name: string;
  control: ModelMetrics;
  challenger: ModelMetrics;
  start_date: string;
  traffic_split: number;
  status: "running" | "completed" | "stopped";
}

export interface ModelMetrics {
  model_version: string;
  auc: number;
  precision: number;
  recall: number;
  f1_score: number;
  avg_latency_ms: number;
  false_positive_rate: number;
  transactions_scored: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  shap_value: number;
}

export type ConnectionStatus = "connected" | "disconnected" | "connecting";
