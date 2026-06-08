/**
 * TypeScript type definitions for the Fraud Detection Platform SDK.
 *
 * All interfaces mirror the platform's Pydantic schemas and REST API contracts.
 */

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

export type TransactionType =
  | "purchase"
  | "transfer"
  | "withdrawal"
  | "deposit"
  | "refund";

export type Decision = "ALLOW" | "REVIEW" | "BLOCK";

export type CaseStatus =
  | "open"
  | "investigating"
  | "escalated"
  | "resolved_fraud"
  | "resolved_legitimate"
  | "closed";

export type CasePriority = "low" | "medium" | "high" | "critical";

export type AlertSeverity = "low" | "medium" | "high" | "critical";

export type WebhookFormat = "generic" | "pagerduty" | "slack";

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

export interface ClientConfig {
  /** Base URL of the fraud detection service. Defaults to http://localhost:8000 */
  baseUrl?: string;
  /** API key for authentication (sent as X-API-Key header) */
  apiKey?: string;
  /** Request timeout in milliseconds. Defaults to 30_000 */
  timeout?: number;
  /** Number of retry attempts on transient failures. Defaults to 3 */
  retries?: number;
  /** Base delay (ms) for exponential backoff between retries. Defaults to 300 */
  retryDelay?: number;
  /** Custom request interceptor called before every HTTP request */
  onRequest?: RequestInterceptor;
  /** Custom response interceptor called after every HTTP response */
  onResponse?: ResponseInterceptor;
}

export type RequestInterceptor = (
  url: string,
  init: RequestInit
) => RequestInit | Promise<RequestInit>;

export type ResponseInterceptor = (
  response: Response,
  url: string
) => void | Promise<void>;

// ---------------------------------------------------------------------------
// Transaction types
// ---------------------------------------------------------------------------

/** Input payload for scoring a single transaction. */
export interface TransactionInput {
  /** UUID of the user making the transaction */
  user_id: string;
  /** Transaction amount (non-negative) */
  amount: number;
  /** ISO 4217 currency code. Defaults to "USD" */
  currency?: string;
  /** Type of transaction */
  transaction_type: TransactionType;
  /** Optional merchant identifier */
  merchant_id?: string;
  /** Optional merchant category code */
  merchant_category?: string;
  /** Optional hashed card number */
  card_number_hash?: string;
  /** Source IP address */
  ip_address?: string;
  /** Device fingerprint */
  device_id?: string;
  /** Latitude of the transaction location */
  latitude?: number;
  /** Longitude of the transaction location */
  longitude?: number;
  /** Explicit transaction ID; auto-generated UUID if omitted */
  transaction_id?: string;
  /** ISO 8601 timestamp; defaults to current UTC time */
  timestamp?: string;
  /** Arbitrary metadata bag */
  metadata?: Record<string, unknown>;
}

/** Raw transaction as stored by the platform (returned from search). */
export interface Transaction {
  transaction_id: string;
  user_id: string;
  amount: string; // Decimal serialised as string
  currency: string;
  transaction_type: TransactionType;
  merchant_id: string | null;
  merchant_category: string | null;
  card_number_hash: string | null;
  ip_address: string | null;
  device_id: string | null;
  latitude: number | null;
  longitude: number | null;
  timestamp: string; // ISO 8601
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Feature vector
// ---------------------------------------------------------------------------

export interface FeatureVector {
  transaction_id: string;
  user_id: string;
  timestamp: string;
  // Velocity
  txn_count_1h: number;
  txn_count_24h: number;
  txn_amount_sum_1h: string;
  txn_amount_sum_24h: string;
  txn_amount_avg_24h: string;
  // Behavioural
  unique_merchants_24h: number;
  unique_countries_24h: number;
  max_amount_24h: string;
  time_since_last_txn_seconds: number;
  // Location / device
  distance_from_last_txn_km: number;
  is_new_device: boolean;
  is_new_merchant: boolean;
  // Amount
  amount_zscore: number;
  amount_to_avg_ratio: number;
}

// ---------------------------------------------------------------------------
// Scored transaction
// ---------------------------------------------------------------------------

export interface ScoredTransaction {
  transaction_id: string;
  user_id: string;
  amount: string; // Decimal serialised as string
  currency: string;
  transaction_type: TransactionType;
  timestamp: string; // ISO 8601
  fraud_score: number; // 0.0 – 1.0
  model_version: string;
  scoring_latency_ms: number;
  scored_at: string; // ISO 8601
  features: FeatureVector | null;
  is_flagged: boolean;
  flag_reason: string | null;
  /** Derived decision — not returned by the API, computed by the SDK */
  decision?: Decision;
}

// ---------------------------------------------------------------------------
// Batch scoring
// ---------------------------------------------------------------------------

export interface BatchScoreRequest {
  transactions: TransactionInput[];
}

export interface BatchScoreError {
  index: number;
  transaction_id: string;
  error: string;
}

export interface BatchScoreResponse {
  results: ScoredTransaction[];
  total: number;
  failed: number;
  latency_ms: number;
  errors: BatchScoreError[];
}

export interface BatchScoreStats {
  total: number;
  flagged: number;
  blocked: number;
  review: number;
  allowed: number;
  avg_score: number;
  max_score: number;
  min_score: number;
  avg_latency_ms: number;
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

export interface CaseNote {
  note_id: string;
  author: string;
  content: string;
  created_at: string; // ISO 8601
}

export interface CaseEvent {
  event_type: string;
  actor: string;
  details: string;
  timestamp: string; // ISO 8601
}

export interface Case {
  case_id: string;
  transaction_id: string;
  user_id: string;
  fraud_score: number;
  amount: number;
  currency: string;
  status: CaseStatus;
  priority: CasePriority;
  assigned_to: string | null;
  created_at: string; // ISO 8601
  updated_at: string; // ISO 8601
  resolved_at: string | null;
  notes: CaseNote[];
  events: CaseEvent[];
  tags: string[];
  is_fraud: boolean | null;
}

export interface CreateCaseInput {
  transaction_id: string;
  priority?: CasePriority;
  assigned_to?: string;
  notes?: string;
  tags?: string[];
}

export interface UpdateCaseInput {
  status?: CaseStatus;
  assigned_to?: string;
  notes?: string;
  actor?: string;
}

export interface CaseFilters {
  status?: CaseStatus;
  priority?: CasePriority;
  assigned_to?: string;
  user_id?: string;
  limit?: number;
  offset?: number;
}

export interface CaseStats {
  total: number;
  by_status: Record<string, number>;
  by_priority: Record<string, number>;
  avg_resolution_hours: number;
}

// ---------------------------------------------------------------------------
// Feedback
// ---------------------------------------------------------------------------

export interface FeedbackRequest {
  transaction_id: string;
  is_fraud: boolean;
  analyst?: string;
  notes?: string;
}

export interface FeedbackEntry {
  feedback_id: string;
  transaction_id: string;
  is_fraud: boolean;
  analyst: string;
  notes: string | null;
  original_score: number | null;
  original_decision: Decision | null;
  created_at: string;
}

export interface FeedbackFilters {
  analyst?: string;
  is_fraud?: boolean;
  limit?: number;
  offset?: number;
}

export interface FeedbackStats {
  total: number;
  fraud_count: number;
  legitimate_count: number;
  false_positive_rate: number;
  false_negative_rate: number;
  agreement_rate: number;
  by_analyst: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Drift detection
// ---------------------------------------------------------------------------

export interface FeatureDriftStats {
  psi: number;
  ks_statistic: number;
  ks_pvalue: number;
  has_drifted: boolean;
}

export interface DriftReport {
  status: "ok" | "warning" | "critical" | "insufficient_data";
  checked_at: string; // ISO 8601
  overall_drift_detected: boolean;
  features: Record<string, FeatureDriftStats>;
  message?: string;
}

export interface DriftHistoryResponse {
  total: number;
  reports: DriftReport[];
}

// ---------------------------------------------------------------------------
// Model info
// ---------------------------------------------------------------------------

export interface ModelInfo {
  model_version: string;
  model_type: string;
  trained_at: string | null;
  feature_count: number;
  thresholds: {
    block: number;
    review: number;
  };
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Webhooks
// ---------------------------------------------------------------------------

export interface WebhookConfig {
  url: string;
  secret: string;
  format: WebhookFormat;
  timeout_seconds?: number;
  max_retries?: number;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export interface ComponentStatus {
  name: string;
  status: "healthy" | "degraded" | "unhealthy";
  latency_ms: number | null;
  detail: string | null;
}

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  uptime_seconds: number;
  timestamp: string;
  components: ComponentStatus[];
}

// ---------------------------------------------------------------------------
// Generic pagination
// ---------------------------------------------------------------------------

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

// ---------------------------------------------------------------------------
// Transaction search
// ---------------------------------------------------------------------------

export interface TransactionFilters {
  user_id?: string;
  min_score?: number;
  max_score?: number;
  is_flagged?: boolean;
  decision?: Decision;
  start_time?: string; // ISO 8601
  end_time?: string; // ISO 8601
  limit?: number;
  offset?: number;
}

export interface TransactionSearchResponse {
  transactions: ScoredTransaction[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface TransactionStats {
  total: number;
  flagged: number;
  avg_score: number;
  max_score: number;
  unique_users: number;
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

export type RuleConditionOperator =
  | "gt"
  | "gte"
  | "lt"
  | "lte"
  | "eq"
  | "neq"
  | "in"
  | "not_in"
  | "contains"
  | "regex";

export type RuleAction = "block" | "review" | "allow" | "flag" | "alert";

export interface RuleCondition {
  field: string;
  operator: RuleConditionOperator;
  value: unknown;
}

export interface Rule {
  rule_id: string;
  name: string;
  description: string | null;
  conditions: RuleCondition[];
  action: RuleAction;
  priority: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  created_by: string | null;
  tags: string[];
  metadata: Record<string, unknown>;
}

export interface CreateRuleInput {
  name: string;
  description?: string;
  conditions: RuleCondition[];
  action: RuleAction;
  priority?: number;
  is_active?: boolean;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface UpdateRuleInput {
  name?: string;
  description?: string;
  conditions?: RuleCondition[];
  action?: RuleAction;
  priority?: number;
  is_active?: boolean;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface RuleFilters {
  is_active?: boolean;
  action?: RuleAction;
  tag?: string;
  limit?: number;
  offset?: number;
}

export interface RuleListResponse {
  rules: Rule[];
  total: number;
  limit: number;
  offset: number;
}

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

export interface DecisionResult {
  transaction_id: string;
  decision: Decision;
  fraud_score: number;
  rules_triggered: string[];
  model_version: string;
  timestamp: string;
  explanation: string | null;
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// WebSocket event types
// ---------------------------------------------------------------------------

export type WebSocketEventType =
  | "transaction.scored"
  | "transaction.flagged"
  | "case.created"
  | "case.updated"
  | "rule.triggered"
  | "alert.fired"
  | "model.updated"
  | "drift.detected";

export interface WebSocketEvent<T = unknown> {
  event_type: WebSocketEventType;
  timestamp: string;
  org_id: string;
  payload: T;
}

export interface WebSocketConfig {
  /** WebSocket URL. Defaults to ws://localhost:8000/ws/events */
  url?: string;
  /** Organization ID for event filtering */
  orgId: string;
  /** User ID for tracking (defaults to "anonymous") */
  userId?: string;
  /** Automatic reconnect on disconnect. Defaults to true */
  autoReconnect?: boolean;
  /** Maximum reconnect attempts. Defaults to 10 */
  maxReconnectAttempts?: number;
  /** Base delay (ms) between reconnect attempts. Defaults to 1000 */
  reconnectDelay?: number;
  /** Heartbeat/ping interval in ms. Defaults to 30_000 */
  pingInterval?: number;
}

export type WebSocketEventHandler<T = unknown> = (
  event: WebSocketEvent<T>
) => void;

export type WebSocketStateChangeHandler = (
  state: "connecting" | "connected" | "disconnected" | "reconnecting"
) => void;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

export interface APIErrorBody {
  detail: string | Record<string, unknown>;
  [key: string]: unknown;
}
