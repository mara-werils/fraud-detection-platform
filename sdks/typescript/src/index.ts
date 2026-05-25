/**
 * @fraud-detection/sdk — TypeScript SDK for the Fraud Detection Platform.
 *
 * @example
 * ```ts
 * import { FraudClient } from "@fraud-detection/sdk";
 *
 * const client = new FraudClient({
 *   baseUrl: "https://fraud.example.com",
 *   apiKey: process.env.FRAUD_API_KEY,
 * });
 *
 * const result = await client.score({
 *   user_id: "user-001",
 *   amount: 4500,
 *   transaction_type: "purchase",
 * });
 *
 * console.log(result.fraud_score, result.decision);
 * ```
 *
 * @module
 */

// Client + error classes
export {
  FraudClient,
  FraudAPIError,
  FraudTimeoutError,
  FraudNetworkError,
} from "./client.js";

// All TypeScript types
export type {
  // Config
  ClientConfig,
  RequestInterceptor,
  ResponseInterceptor,

  // Enums
  TransactionType,
  Decision,
  CaseStatus,
  CasePriority,
  AlertSeverity,
  WebhookFormat,

  // Transactions
  TransactionInput,
  Transaction,
  ScoredTransaction,
  TransactionFilters,
  TransactionSearchResponse,
  TransactionStats,

  // Feature vector
  FeatureVector,

  // Batch scoring
  BatchScoreRequest,
  BatchScoreResponse,
  BatchScoreStats,
  BatchScoreError,

  // Cases
  CaseNote,
  CaseEvent,
  Case,
  CreateCaseInput,
  UpdateCaseInput,
  CaseFilters,
  CaseStats,

  // Feedback
  FeedbackRequest,
  FeedbackEntry,
  FeedbackFilters,
  FeedbackStats,

  // Drift
  FeatureDriftStats,
  DriftReport,
  DriftHistoryResponse,

  // Model
  ModelInfo,

  // Webhooks
  WebhookConfig,

  // Health
  ComponentStatus,
  HealthStatus,

  // Generic
  PaginatedResponse,
  APIErrorBody,
} from "./types.js";
