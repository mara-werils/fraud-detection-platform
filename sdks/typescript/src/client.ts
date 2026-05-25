/**
 * FraudClient — main HTTP client for the Fraud Detection Platform API.
 *
 * Features
 * --------
 * - Native-fetch transport (no axios, no extra dependencies)
 * - Automatic retry with exponential backoff on 429 / 5xx responses
 * - Request/response interceptor hooks
 * - Typed custom error class (FraudAPIError)
 * - Timeout via AbortController
 * - Decision field derived on the client for every ScoredTransaction
 */

import {
  BatchScoreResponse,
  BatchScoreStats,
  Case,
  CaseFilters,
  CaseStats,
  ClientConfig,
  CreateCaseInput,
  Decision,
  DriftHistoryResponse,
  DriftReport,
  FeedbackEntry,
  FeedbackFilters,
  FeedbackRequest,
  FeedbackStats,
  HealthStatus,
  ModelInfo,
  ScoredTransaction,
  TransactionFilters,
  TransactionInput,
  TransactionSearchResponse,
  TransactionStats,
  UpdateCaseInput,
} from "./types.js";

// ---------------------------------------------------------------------------
// Custom error
// ---------------------------------------------------------------------------

export class FraudAPIError extends Error {
  /** HTTP status code returned by the API */
  public readonly statusCode: number;
  /** Raw error body returned by the API */
  public readonly body: unknown;
  /** URL that caused the error */
  public readonly url: string;

  constructor(message: string, statusCode: number, body: unknown, url: string) {
    super(message);
    this.name = "FraudAPIError";
    this.statusCode = statusCode;
    this.body = body;
    this.url = url;
    // Maintain prototype chain for instanceof checks
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class FraudTimeoutError extends Error {
  public readonly url: string;

  constructor(url: string, timeoutMs: number) {
    super(`Request to ${url} timed out after ${timeoutMs}ms`);
    this.name = "FraudTimeoutError";
    this.url = url;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class FraudNetworkError extends Error {
  public readonly url: string;
  public readonly cause: unknown;

  constructor(url: string, cause: unknown) {
    super(`Network error while requesting ${url}: ${String(cause)}`);
    this.name = "FraudNetworkError";
    this.url = url;
    this.cause = cause;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

const DEFAULT_BASE_URL = "http://localhost:8000";
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_RETRIES = 3;
const DEFAULT_RETRY_DELAY_MS = 300;

/** HTTP status codes that should trigger a retry. */
const RETRYABLE_STATUSES = new Set([408, 429, 500, 502, 503, 504]);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Compute exponential backoff with full-jitter. */
function backoffMs(attempt: number, baseMs: number): number {
  const cap = 30_000;
  const exp = Math.min(cap, baseMs * 2 ** attempt);
  return Math.random() * exp;
}

/** Derive an SDK-level Decision label from a raw fraud_score. */
function deriveDecision(score: number): Decision {
  if (score >= 0.8) return "BLOCK";
  if (score >= 0.5) return "REVIEW";
  return "ALLOW";
}

/** Attach a `decision` field to every ScoredTransaction. */
function annotate(txn: ScoredTransaction): ScoredTransaction {
  return { ...txn, decision: deriveDecision(txn.fraud_score) };
}

/** Build a URL query string from a plain object, skipping undefined values. */
function buildQuery(params: Record<string, unknown>): string {
  const pairs: string[] = [];
  for (const [key, val] of Object.entries(params)) {
    if (val === undefined || val === null) continue;
    pairs.push(`${encodeURIComponent(key)}=${encodeURIComponent(String(val))}`);
  }
  return pairs.length ? `?${pairs.join("&")}` : "";
}

/** Generate a UUID v4. Falls back to a timestamp-based ID if crypto is unavailable. */
function uuid(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Simple fallback for environments without crypto.randomUUID
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// ---------------------------------------------------------------------------
// Main client class
// ---------------------------------------------------------------------------

export class FraudClient {
  private readonly _baseUrl: string;
  private readonly _headers: Record<string, string>;
  private readonly _timeout: number;
  private readonly _retries: number;
  private readonly _retryDelay: number;
  private readonly _onRequest: ClientConfig["onRequest"];
  private readonly _onResponse: ClientConfig["onResponse"];

  /**
   * Create a new FraudClient.
   *
   * @example
   * ```ts
   * const client = new FraudClient({
   *   baseUrl: "https://fraud.example.com",
   *   apiKey: "fdp_live_...",
   *   timeout: 10_000,
   *   retries: 3,
   * });
   * ```
   */
  constructor(config: ClientConfig = {}) {
    this._baseUrl = (config.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, "");
    this._timeout = config.timeout ?? DEFAULT_TIMEOUT_MS;
    this._retries = config.retries ?? DEFAULT_RETRIES;
    this._retryDelay = config.retryDelay ?? DEFAULT_RETRY_DELAY_MS;
    this._onRequest = config.onRequest;
    this._onResponse = config.onResponse;

    this._headers = {
      "Content-Type": "application/json",
      Accept: "application/json",
      "User-Agent": "@fraud-detection/sdk/1.0.0",
    };

    if (config.apiKey) {
      this._headers["X-API-Key"] = config.apiKey;
    }
  }

  // -------------------------------------------------------------------------
  // Scoring
  // -------------------------------------------------------------------------

  /**
   * Score a single transaction and return the full result with a derived
   * Decision label (ALLOW / REVIEW / BLOCK).
   *
   * @example
   * ```ts
   * const result = await client.score({
   *   user_id: "user-001",
   *   amount: 4500,
   *   transaction_type: "purchase",
   *   merchant_id: "merch-42",
   * });
   * console.log(result.fraud_score, result.decision);
   * ```
   */
  async score(transaction: TransactionInput): Promise<ScoredTransaction> {
    const payload: Record<string, unknown> = {
      transaction_id: transaction.transaction_id ?? uuid(),
      timestamp: transaction.timestamp ?? new Date().toISOString(),
      ...transaction,
    };

    const raw = await this._request<ScoredTransaction>("POST", "/score", {
      body: payload,
    });
    return annotate(raw);
  }

  /**
   * Score multiple transactions in a single request.
   * Transactions are processed concurrently on the server for maximum throughput.
   *
   * @example
   * ```ts
   * const response = await client.batchScore([
   *   { user_id: "u1", amount: 100, transaction_type: "purchase" },
   *   { user_id: "u2", amount: 9999, transaction_type: "transfer" },
   * ]);
   * console.log(`Processed ${response.total}, failed ${response.failed}`);
   * ```
   */
  async batchScore(
    transactions: TransactionInput[]
  ): Promise<BatchScoreResponse> {
    const enriched = transactions.map((t) => ({
      transaction_id: t.transaction_id ?? uuid(),
      timestamp: t.timestamp ?? new Date().toISOString(),
      ...t,
    }));

    const raw = await this._request<BatchScoreResponse>(
      "POST",
      "/api/v1/batch/score",
      { body: { transactions: enriched } }
    );

    return {
      ...raw,
      results: raw.results.map(annotate),
    };
  }

  /**
   * Score a batch and return only aggregate statistics (no individual results).
   * Useful for pipeline monitoring without the overhead of transferring full results.
   */
  async batchScoreStats(
    transactions: TransactionInput[]
  ): Promise<BatchScoreStats> {
    const enriched = transactions.map((t) => ({
      transaction_id: t.transaction_id ?? uuid(),
      timestamp: t.timestamp ?? new Date().toISOString(),
      ...t,
    }));

    return this._request<BatchScoreStats>(
      "POST",
      "/api/v1/batch/score/stats",
      { body: { transactions: enriched } }
    );
  }

  // -------------------------------------------------------------------------
  // Transaction search
  // -------------------------------------------------------------------------

  /**
   * Search and filter scored transactions.
   *
   * @example
   * ```ts
   * const page = await client.searchTransactions({
   *   min_score: 0.8,
   *   is_flagged: true,
   *   limit: 20,
   * });
   * console.log(`${page.total} high-risk transactions found`);
   * ```
   */
  async searchTransactions(
    filters: TransactionFilters = {}
  ): Promise<TransactionSearchResponse> {
    const raw = await this._request<TransactionSearchResponse>(
      "GET",
      "/api/v1/transactions",
      { query: filters as Record<string, unknown> }
    );

    return {
      ...raw,
      transactions: raw.transactions.map(annotate),
    };
  }

  /**
   * Get a single scored transaction by its ID.
   * Throws FraudAPIError with statusCode 404 if not found.
   */
  async getTransaction(transactionId: string): Promise<ScoredTransaction> {
    const raw = await this._request<ScoredTransaction>(
      "GET",
      `/api/v1/transactions/${encodeURIComponent(transactionId)}`
    );
    return annotate(raw);
  }

  /**
   * Get aggregate statistics for all scored transactions in the store.
   */
  async getTransactionStats(): Promise<TransactionStats> {
    return this._request<TransactionStats>(
      "GET",
      "/api/v1/transactions/stats"
    );
  }

  // -------------------------------------------------------------------------
  // Cases
  // -------------------------------------------------------------------------

  /**
   * List fraud investigation cases with optional filtering.
   *
   * @example
   * ```ts
   * const { cases, total } = await client.getCases({ status: "open", priority: "high" });
   * ```
   */
  async getCases(
    filters: CaseFilters = {}
  ): Promise<{ cases: Case[]; total: number; limit: number; offset: number }> {
    return this._request(
      "GET",
      "/api/v1/cases",
      { query: filters as Record<string, unknown> }
    );
  }

  /**
   * Get a single fraud case by ID.
   */
  async getCase(caseId: string): Promise<Case> {
    return this._request<Case>(
      "GET",
      `/api/v1/cases/${encodeURIComponent(caseId)}`
    );
  }

  /**
   * Create a new fraud investigation case for a suspicious transaction.
   *
   * @example
   * ```ts
   * const fraudCase = await client.createCase({
   *   transaction_id: "txn-abc123",
   *   priority: "high",
   *   assigned_to: "analyst-jane",
   *   notes: "Unusual cross-border transfer pattern.",
   * });
   * ```
   */
  async createCase(input: CreateCaseInput): Promise<Case> {
    return this._request<Case>("POST", "/api/v1/cases", { body: input });
  }

  /**
   * Update the status, assignment, or add a note to an existing case.
   */
  async updateCase(caseId: string, update: UpdateCaseInput): Promise<Case> {
    return this._request<Case>(
      "PATCH",
      `/api/v1/cases/${encodeURIComponent(caseId)}`,
      { body: update }
    );
  }

  /**
   * Get aggregate statistics across all cases.
   */
  async getCaseStats(): Promise<CaseStats> {
    return this._request<CaseStats>("GET", "/api/v1/cases/stats");
  }

  // -------------------------------------------------------------------------
  // Feedback
  // -------------------------------------------------------------------------

  /**
   * Submit analyst feedback on a scored transaction.
   * Feedback is used to measure model accuracy and generate training data.
   *
   * @example
   * ```ts
   * await client.submitFeedback({
   *   transaction_id: "txn-abc123",
   *   is_fraud: true,
   *   analyst: "jane.doe",
   *   notes: "Confirmed by customer call-back.",
   * });
   * ```
   */
  async submitFeedback(feedback: FeedbackRequest): Promise<FeedbackEntry> {
    return this._request<FeedbackEntry>("POST", "/api/v1/feedback", {
      body: feedback,
    });
  }

  /**
   * List feedback entries with optional filtering.
   */
  async listFeedback(
    filters: FeedbackFilters = {}
  ): Promise<{ entries: FeedbackEntry[]; total: number }> {
    return this._request(
      "GET",
      "/api/v1/feedback",
      { query: filters as Record<string, unknown> }
    );
  }

  /**
   * Get feedback statistics including false positive / negative rates.
   */
  async getFeedbackStats(): Promise<FeedbackStats> {
    return this._request<FeedbackStats>("GET", "/api/v1/feedback/stats");
  }

  /**
   * Export feedback as labelled training data for model retraining pipelines.
   */
  async exportTrainingData(): Promise<{
    total: number;
    format: string;
    data: FeedbackEntry[];
  }> {
    return this._request("GET", "/api/v1/feedback/export");
  }

  // -------------------------------------------------------------------------
  // Drift detection
  // -------------------------------------------------------------------------

  /**
   * Get the latest feature drift report.
   * Returns PSI and KS statistics for every monitored feature.
   *
   * @example
   * ```ts
   * const report = await client.getDriftReport();
   * if (report.overall_drift_detected) {
   *   console.warn("Model drift detected — consider retraining.");
   * }
   * ```
   */
  async getDriftReport(): Promise<DriftReport> {
    return this._request<DriftReport>("GET", "/api/v1/drift/report");
  }

  /**
   * Get historical drift reports.
   * @param limit Maximum number of reports to return (1–100, default 10)
   */
  async getDriftHistory(limit = 10): Promise<DriftHistoryResponse> {
    return this._request<DriftHistoryResponse>("GET", "/api/v1/drift/history", {
      query: { limit },
    });
  }

  // -------------------------------------------------------------------------
  // Model information
  // -------------------------------------------------------------------------

  /**
   * Get metadata about the currently loaded scoring model.
   *
   * @example
   * ```ts
   * const info = await client.getModelInfo();
   * console.log(`Model ${info.model_version} (${info.model_type})`);
   * ```
   */
  async getModelInfo(): Promise<ModelInfo> {
    return this._request<ModelInfo>("GET", "/model/info");
  }

  // -------------------------------------------------------------------------
  // Health
  // -------------------------------------------------------------------------

  /**
   * Liveness probe — returns quickly; confirms the process is alive.
   */
  async healthCheck(): Promise<HealthStatus> {
    return this._request<HealthStatus>("GET", "/health/ready");
  }

  /**
   * Full dependency health check including Redis, Kafka, PostgreSQL, and scorer.
   * May return degraded status if non-critical components are down.
   */
  async healthStatus(): Promise<HealthStatus> {
    return this._request<HealthStatus>("GET", "/health/status");
  }

  // -------------------------------------------------------------------------
  // Private: HTTP transport
  // -------------------------------------------------------------------------

  /**
   * Build the full URL with optional query string.
   */
  private _buildUrl(path: string, query?: Record<string, unknown>): string {
    const base = `${this._baseUrl}${path}`;
    if (!query || Object.keys(query).length === 0) return base;
    return `${base}${buildQuery(query)}`;
  }

  /**
   * Core request method with retry logic.
   */
  private async _request<T>(
    method: string,
    path: string,
    options: {
      body?: unknown;
      query?: Record<string, unknown>;
    } = {}
  ): Promise<T> {
    const url = this._buildUrl(path, options.query);

    return this._retry<T>(async () => {
      let init: RequestInit = {
        method,
        headers: { ...this._headers },
      };

      if (options.body !== undefined) {
        (init.headers as Record<string, string>)["Content-Type"] =
          "application/json";
        init.body = JSON.stringify(options.body);
      }

      // Run request interceptor if provided
      if (this._onRequest) {
        init = await this._onRequest(url, init);
      }

      // Attach timeout via AbortController
      const controller = new AbortController();
      const timer = setTimeout(
        () => controller.abort(),
        this._timeout
      );
      init.signal = controller.signal;

      let response: Response;
      try {
        response = await fetch(url, init);
      } catch (err) {
        clearTimeout(timer);
        if (
          err instanceof Error &&
          err.name === "AbortError"
        ) {
          throw new FraudTimeoutError(url, this._timeout);
        }
        throw new FraudNetworkError(url, err);
      }
      clearTimeout(timer);

      // Run response interceptor if provided
      if (this._onResponse) {
        await this._onResponse(response, url);
      }

      if (!response.ok) {
        let body: unknown;
        try {
          body = await response.json();
        } catch {
          body = await response.text().catch(() => null);
        }

        const detail =
          body != null &&
          typeof body === "object" &&
          "detail" in (body as object)
            ? String((body as { detail: unknown }).detail)
            : String(body ?? response.statusText);

        throw new FraudAPIError(
          `API error ${response.status}: ${detail}`,
          response.status,
          body,
          url
        );
      }

      // 204 No Content — return empty object
      if (response.status === 204) {
        return {} as T;
      }

      return response.json() as Promise<T>;
    });
  }

  /**
   * Execute `fn` with automatic retry on transient errors.
   * Uses exponential backoff with full jitter.
   */
  private async _retry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: unknown;

    for (let attempt = 0; attempt <= this._retries; attempt++) {
      try {
        return await fn();
      } catch (err) {
        lastError = err;

        // Abort early for non-retryable errors
        if (err instanceof FraudTimeoutError) throw err;
        if (err instanceof FraudNetworkError) {
          if (attempt >= this._retries) throw err;
        }
        if (err instanceof FraudAPIError) {
          if (!RETRYABLE_STATUSES.has(err.statusCode)) throw err;
          if (attempt >= this._retries) throw err;
        }

        const delay = backoffMs(attempt, this._retryDelay);
        await sleep(delay);
      }
    }

    throw lastError;
  }
}
