"""LLM-based explanation generator for fraud scoring decisions.

Uses the Anthropic Claude API for production explanations with a
template-based fallback when the LLM is unavailable or rate-limited.
Explanations are generated asynchronously and cached in Redis.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog
from redis.asyncio import Redis

from shared.schemas import FeatureVector, Transaction

logger = structlog.get_logger(__name__)

# Redis cache TTL for explanations (1 hour)
EXPLANATION_TTL_SECONDS = 3600

# Rate limit: max requests per second to LLM
LLM_MAX_RPS = 10

# Claude model to use
CLAUDE_MODEL = "claude-sonnet-4-20250514"


def _build_prompt(
    transaction: Transaction,
    features: FeatureVector,
    scores: dict[str, float],
    decision: str,
) -> str:
    """Build the structured prompt for explanation generation.

    Args:
        transaction: The transaction being explained.
        features: Feature vector used during scoring.
        scores: Individual model scores (e.g. xgboost, gnn, rules, ensemble).
        decision: Final decision string (block / review / allow).

    Returns:
        Formatted prompt string.
    """
    return f"""You are a fraud analyst assistant. Generate a clear, concise explanation
for why the following transaction was scored as it was. The explanation should be
understandable by a non-technical fraud analyst.

## Transaction Details
- Transaction ID: {transaction.transaction_id}
- User ID: {transaction.user_id}
- Amount: ${transaction.amount} {transaction.currency}
- Type: {transaction.transaction_type.value}
- Merchant: {transaction.merchant_id or 'N/A'}
- Timestamp: {transaction.timestamp.isoformat()}
- Device ID: {transaction.device_id or 'N/A'}
- IP Address: {transaction.ip_address or 'N/A'}

## Feature Signals
- Transactions in last hour: {features.txn_count_1h}
- Transactions in last 24h: {features.txn_count_24h}
- Average amount (24h): ${features.txn_amount_avg_24h}
- Amount z-score: {features.amount_zscore:.2f}
- Amount-to-average ratio: {features.amount_to_avg_ratio:.2f}
- New device: {features.is_new_device}
- New merchant: {features.is_new_merchant}
- Distance from last transaction: {features.distance_from_last_txn_km:.1f} km
- Unique countries (24h): {features.unique_countries_24h}

## Model Scores
{chr(10).join(f'- {name}: {score:.4f}' for name, score in scores.items())}

## Decision: {decision.upper()}

Write a 2-3 sentence explanation suitable for a fraud analyst reviewing this alert.
Focus on the most suspicious signals and why they are concerning."""


def generate_template_explanation(
    transaction: Transaction,
    features: FeatureVector,
    scores: dict[str, float],
    decision: str,
) -> str:
    """Generate a template-based explanation when LLM is unavailable.

    Produces a human-readable explanation highlighting the most important
    risk signals.

    Args:
        transaction: The transaction being explained.
        features: Feature vector used during scoring.
        scores: Model scores dict.
        decision: Final decision string.

    Returns:
        Template-generated explanation string.
    """
    parts: list[str] = []

    action = "blocked" if decision == "block" else "flagged for review"
    parts.append(f"Transaction {action}")

    signals: list[str] = []

    # High amount signal
    avg = float(features.txn_amount_avg_24h)
    amount = float(transaction.amount)
    if avg > 0:
        ratio = amount / avg
        if ratio > 1.5:
            signals.append(
                f"amount (${amount:,.0f}) is {ratio:.1f}x above average (${avg:,.0f})"
            )

    # New device
    if features.is_new_device:
        signals.append("new device detected")

    # Geographic anomaly
    dist = features.distance_from_last_txn_km
    if dist > 100:
        signals.append(f"location {dist:,.0f}km from usual")

    # Velocity
    if features.txn_count_1h > 5:
        signals.append(f"{features.txn_count_1h} transactions in the last hour")

    # International
    if features.unique_countries_24h > 1:
        signals.append(
            f"activity across {features.unique_countries_24h} countries in 24h"
        )

    # New merchant
    if features.is_new_merchant:
        signals.append("new merchant")

    if signals:
        parts.append(": " + ", ".join(signals))
    else:
        ensemble_score = scores.get("ensemble", 0.0)
        parts.append(f": ensemble fraud score {ensemble_score:.2f}")

    return "".join(parts)


class LLMExplainer:
    """Async LLM explanation generator with caching and rate limiting.

    Args:
        anthropic_api_key: API key for the Anthropic Claude API.
        redis: Async Redis client for caching explanations.
        max_rps: Maximum requests per second to the LLM.
        model: Claude model identifier.
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        redis: Redis | None = None,
        max_rps: int = LLM_MAX_RPS,
        model: str = CLAUDE_MODEL,
    ) -> None:
        self._api_key = anthropic_api_key
        self._redis = redis
        self._max_rps = max_rps
        self._model = model

        # Rate limiting via semaphore + token bucket
        self._semaphore = asyncio.Semaphore(max_rps)
        self._last_request_time: float = 0.0
        self._rate_lock = asyncio.Lock()

        # Lazy client initialization
        self._client: Any | None = None
        self._client_available: bool | None = None  # None = not yet checked

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily create the Anthropic async client.

        Returns:
            An ``anthropic.AsyncAnthropic`` instance.

        Raises:
            RuntimeError: If the anthropic SDK is not installed or no API key.
        """
        if self._client is not None:
            return self._client

        if not self._api_key:
            self._client_available = False
            raise RuntimeError("No Anthropic API key configured")

        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            self._client_available = True
            return self._client
        except ImportError as exc:
            self._client_available = False
            raise RuntimeError("anthropic SDK not installed") from exc

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _acquire_rate_limit(self) -> None:
        """Wait until we are within the rate limit budget."""
        async with self._rate_lock:
            now = time.monotonic()
            min_interval = 1.0 / self._max_rps
            elapsed = now - self._last_request_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    async def _get_cached(self, txn_id: UUID) -> str | None:
        """Retrieve a cached explanation from Redis."""
        if self._redis is None:
            return None
        try:
            key = f"explanation:{txn_id}"
            raw = await self._redis.get(key)
            return raw.decode("utf-8") if raw else None
        except Exception as exc:
            logger.warning("llm_cache_read_error", error=str(exc))
            return None

    async def _set_cached(self, txn_id: UUID, explanation: str) -> None:
        """Store an explanation in Redis with TTL."""
        if self._redis is None:
            return
        try:
            key = f"explanation:{txn_id}"
            await self._redis.set(key, explanation.encode("utf-8"), ex=EXPLANATION_TTL_SECONDS)
        except Exception as exc:
            logger.warning("llm_cache_write_error", error=str(exc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def explain(
        self,
        transaction: Transaction,
        features: FeatureVector,
        scores: dict[str, float],
        decision: str,
    ) -> str:
        """Generate a human-readable explanation for a scoring decision.

        This method is designed to be called asynchronously (fire-and-forget)
        so it does NOT block the scoring response.

        1. Check Redis cache first.
        2. Try Claude API with rate limiting.
        3. Fall back to template if LLM unavailable.

        Args:
            transaction: The scored transaction.
            features: Feature vector used in scoring.
            scores: Dict of model name to score.
            decision: Final decision (block / review / allow).

        Returns:
            Human-readable explanation string.
        """
        # 1. Check cache
        cached = await self._get_cached(transaction.transaction_id)
        if cached is not None:
            logger.debug(
                "llm_explanation_cache_hit",
                txn_id=str(transaction.transaction_id),
            )
            return cached

        # 2. Try LLM
        explanation = await self._generate_llm_explanation(
            transaction, features, scores, decision
        )

        # 3. Cache the result
        await self._set_cached(transaction.transaction_id, explanation)
        return explanation

    async def _generate_llm_explanation(
        self,
        transaction: Transaction,
        features: FeatureVector,
        scores: dict[str, float],
        decision: str,
    ) -> str:
        """Attempt LLM generation, falling back to template on failure."""
        # Quick check: if we already know the client is unavailable, skip
        if self._client_available is False:
            return generate_template_explanation(
                transaction, features, scores, decision
            )

        try:
            client = self._get_client()
        except RuntimeError:
            logger.warning("llm_client_unavailable", reason="no_api_key_or_sdk")
            return generate_template_explanation(
                transaction, features, scores, decision
            )

        prompt = _build_prompt(transaction, features, scores, decision)

        try:
            await self._acquire_rate_limit()

            async with self._semaphore:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=self._model,
                        max_tokens=300,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=10.0,
                )

            explanation = response.content[0].text
            logger.info(
                "llm_explanation_generated",
                txn_id=str(transaction.transaction_id),
                model=self._model,
            )
            return explanation

        except asyncio.TimeoutError:
            logger.warning(
                "llm_explanation_timeout",
                txn_id=str(transaction.transaction_id),
            )
        except Exception as exc:
            logger.error(
                "llm_explanation_error",
                txn_id=str(transaction.transaction_id),
                error=str(exc),
            )

        # Fallback to template
        return generate_template_explanation(
            transaction, features, scores, decision
        )
