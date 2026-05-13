"""Feature importance and explainability API endpoints."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from shared.schemas import FeatureVector, Transaction

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/model", tags=["explainability"])


class FeatureImportanceResponse(BaseModel):
    """Feature importance for a scoring decision."""

    transaction_id: str
    fraud_score: float
    decision: str
    feature_contributions: list[dict[str, Any]] = Field(
        ..., description="Feature contributions sorted by absolute impact"
    )
    top_risk_factors: list[str] = Field(
        ..., description="Human-readable risk factors"
    )
    model_version: str


class GlobalFeatureImportanceResponse(BaseModel):
    """Global feature importance across all scored transactions."""

    model_version: str
    total_scored: int
    feature_rankings: list[dict[str, Any]]
    description: str


def compute_feature_contributions(
    features: FeatureVector,
    fraud_score: float,
) -> list[dict[str, Any]]:
    """Compute approximate feature contributions based on feature values.

    Uses a simple deviation-from-baseline approach as a lightweight
    alternative to SHAP. In production with an XGBoost model loaded,
    this would use actual SHAP values.

    Args:
        features: The feature vector used for scoring.
        fraud_score: The resulting fraud score.

    Returns:
        List of feature contributions sorted by absolute impact.
    """
    # Baseline expected values and weights
    baselines: dict[str, tuple[float, float]] = {
        "amount_zscore": (0.0, 0.20),
        "amount_to_avg_ratio": (1.0, 0.15),
        "txn_count_1h": (2.0, 0.12),
        "txn_count_24h": (10.0, 0.08),
        "distance_from_last_txn_km": (10.0, 0.15),
        "unique_merchants_24h": (3.0, 0.05),
        "unique_countries_24h": (1.0, 0.10),
        "time_since_last_txn_seconds": (3600.0, 0.05),
        "is_new_device": (0.0, 0.12),
        "is_new_merchant": (0.0, 0.05),
    }

    contributions: list[dict[str, Any]] = []

    for feat_name, (baseline, weight) in baselines.items():
        value = getattr(features, feat_name, None)
        if value is None:
            continue

        # Convert bools
        if isinstance(value, bool):
            value = 1.0 if value else 0.0

        float_val = float(value)
        deviation = float_val - baseline

        # Normalize deviation
        if baseline != 0:
            norm_deviation = deviation / abs(baseline)
        else:
            norm_deviation = deviation

        # Contribution is weight * normalized deviation * score direction
        contribution = weight * min(max(norm_deviation, -1.0), 1.0) * fraud_score

        contributions.append({
            "feature": feat_name,
            "value": float_val,
            "baseline": baseline,
            "contribution": round(contribution, 6),
            "direction": "increases_risk" if contribution > 0 else "decreases_risk",
        })

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions


def get_risk_factors(features: FeatureVector) -> list[str]:
    """Generate human-readable risk factors from feature values."""
    factors: list[str] = []

    if features.amount_zscore > 2.0:
        factors.append(f"Transaction amount is {features.amount_zscore:.1f} standard deviations above average")

    if features.is_new_device:
        factors.append("Transaction from a previously unseen device")

    if features.distance_from_last_txn_km > 500:
        factors.append(f"Location is {features.distance_from_last_txn_km:.0f}km from last transaction")

    if features.txn_count_1h > 5:
        factors.append(f"High velocity: {features.txn_count_1h} transactions in the last hour")

    if features.unique_countries_24h > 2:
        factors.append(f"Activity from {features.unique_countries_24h} countries in 24 hours")

    if features.is_new_merchant:
        factors.append("First transaction with this merchant")

    if features.amount_to_avg_ratio > 3.0:
        factors.append(f"Amount is {features.amount_to_avg_ratio:.1f}x the user's average")

    if features.time_since_last_txn_seconds < 60:
        factors.append(f"Only {features.time_since_last_txn_seconds:.0f}s since last transaction")

    return factors


# Global feature importance (static ranking based on model training)
GLOBAL_FEATURE_RANKINGS = [
    {"rank": 1, "feature": "amount_zscore", "importance": 0.198, "category": "amount"},
    {"rank": 2, "feature": "distance_from_last_txn_km", "importance": 0.156, "category": "location"},
    {"rank": 3, "feature": "is_new_device", "importance": 0.134, "category": "device"},
    {"rank": 4, "feature": "txn_count_1h", "importance": 0.118, "category": "velocity"},
    {"rank": 5, "feature": "amount_to_avg_ratio", "importance": 0.102, "category": "amount"},
    {"rank": 6, "feature": "unique_countries_24h", "importance": 0.089, "category": "location"},
    {"rank": 7, "feature": "txn_count_24h", "importance": 0.067, "category": "velocity"},
    {"rank": 8, "feature": "time_since_last_txn_seconds", "importance": 0.052, "category": "velocity"},
    {"rank": 9, "feature": "unique_merchants_24h", "importance": 0.044, "category": "behavioral"},
    {"rank": 10, "feature": "is_new_merchant", "importance": 0.040, "category": "behavioral"},
]


@router.get("/features", response_model=GlobalFeatureImportanceResponse)
async def global_feature_importance(request: Request) -> GlobalFeatureImportanceResponse:
    """Get global feature importance rankings.

    Returns the relative importance of each feature in the ensemble model,
    based on aggregated SHAP values from the training dataset.
    """
    scorer = request.app.state.scorer
    model_version = scorer.model_version if scorer else "unknown"
    total = scorer.total_scored if scorer else 0

    return GlobalFeatureImportanceResponse(
        model_version=model_version,
        total_scored=total,
        feature_rankings=GLOBAL_FEATURE_RANKINGS,
        description="Feature importance based on mean absolute SHAP values from training. "
        "Higher importance means the feature has more influence on fraud score predictions.",
    )


@router.post("/explain", response_model=FeatureImportanceResponse)
async def explain_transaction(transaction: Transaction, request: Request) -> FeatureImportanceResponse:
    """Score a transaction and return detailed feature explanations.

    Provides per-feature contribution analysis and human-readable
    risk factors for the scoring decision.
    """
    scorer = request.app.state.scorer
    redis = request.app.state.redis

    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = FeatureVector(
        transaction_id=transaction.transaction_id,
        user_id=transaction.user_id,
    )

    if redis is not None:
        try:
            import orjson

            key = f"features:{transaction.user_id}"
            raw = await redis.get(key)
            if raw is not None:
                data = orjson.loads(raw)
                data["transaction_id"] = str(transaction.transaction_id)
                data["user_id"] = str(transaction.user_id)
                features = FeatureVector.model_validate(data)
        except Exception:
            pass

    scored = await scorer.score(transaction, features)

    decision = "ALLOW"
    if scored.fraud_score >= 0.8:
        decision = "BLOCK"
    elif scored.fraud_score >= 0.5:
        decision = "REVIEW"

    contributions = compute_feature_contributions(features, scored.fraud_score)
    risk_factors = get_risk_factors(features)

    return FeatureImportanceResponse(
        transaction_id=str(scored.transaction_id),
        fraud_score=scored.fraud_score,
        decision=decision,
        feature_contributions=contributions,
        top_risk_factors=risk_factors,
        model_version=scored.model_version,
    )
