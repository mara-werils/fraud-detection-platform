"""A/B testing API with statistical significance."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from scoring.services.ab_statistics import ABStatisticsCalculator

router = APIRouter(prefix="/api/v1/ab", tags=["ab-testing"])


@router.get("/results")
async def ab_results(request: Request) -> dict[str, Any]:
    """Get current A/B test results with statistical significance analysis.

    Returns per-group metrics, z-tests for proportions (flag/block rates),
    t-tests for continuous metrics (scores, latency), and an automatic
    winner recommendation.
    """
    ab_manager = getattr(request.app.state, "ab_manager", None)
    if ab_manager is None:
        raise HTTPException(status_code=503, detail="A/B testing not configured")

    raw_results = ab_manager.get_results()
    groups = raw_results.get("groups", {})
    group_names = list(groups.keys())

    if len(group_names) < 2:
        return {**raw_results, "statistical_analysis": None}

    # Use first group as control, second as treatment
    control_name = group_names[0]
    treatment_name = group_names[1]

    calculator = ABStatisticsCalculator(confidence_level=0.95, min_sample_size=100)
    analysis = calculator.analyze(
        experiment_name=raw_results.get("experiment", "default"),
        control_name=control_name,
        treatment_name=treatment_name,
        control_metrics=groups[control_name],
        treatment_metrics=groups[treatment_name],
    )

    return {
        **raw_results,
        "statistical_analysis": analysis.to_dict(),
    }
