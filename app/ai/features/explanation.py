"""Assemble canonical explanation inputs from enriched ranking results.

Example:
```python
from app.ai.features.explanation import build_explanation_input

input_data = build_explanation_input(ranked_result)
print(input_data.ranking_result.field_name)
print(input_data.feature_context.crop.crop_name if input_data.feature_context else None)
```
"""

from __future__ import annotations

from typing import Protocol

from app.ai.contracts.explanation import (
    ExplanationEconomicMetadata,
    ExplanationInput,
    RankedExplanationRequest,
    build_explanation_input_from_ranked_request,
)
from app.ai.features.types import FeatureSummaryBundle
from app.engines.scoring_types import ScoreBlocker, ScoreComponent, SuitabilityResult


class _SupportsExplanationFeatureAssembly(Protocol):
    rank: int
    field_id: int
    field_name: str
    crop_id: int
    total_score: float
    economic_score: float
    estimated_revenue: float | None
    estimated_cost: float | None
    estimated_profit: float | None
    ranking_score: float
    breakdown: dict[str, ScoreComponent]
    blockers: list[ScoreBlocker]
    reasons: list[str]
    climate_reasons: list[str]
    climate_warnings: list[str]
    climate_strengths: list[str]
    climate_weaknesses: list[str]
    climate_risks: list[str]
    economic_strengths: list[str]
    economic_weaknesses: list[str]
    economic_risks: list[str]
    result: SuitabilityResult
    feature_context: FeatureSummaryBundle | None


def build_explanation_input(
    ranked_result: _SupportsExplanationFeatureAssembly,
) -> ExplanationInput:
    """Map an enriched ranked result into the canonical explanation input."""

    feature_context = getattr(ranked_result, "feature_context", None)
    crop_name = None
    if feature_context is not None:
        crop_name = feature_context.crop.crop_name
    else:
        crop_name = getattr(ranked_result, "crop_name", None)

    strengths = list(getattr(ranked_result, "economic_strengths", []))
    weaknesses = list(getattr(ranked_result, "economic_weaknesses", []))
    risks = list(getattr(ranked_result, "economic_risks", []))
    economic_metadata = None
    if strengths or weaknesses or risks:
        economic_metadata = ExplanationEconomicMetadata(
            strengths=strengths,
            weaknesses=weaknesses,
            risks=risks,
            estimated_revenue=getattr(ranked_result, "estimated_revenue", None),
            estimated_cost=getattr(ranked_result, "estimated_cost", None),
            estimated_profit=getattr(ranked_result, "estimated_profit", None),
        )

    return build_explanation_input_from_ranked_request(
        RankedExplanationRequest(
            field_name=ranked_result.field_name,
            total_score=ranked_result.total_score,
            ranking_score=ranked_result.ranking_score,
            breakdown=ranked_result.breakdown,
            blockers=ranked_result.blockers,
            reasons=ranked_result.reasons,
            penalties=ranked_result.result.penalties,
            climate_reasons=list(getattr(ranked_result, "climate_reasons", [])),
            climate_warnings=list(getattr(ranked_result, "climate_warnings", [])),
            climate_strengths=list(getattr(ranked_result, "climate_strengths", [])),
            climate_weaknesses=list(getattr(ranked_result, "climate_weaknesses", [])),
            climate_risks=list(getattr(ranked_result, "climate_risks", [])),
            economic_strengths=strengths,
            economic_weaknesses=weaknesses,
            economic_risks=risks,
            economic_metadata=economic_metadata,
            field_id=ranked_result.field_id,
            crop_id=ranked_result.crop_id,
            crop_name=crop_name,
            rank=ranked_result.rank,
            economic_score=ranked_result.economic_score,
            estimated_profit=ranked_result.estimated_profit,
            feature_context=feature_context,
        )
    )
