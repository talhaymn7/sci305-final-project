"""Canonical contracts and compatibility helpers for explanation providers.

Example:
```python
from app.ai.contracts.explanation import (
    ExplanationBlocker,
    ExplanationInput,
    ExplanationOutput,
    ExplanationRankingSummary,
    ExplanationScoreComponent,
    adapt_explanation_output,
)
from app.ai.providers.rule_based.explanation import RuleBasedExplanationProvider

input_data = ExplanationInput(
    ranking_result=ExplanationRankingSummary(
        field_id=101,
        field_name="North Parcel",
        crop_id=7,
        crop_name="Corn",
        rank=1,
        total_score=84.0,
        ranking_score=84.0,
        economic_score=0.0,
        estimated_profit=None,
    ),
    score_breakdown={
        "ph_compatibility": ExplanationScoreComponent(
            key="ph_compatibility",
            label="Soil pH",
            weight=1.0,
            awarded_points=15.0,
            max_points=15.0,
            status="ideal",
            reasons=["pH is within ideal range."],
        )
    },
    blockers=[],
    reasons=["pH is within ideal range."],
    penalties=[],
)

provider = RuleBasedExplanationProvider()
output = provider.explain(input_data)
legacy_result = adapt_explanation_output(output)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from app.ai.contracts.metadata import AITraceMetadata
from app.ai.features.types import FeatureSummaryBundle
from app.engines.scoring_types import ScoreBlocker, ScoreComponent, ScorePenalty, SuitabilityResult
from app.models.crop_profile import CropProfile
from app.models.field import Field
from app.schemas.ai_metadata import AITraceMetadataRead
from app.schemas.explanation import FieldExplanation


@dataclass(slots=True)
class ExplanationRankingSummary:
    """Transport-friendly ranking context for an explanation request."""

    field_id: int | None
    field_name: str
    crop_id: int | None
    crop_name: str | None
    rank: int | None
    total_score: float
    ranking_score: float
    economic_score: float | None
    estimated_profit: float | None


@dataclass(slots=True)
class ExplanationScoreComponent:
    """Normalized score component used by explanation providers."""

    key: str
    label: str
    weight: float
    awarded_points: float
    max_points: float
    status: str
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExplanationBlocker:
    """Normalized hard blocker used by explanation providers."""

    code: str
    dimension: str
    message: str


@dataclass(slots=True)
class ExplanationPenalty:
    """Normalized score penalty used to surface weaknesses."""

    dimension: str
    points_lost: float
    message: str


@dataclass(slots=True)
class ExplanationYieldMetadata:
    """Optional yield context that future providers can reference."""

    predicted_yield: float | None = None
    yield_range_min: float | None = None
    yield_range_max: float | None = None
    confidence: float | None = None


@dataclass(slots=True)
class ExplanationRiskMetadata:
    """Optional precomputed risks that can supplement provider-generated risks."""

    risks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExplanationEconomicMetadata:
    """Optional economic strengths and weaknesses used in explanation text."""

    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    estimated_revenue: float | None = None
    estimated_cost: float | None = None
    estimated_profit: float | None = None


@dataclass(slots=True)
class ExplanationClimateMetadata:
    """Optional climate context surfaced in deterministic explanations."""

    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExplanationInput:
    """Canonical provider input for explanation generation."""

    ranking_result: ExplanationRankingSummary
    score_breakdown: dict[str, ExplanationScoreComponent]
    blockers: list[ExplanationBlocker]
    reasons: list[str]
    penalties: list[ExplanationPenalty]
    yield_metadata: ExplanationYieldMetadata | None = None
    risk_metadata: ExplanationRiskMetadata | None = None
    economic_metadata: ExplanationEconomicMetadata | None = None
    climate_metadata: ExplanationClimateMetadata | None = None
    feature_context: FeatureSummaryBundle | None = None


@dataclass(slots=True)
class ExplanationOutput:
    """Canonical provider output for explanation generation."""

    short_explanation: str
    detailed_explanation: str
    strengths: list[str]
    weaknesses: list[str]
    risks: list[str]
    confidence_note: str
    metadata: AITraceMetadata

    @property
    def provider_name(self) -> str:
        return self.metadata.provider_name

    @property
    def provider_version(self) -> str | None:
        return self.metadata.provider_version

    @property
    def generated_at(self) -> datetime:
        return self.metadata.generated_at

    @property
    def confidence(self) -> float | None:
        return self.metadata.confidence

    @property
    def debug_info(self) -> dict[str, object] | None:
        return self.metadata.debug_info


@dataclass(slots=True)
class SuitabilityExplanationRequest:
    """Inputs required to explain a single suitability result."""

    result: SuitabilityResult
    field_obj: Field
    crop: CropProfile | None = None
    rank: int = 1
    yield_metadata: ExplanationYieldMetadata | None = None
    risk_metadata: ExplanationRiskMetadata | None = None
    economic_metadata: ExplanationEconomicMetadata | None = None
    climate_metadata: ExplanationClimateMetadata | None = None
    feature_context: FeatureSummaryBundle | None = None


@dataclass(slots=True)
class RankedExplanationRequest:
    """Inputs required to explain an already-ranked candidate."""

    field_name: str
    total_score: float
    ranking_score: float
    breakdown: dict[str, ScoreComponent]
    blockers: list[ScoreBlocker]
    reasons: list[str]
    penalties: list[ScorePenalty]
    climate_reasons: list[str] = field(default_factory=list)
    climate_warnings: list[str] = field(default_factory=list)
    climate_strengths: list[str] = field(default_factory=list)
    climate_weaknesses: list[str] = field(default_factory=list)
    climate_risks: list[str] = field(default_factory=list)
    economic_strengths: list[str] = field(default_factory=list)
    economic_weaknesses: list[str] = field(default_factory=list)
    economic_risks: list[str] = field(default_factory=list)
    yield_metadata: ExplanationYieldMetadata | None = None
    risk_metadata: ExplanationRiskMetadata | None = None
    economic_metadata: ExplanationEconomicMetadata | None = None
    climate_metadata: ExplanationClimateMetadata | None = None
    field_id: int | None = None
    crop_id: int | None = None
    crop_name: str | None = None
    rank: int | None = None
    economic_score: float | None = None
    estimated_profit: float | None = None
    feature_context: FeatureSummaryBundle | None = None


class ExplanationProvider(Protocol):
    """Build canonical explanations from transport-friendly explanation inputs."""

    def explain(self, request: ExplanationInput) -> ExplanationOutput:
        """Return a canonical explanation for `request`."""


def build_explanation_input_from_suitability_request(
    request: SuitabilityExplanationRequest,
) -> ExplanationInput:
    """Build the canonical explanation input from a suitability request."""

    return ExplanationInput(
        ranking_result=ExplanationRankingSummary(
            field_id=request.result.field_id,
            field_name=request.field_obj.name,
            crop_id=request.result.crop_id,
            crop_name=request.crop.crop_name if request.crop is not None else None,
            rank=request.rank,
            total_score=request.result.total_score,
            ranking_score=request.result.total_score,
            economic_score=None,
            estimated_profit=None,
        ),
        score_breakdown=_build_score_breakdown(request.result.score_breakdown),
        blockers=_build_blockers(request.result.blockers),
        reasons=list(request.result.reasons),
        penalties=_build_penalties(request.result.penalties),
        yield_metadata=request.yield_metadata,
        risk_metadata=request.risk_metadata,
        economic_metadata=request.economic_metadata,
        climate_metadata=_merge_climate_metadata(
            request.climate_metadata,
            request.result.climate_reasons,
            request.result.climate_warnings,
            request.result.climate_strengths,
            request.result.climate_weaknesses,
            request.result.climate_risks,
        ),
        feature_context=request.feature_context,
    )


def build_explanation_input_from_ranked_request(
    request: RankedExplanationRequest,
) -> ExplanationInput:
    """Build the canonical explanation input from a ranked explanation request."""

    return ExplanationInput(
        ranking_result=ExplanationRankingSummary(
            field_id=request.field_id,
            field_name=request.field_name,
            crop_id=request.crop_id,
            crop_name=request.crop_name,
            rank=request.rank,
            total_score=request.total_score,
            ranking_score=request.ranking_score,
            economic_score=request.economic_score,
            estimated_profit=request.estimated_profit,
        ),
        score_breakdown=_build_score_breakdown(request.breakdown),
        blockers=_build_blockers(request.blockers),
        reasons=list(request.reasons),
        penalties=_build_penalties(request.penalties),
        yield_metadata=request.yield_metadata,
        risk_metadata=request.risk_metadata,
        economic_metadata=_merge_economic_metadata(
            request.economic_metadata,
            request.economic_strengths,
            request.economic_weaknesses,
            request.economic_risks,
        ),
        climate_metadata=_merge_climate_metadata(
            request.climate_metadata,
            request.climate_reasons,
            request.climate_warnings,
            request.climate_strengths,
            request.climate_weaknesses,
            request.climate_risks,
        ),
        feature_context=request.feature_context,
    )


def adapt_explanation_output(output: ExplanationOutput) -> FieldExplanation:
    """Adapt canonical explanation output to the legacy API-facing schema."""

    return FieldExplanation(
        short_explanation=output.short_explanation,
        detailed_explanation=output.detailed_explanation,
        strengths=list(output.strengths),
        weaknesses=list(output.weaknesses),
        risks=list(output.risks),
        metadata=adapt_trace_metadata(output.metadata),
    )


def _build_score_breakdown(
    breakdown: dict[str, ScoreComponent],
) -> dict[str, ExplanationScoreComponent]:
    return {
        key: ExplanationScoreComponent(
            key=component.key,
            label=component.label,
            weight=component.weight,
            awarded_points=component.awarded_points,
            max_points=component.max_points,
            status=component.status.value,
            reasons=list(component.reasons),
        )
        for key, component in breakdown.items()
    }


def _build_blockers(blockers: list[ScoreBlocker]) -> list[ExplanationBlocker]:
    return [
        ExplanationBlocker(
            code=blocker.code,
            dimension=blocker.dimension,
            message=blocker.message,
        )
        for blocker in blockers
    ]


def _build_penalties(penalties: list[ScorePenalty]) -> list[ExplanationPenalty]:
    return [
        ExplanationPenalty(
            dimension=penalty.dimension,
            points_lost=penalty.points_lost,
            message=penalty.message,
        )
        for penalty in penalties
    ]


def _merge_economic_metadata(
    metadata: ExplanationEconomicMetadata | None,
    strengths: list[str],
    weaknesses: list[str],
    risks: list[str] | None = None,
) -> ExplanationEconomicMetadata | None:
    merged_strengths = _dedupe_messages(
        [*(metadata.strengths if metadata is not None else []), *strengths]
    )
    merged_weaknesses = _dedupe_messages(
        [*(metadata.weaknesses if metadata is not None else []), *weaknesses]
    )
    merged_risks = _dedupe_messages(
        [*(metadata.risks if metadata is not None else []), *((risks or []))]
    )

    if not merged_strengths and not merged_weaknesses and not merged_risks:
        return metadata

    return ExplanationEconomicMetadata(
        strengths=merged_strengths,
        weaknesses=merged_weaknesses,
        risks=merged_risks,
        estimated_revenue=metadata.estimated_revenue if metadata is not None else None,
        estimated_cost=metadata.estimated_cost if metadata is not None else None,
        estimated_profit=metadata.estimated_profit if metadata is not None else None,
    )


def _merge_climate_metadata(
    metadata: ExplanationClimateMetadata | None,
    reasons: list[str],
    warnings: list[str],
    strengths: list[str],
    weaknesses: list[str],
    risks: list[str],
) -> ExplanationClimateMetadata | None:
    merged_reasons = _dedupe_messages(
        [*(metadata.reasons if metadata is not None else []), *reasons]
    )
    merged_warnings = _dedupe_messages(
        [*(metadata.warnings if metadata is not None else []), *warnings]
    )
    merged_strengths = _dedupe_messages(
        [*(metadata.strengths if metadata is not None else []), *strengths]
    )
    merged_weaknesses = _dedupe_messages(
        [*(metadata.weaknesses if metadata is not None else []), *weaknesses]
    )
    merged_risks = _dedupe_messages(
        [*(metadata.risks if metadata is not None else []), *risks]
    )

    if (
        not merged_reasons
        and not merged_warnings
        and not merged_strengths
        and not merged_weaknesses
        and not merged_risks
    ):
        return metadata

    return ExplanationClimateMetadata(
        reasons=merged_reasons,
        warnings=merged_warnings,
        strengths=merged_strengths,
        weaknesses=merged_weaknesses,
        risks=merged_risks,
    )


def _dedupe_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for message in messages:
        normalized = message.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


ExplanationOutputAdapter = adapt_explanation_output


def adapt_trace_metadata(metadata: AITraceMetadata) -> AITraceMetadataRead:
    """Adapt reusable contract metadata into the API-safe metadata schema."""

    normalized = metadata.normalized()
    return AITraceMetadataRead(
        provider_name=normalized.provider_name,
        provider_version=normalized.provider_version,
        generated_at=normalized.generated_at,
        confidence=normalized.confidence,
        debug_info=normalized.debug_info,
    )
