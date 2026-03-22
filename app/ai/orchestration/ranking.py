"""Provider-based ranking orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from app.ai.contracts.ranking import RankingAugmentationProvider
from app.ai.contracts.suitability import SuitabilityProvider
from app.ai.features.loaders import build_feature_summary_bundle
from app.ai.features.types import FeatureSummaryBundle
from app.ai.registry import get_ai_provider_registry
from app.engines.scoring_types import ScoreBlocker, ScoreComponent, SuitabilityResult
from app.models.crop_profile import CropProfile
from app.models.field import Field
from app.models.soil_test import SoilTest
from app.schemas.weather_history import ClimateSummary
from app.schemas.yield_prediction import YieldPredictionRange, YieldPredictionResult
from app.services.economic_service import EconomicAssessment


SoilLookup = dict[int, SoilTest | None] | Callable[[Field], SoilTest | None]
ClimateLookup = dict[int, ClimateSummary | None] | Callable[[Field], ClimateSummary | None]
EconomicLookup = dict[int, EconomicAssessment | None] | Callable[[Field], EconomicAssessment | None]
YieldLookup = dict[int, YieldPredictionResult | None] | Callable[[Field], YieldPredictionResult | None]


@dataclass(slots=True)
class _FieldRankingCandidate:
    field_obj: Field
    scoring_result: SuitabilityResult
    economic_assessment: EconomicAssessment | None
    yield_prediction: YieldPredictionResult | None
    feature_context: FeatureSummaryBundle
    economic_score: float = 0.0
    total_score: float = 0.0
    ranking_score: float = 0.0


@dataclass(slots=True)
class RankedFieldResultInternal:
    """Detailed ranked candidate used inside ranking flows."""

    rank: int
    field_id: int | str | UUID
    field_name: str
    crop_id: int | str | UUID
    total_score: float
    economic_score: float
    estimated_revenue: float | None
    estimated_cost: float | None
    estimated_profit: float | None
    ranking_score: float
    breakdown: dict[str, ScoreComponent]
    blockers: list[ScoreBlocker]
    reasons: list[str]
    economic_strengths: list[str]
    economic_weaknesses: list[str]
    economic_risks: list[str]
    climate_reasons: list[str]
    climate_warnings: list[str]
    climate_strengths: list[str]
    climate_weaknesses: list[str]
    climate_risks: list[str]
    yield_prediction: YieldPredictionResult | None
    result: SuitabilityResult
    feature_context: FeatureSummaryBundle

    @property
    def score(self) -> float:
        """Compatibility alias for existing API adapter code."""

        return self.total_score

    @property
    def agronomic_score(self) -> float:
        """Return the agronomic suitability score used by current ranking logic."""

        return self.result.agronomic_score

    @property
    def climate_score(self) -> float | None:
        """Return the normalized climate score when climate scoring was available."""

        return self.result.climate_score

    @property
    def risk_score(self) -> float | None:
        """Return the future numeric risk score when a scorer is introduced."""

        return None

    @property
    def confidence_score(self) -> float | None:
        """Return the available confidence score for downstream UI consumers."""

        confidence_scores: list[float] = []
        if self.result.confidence_score is not None:
            confidence_scores.append(self.result.confidence_score)
        if self.yield_prediction is not None and self.yield_prediction.confidence_score is not None:
            confidence_scores.append(self.yield_prediction.confidence_score)
        if not confidence_scores:
            return None
        return round(sum(confidence_scores) / len(confidence_scores), 2)

    @property
    def predicted_yield(self) -> float | None:
        """Return the predicted yield when economic analysis produced one."""

        if self.yield_prediction is None:
            return None
        return self.yield_prediction.predicted_yield_per_hectare

    @property
    def predicted_yield_range(self) -> YieldPredictionRange | None:
        """Return the predicted yield range when economic analysis produced one."""

        if self.yield_prediction is None:
            return None
        return self.yield_prediction.predicted_yield_range

    @property
    def predicted_yield_min(self) -> float | None:
        """Return the lower predicted-yield bound when available."""

        if self.yield_prediction is None:
            return None
        return self.yield_prediction.predicted_yield_min

    @property
    def predicted_yield_max(self) -> float | None:
        """Return the upper predicted-yield bound when available."""

        if self.yield_prediction is None:
            return None
        return self.yield_prediction.predicted_yield_max


@dataclass(slots=True)
class RankingResult:
    """Ordered ranking output for a crop across multiple fields."""

    crop_id: int
    ranked_fields: list[RankedFieldResultInternal] = field(default_factory=list)

    def top(self, top_n: int | None) -> list[RankedFieldResultInternal]:
        """Return the top-N ranked entries or all when no limit is provided."""

        if top_n is None:
            return self.ranked_fields
        if top_n <= 0:
            return []
        return self.ranked_fields[:top_n]


def _resolve_soil_test(field_obj: Field, soil_tests: SoilLookup) -> SoilTest | None:
    if callable(soil_tests):
        return soil_tests(field_obj)
    return soil_tests.get(field_obj.id)


def _resolve_climate_summary(
    field_obj: Field,
    climate_summaries: ClimateLookup | None,
) -> ClimateSummary | None:
    if climate_summaries is None:
        return None
    if callable(climate_summaries):
        return climate_summaries(field_obj)
    return climate_summaries.get(field_obj.id)


def _resolve_economic_assessment(
    field_obj: Field,
    economic_assessments: EconomicLookup | None,
) -> EconomicAssessment | None:
    if economic_assessments is None:
        return None
    if callable(economic_assessments):
        return economic_assessments(field_obj)
    return economic_assessments.get(field_obj.id)


def _resolve_yield_prediction(
    field_obj: Field,
    yield_predictions: YieldLookup | None,
) -> YieldPredictionResult | None:
    if yield_predictions is None:
        return None
    if callable(yield_predictions):
        return yield_predictions(field_obj)
    return yield_predictions.get(field_obj.id)


def _merge_reasons(*reason_sets: list[str]) -> list[str]:
    merged: list[str] = []
    for reason_set in reason_sets:
        for reason in reason_set:
            if reason not in merged:
                merged.append(reason)
    return merged


class RankingOrchestrator:
    """Compose suitability and ranking-augmentation providers for ranking flows."""

    def __init__(
        self,
        *,
        suitability_provider: SuitabilityProvider | None = None,
        ranking_augmentation_provider: RankingAugmentationProvider | None = None,
    ) -> None:
        registry = get_ai_provider_registry()
        self.suitability_provider = suitability_provider or registry.get_suitability_provider()
        self.ranking_augmentation_provider = (
            ranking_augmentation_provider or registry.get_ranking_augmentation_provider()
        )

    def _score_field_candidates(
        self,
        fields: list[Field],
        crop_profile: CropProfile,
        soil_tests: SoilLookup,
        climate_summaries: ClimateLookup | None = None,
        economic_assessments: EconomicLookup | None = None,
        yield_predictions: YieldLookup | None = None,
    ) -> list[_FieldRankingCandidate]:
        scored_fields: list[_FieldRankingCandidate] = []
        for field_obj in fields:
            soil_test = _resolve_soil_test(field_obj, soil_tests)
            climate_summary = _resolve_climate_summary(field_obj, climate_summaries)
            economic_assessment = _resolve_economic_assessment(field_obj, economic_assessments)
            yield_prediction = _resolve_yield_prediction(field_obj, yield_predictions)
            scoring_result = self.suitability_provider.calculate_suitability(
                field_obj,
                crop_profile,
                soil_test,
                climate_summary=climate_summary,
            )
            scored_fields.append(
                _FieldRankingCandidate(
                    field_obj=field_obj,
                    scoring_result=scoring_result,
                    economic_assessment=economic_assessment,
                    yield_prediction=yield_prediction
                    or (
                        economic_assessment.yield_prediction
                        if economic_assessment is not None
                        else None
                    ),
                    feature_context=build_feature_summary_bundle(
                        field_obj,
                        crop_profile,
                        soil_test=soil_test,
                        climate_summary=climate_summary,
                    ),
                    total_score=scoring_result.total_score,
                    ranking_score=scoring_result.total_score,
                )
            )
        return scored_fields

    @staticmethod
    def _ranked_result_from_score(
        rank: int,
        crop_profile: CropProfile,
        candidate: _FieldRankingCandidate,
    ) -> RankedFieldResultInternal:
        field_obj = candidate.field_obj
        scoring_result = candidate.scoring_result
        economic_assessment = candidate.economic_assessment
        return RankedFieldResultInternal(
            rank=rank,
            field_id=field_obj.id,
            field_name=field_obj.name,
            crop_id=crop_profile.id,
            total_score=round(candidate.total_score, 2),
            economic_score=round(candidate.economic_score, 2),
            estimated_revenue=(
                economic_assessment.estimated_revenue
                if economic_assessment is not None
                else None
            ),
            estimated_cost=(
                economic_assessment.estimated_cost
                if economic_assessment is not None
                else None
            ),
            estimated_profit=(
                economic_assessment.estimated_profit
                if economic_assessment is not None
                else None
            ),
            ranking_score=round(candidate.ranking_score, 2),
            breakdown=scoring_result.score_breakdown,
            blockers=scoring_result.blockers,
            reasons=_merge_reasons(
                scoring_result.reasons,
                scoring_result.climate_warnings,
                economic_assessment.reasons if economic_assessment is not None else [],
            ),
            economic_strengths=economic_assessment.strengths if economic_assessment is not None else [],
            economic_weaknesses=economic_assessment.weaknesses if economic_assessment is not None else [],
            economic_risks=economic_assessment.risks if economic_assessment is not None else [],
            climate_reasons=list(scoring_result.climate_reasons),
            climate_warnings=list(scoring_result.climate_warnings),
            climate_strengths=list(scoring_result.climate_strengths),
            climate_weaknesses=list(scoring_result.climate_weaknesses),
            climate_risks=list(scoring_result.climate_risks),
            yield_prediction=candidate.yield_prediction,
            result=scoring_result,
            feature_context=candidate.feature_context,
        )

    def rank_fields_for_crop(
        self,
        fields: list[Field],
        crop_profile: CropProfile,
        soil_tests: SoilLookup,
        top_n: int | None = None,
        climate_summaries: ClimateLookup | None = None,
        economic_assessments: EconomicLookup | None = None,
        yield_predictions: YieldLookup | None = None,
    ) -> RankingResult:
        """Rank multiple fields for a selected crop using configured providers."""

        scored_fields = self._score_field_candidates(
            fields,
            crop_profile,
            soil_tests,
            climate_summaries=climate_summaries,
            economic_assessments=economic_assessments,
            yield_predictions=yield_predictions,
        )
        self.ranking_augmentation_provider.apply_ranking_scores(
            scored_fields,
            economics_enabled=economic_assessments is not None,
        )
        for candidate in scored_fields:
            candidate.total_score = round(candidate.ranking_score or candidate.total_score, 2)
        scored_fields.sort(key=lambda item: item.ranking_score, reverse=True)

        if top_n is not None and top_n <= 0:
            return RankingResult(crop_id=crop_profile.id, ranked_fields=[])

        limited_fields = scored_fields if top_n is None else scored_fields[:top_n]
        ranked_fields = [
            self._ranked_result_from_score(index + 1, crop_profile, candidate)
            for index, candidate in enumerate(limited_fields)
        ]
        return RankingResult(crop_id=crop_profile.id, ranked_fields=ranked_fields)


def rank_fields_for_crop(
    fields: list[Field],
    crop_profile: CropProfile,
    soil_tests: SoilLookup,
    top_n: int | None = None,
    climate_summaries: ClimateLookup | None = None,
    economic_assessments: EconomicLookup | None = None,
    yield_predictions: YieldLookup | None = None,
) -> RankingResult:
    """Compatibility wrapper that uses the default ranking orchestrator."""

    return RankingOrchestrator().rank_fields_for_crop(
        fields,
        crop_profile,
        soil_tests,
        top_n=top_n,
        climate_summaries=climate_summaries,
        economic_assessments=economic_assessments,
        yield_predictions=yield_predictions,
    )
