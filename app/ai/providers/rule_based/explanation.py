"""Rule-based explanation provider."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from app.ai.contracts.explanation import (
    ExplanationBlocker,
    ExplanationInput,
    ExplanationOutput,
    ExplanationPenalty,
    ExplanationProvider,
    ExplanationScoreComponent,
    RankedExplanationRequest,
    SuitabilityExplanationRequest,
    adapt_explanation_output,
    build_explanation_input_from_ranked_request,
    build_explanation_input_from_suitability_request,
)
from app.ai.contracts.metadata import AITraceMetadata
from app.ai.contracts.risk import RiskScorer, RiskScoringRequest
from app.ai.providers.rule_based.risk import RuleBasedRiskScoringProvider
from app.engines.scoring_types import ScoreBlocker, ScoreComponent, ScorePenalty, ScoreStatus, SuitabilityResult
from app.models.crop_profile import CropProfile
from app.models.field import Field
from app.schemas.explanation import FieldExplanation


COMPONENT_ORDER = [
    "soil_compatibility",
    "ph_compatibility",
    "drainage_compatibility",
    "water_availability_compatibility",
    "slope_compatibility",
    "climate_compatibility",
]
COMPONENT_ORDER_INDEX = {key: index for index, key in enumerate(COMPONENT_ORDER)}


class _SupportsRankedExplanation(Protocol):
    field_name: str
    total_score: float
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
    result: SuitabilityResult


@dataclass(slots=True)
class _ExplanationContext:
    """Normalized explanation input shared by ranking and recommendation flows."""

    field_name: str
    display_score: float
    breakdown: dict[str, ExplanationScoreComponent]
    penalties: list[ExplanationPenalty]
    blockers: list[ExplanationBlocker]
    reasons: list[str]
    climate_reasons: list[str]
    climate_strengths: list[str]
    climate_weaknesses: list[str]
    climate_risks: list[str]
    economic_strengths: list[str]
    economic_weaknesses: list[str]
    precomputed_risks: list[str]


def _ordered_components(
    breakdown: dict[str, ExplanationScoreComponent],
) -> list[tuple[str, ExplanationScoreComponent]]:
    seen: set[str] = set()
    ordered: list[tuple[str, ExplanationScoreComponent]] = []

    for key in COMPONENT_ORDER:
        if key in breakdown:
            ordered.append((key, breakdown[key]))
            seen.add(key)

    for key, component in breakdown.items():
        if key not in seen:
            ordered.append((key, component))

    return ordered


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


def _as_clause(message: str) -> str:
    clause = message.strip().rstrip(".")
    if not clause:
        return clause
    return clause[:1].lower() + clause[1:]


def _join_clauses(messages: list[str]) -> str:
    clauses = [_as_clause(message) for message in messages if message.strip()]
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return f"{', '.join(clauses[:-1])}, and {clauses[-1]}"


def _as_sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.endswith((".", "!", "?")):
        return stripped
    return f"{stripped}."


def _suitability_label(total_score: float, blockers: list[ExplanationBlocker]) -> str:
    if blockers or total_score < 40:
        return "not suitable"
    if total_score >= 80:
        return "highly suitable"
    if total_score >= 60:
        return "moderately suitable"
    return "marginally suitable"


def _collect_strengths(
    breakdown: dict[str, ExplanationScoreComponent],
    climate_strengths: list[str],
    economic_strengths: list[str],
) -> list[str]:
    strengths: list[str] = []
    for _, component in _ordered_components(breakdown):
        if component.key == "climate_compatibility":
            continue
        if component.status == ScoreStatus.IDEAL.value:
            strengths.extend(component.reasons)
    strengths = [*_normalize_climate_messages(climate_strengths), *strengths]
    strengths.extend(economic_strengths)
    return _dedupe_messages(strengths)


def _collect_weaknesses(
    penalties: list[ExplanationPenalty],
    climate_weaknesses: list[str],
    economic_weaknesses: list[str],
) -> list[str]:
    ordered_penalties = sorted(
        penalties,
        key=lambda penalty: (
            -penalty.points_lost,
            COMPONENT_ORDER_INDEX.get(penalty.dimension, len(COMPONENT_ORDER_INDEX)),
        ),
    )
    weaknesses = _normalize_climate_messages(climate_weaknesses)
    weaknesses.extend(
        penalty.message
        for penalty in ordered_penalties
        if penalty.dimension != "climate_compatibility"
    )
    weaknesses.extend(economic_weaknesses)
    return _dedupe_messages(weaknesses)


def _normalize_climate_messages(messages: list[str]) -> list[str]:
    return _dedupe_messages([_normalize_climate_message(message) for message in messages])


def _normalize_climate_message(message: str) -> str:
    normalized = message.strip()
    lookup = {
        "Temperature within optimal range.": "Average temperature is within the crop's ideal range.",
        "Recent temperatures stayed within the crop's preferred range.": "Average temperature is within the crop's ideal range.",
        "Recent temperatures were cooler than the crop's preferred range.": "Average temperature is below the crop's ideal range.",
        "Recent temperatures were warmer than the crop's preferred range.": "Average temperature is above the crop's ideal range.",
        "Recent temperatures fell outside the crop's tolerable range.": "Average temperature is outside the crop's tolerable range.",
        "Rainfall over the lookback period stayed within the crop's preferred range.": "Recent rainfall is within the crop's preferred range.",
        "Rainfall over the lookback period was below the crop's preferred range.": "Recent rainfall is below the crop's preferred threshold.",
        "Rainfall over the lookback period was materially below the crop's preferred range.": "Recent rainfall is below the crop's preferred threshold.",
        "Rainfall over the lookback period was above the crop's preferred range.": "Recent rainfall is above the crop's preferred threshold.",
        "Rainfall over the lookback period was materially above the crop's preferred range.": "Recent rainfall is above the crop's preferred threshold.",
        "Rainfall insufficient.": "Recent rainfall is below the crop's preferred threshold.",
        "Recent frost exposure stayed within the crop's tolerance.": "Recent frost exposure is within the crop's tolerance.",
        "Elevated frost risk was observed in the recent climate window.": "Frost risk is elevated in the recent climate window.",
        "High frost risk detected.": "Frost risk is elevated in the recent climate window.",
        "Recent heat stress stayed within the crop's tolerance.": "Recent heat exposure is within the crop's tolerance.",
        "Elevated heat stress was observed in the recent climate window.": "Recent heat stress reduces suitability.",
        "High heat risk detected.": "Recent heat stress reduces suitability.",
        "Recent climate summary is unavailable; climate scoring used a conservative fallback.": "Recent climate summary is unavailable for deterministic scoring.",
        "Recent climate summary is unavailable for this field.": "Recent climate summary is unavailable for deterministic scoring.",
        "Climate observations were incomplete for the recent scoring window.": "Recent climate observations are incomplete for deterministic scoring.",
    }
    return lookup.get(normalized, normalized)


def _collect_additional_reasons(
    reasons: list[str],
    strengths: list[str],
    weaknesses: list[str],
    risks: list[str],
) -> list[str]:
    used = set(strengths + weaknesses + risks)
    return [reason for reason in _dedupe_messages(reasons) if reason not in used]


def _build_short_explanation(
    display_score: float,
    strengths: list[str],
    weaknesses: list[str],
    risks: list[str],
    climate_strengths: list[str],
    climate_weaknesses: list[str],
    climate_risks: list[str],
    reasons: list[str],
    blockers: list[ExplanationBlocker],
) -> str:
    label = _suitability_label(display_score, blockers)
    economic_strength = next(
        (message for message in strengths if "profit" in message.lower() or "market price" in message.lower()),
        None,
    )
    economic_weakness = next(
        (
            message
            for message in weaknesses
            if "profit" in message.lower() or "market price" in message.lower() or "cost" in message.lower()
        ),
        None,
    )
    normalized_climate_strengths = _normalize_climate_messages(climate_strengths)
    normalized_climate_weaknesses = _normalize_climate_messages(climate_weaknesses)
    normalized_climate_risks = _normalize_climate_messages(climate_risks)

    if blockers and normalized_climate_risks:
        return _as_sentence(f"This field is {label} because {_join_clauses(normalized_climate_risks[:2])}")
    if blockers and risks:
        return _as_sentence(f"This field is {label} because {_join_clauses(risks[:2])}")
    if display_score >= 80 and normalized_climate_strengths:
        return _as_sentence(f"This field ranked highly because {_join_clauses(normalized_climate_strengths[:2])}")
    if normalized_climate_weaknesses and display_score < 80:
        return _as_sentence(f"This field lost points because {_join_clauses(normalized_climate_weaknesses[:2])}")
    if normalized_climate_risks and display_score < 70:
        return _as_sentence(f"This field faces material risks because {_join_clauses(normalized_climate_risks[:2])}")
    if display_score >= 80 and economic_strength:
        return _as_sentence(economic_strength)
    if economic_weakness and display_score < 70:
        return _as_sentence(economic_weakness)

    if display_score >= 80 and strengths:
        return _as_sentence(f"This field ranked highly because {_join_clauses(strengths[:2])}")

    if display_score >= 60 and strengths:
        return _as_sentence(f"This field is {label} because {_join_clauses(strengths[:2])}")

    if weaknesses:
        return _as_sentence(f"This field lost points because {_join_clauses(weaknesses[:2])}")

    if risks:
        return _as_sentence(f"This field faces material risks because {_join_clauses(risks[:2])}")

    if reasons:
        return _as_sentence(f"This field is {label} based on the current scoring results")

    return _as_sentence(f"This field is {label}")


def _build_detailed_explanation(
    field_name: str,
    display_score: float,
    strengths: list[str],
    weaknesses: list[str],
    risks: list[str],
    climate_strengths: list[str],
    climate_weaknesses: list[str],
    climate_risks: list[str],
    additional_reasons: list[str],
    blockers: list[ExplanationBlocker],
) -> str:
    label = _suitability_label(display_score, blockers)
    sentences = [
        _as_sentence(
            f"Field '{field_name}' is {label} based on the current scoring results (score: {display_score:.1f}/100)"
        )
    ]
    climate_messages = _normalize_climate_messages(
        [*climate_strengths, *climate_weaknesses, *climate_risks]
    )

    if blockers:
        blocker_messages = _dedupe_messages([blocker.message for blocker in blockers])
        sentences.append(_as_sentence("Blockers: " + "; ".join(blocker_messages[:3])))
    elif strengths:
        sentences.append(_as_sentence("Strengths: " + "; ".join(strengths[:3])))

    if climate_messages:
        sentences.append(_as_sentence("Climate: " + "; ".join(climate_messages[:3])))

    if weaknesses:
        sentences.append(_as_sentence("Weaknesses: " + "; ".join(weaknesses[:3])))

    if blockers and strengths:
        sentences.append(_as_sentence("Strengths: " + "; ".join(strengths[:3])))
    elif not blockers and risks:
        sentences.append(_as_sentence("Risks: " + "; ".join(risks[:3])))
    elif not blockers and not weaknesses and additional_reasons:
        sentences.append(_as_sentence("Additional factors: " + "; ".join(additional_reasons[:3])))
    elif not blockers and not risks:
        sentences.append("No significant risks were identified from the current scoring inputs.")

    return " ".join(sentence for sentence in sentences[:4] if sentence)


def _build_confidence_note(context: _ExplanationContext) -> str:
    if context.blockers:
        return (
            "Confidence is high because explicit agronomic blockers were detected in the deterministic scoring inputs."
        )

    statuses = {component.status for component in context.breakdown.values()}
    if ScoreStatus.MISSING.value in statuses or ScoreStatus.LIMITED.value in statuses:
        return (
            "Confidence is moderate because the explanation reflects deterministic scoring inputs with incomplete "
            "or limiting factors."
        )

    return (
        "Confidence is moderate because the explanation is derived from deterministic scoring inputs rather than "
        "observed field outcomes."
    )


def _build_confidence_value(context: _ExplanationContext) -> float:
    if context.blockers:
        return 0.9

    statuses = {component.status for component in context.breakdown.values()}
    if ScoreStatus.MISSING.value in statuses or ScoreStatus.LIMITED.value in statuses:
        return 0.72

    return 0.78


def _to_score_component(component: ExplanationScoreComponent) -> ScoreComponent:
    return ScoreComponent(
        key=component.key,
        label=component.label,
        weight=component.weight,
        awarded_points=component.awarded_points,
        max_points=component.max_points,
        status=ScoreStatus(component.status),
        reasons=list(component.reasons),
    )


def _to_score_blocker(blocker: ExplanationBlocker) -> ScoreBlocker:
    return ScoreBlocker(
        code=blocker.code,
        dimension=blocker.dimension,
        message=blocker.message,
    )


class RuleBasedExplanationProvider(ExplanationProvider):
    """Default explanation provider built from deterministic scoring outputs."""

    def __init__(self, risk_provider: RiskScorer | None = None) -> None:
        self.risk_provider = risk_provider or RuleBasedRiskScoringProvider()

    def explain(self, request: ExplanationInput) -> ExplanationOutput:
        """Build a canonical explanation from a transport-friendly explanation input."""

        context = _ExplanationContext(
            field_name=request.ranking_result.field_name,
            display_score=request.ranking_result.ranking_score,
            breakdown=request.score_breakdown,
            penalties=request.penalties,
            blockers=request.blockers,
            reasons=request.reasons,
            climate_reasons=(
                _normalize_climate_messages(request.climate_metadata.reasons)
                if request.climate_metadata is not None
                else []
            ),
            climate_strengths=(
                _normalize_climate_messages(request.climate_metadata.strengths)
                if request.climate_metadata is not None
                else []
            ),
            climate_weaknesses=(
                _normalize_climate_messages(request.climate_metadata.weaknesses)
                if request.climate_metadata is not None
                else []
            ),
            climate_risks=(
                _normalize_climate_messages(
                    [
                        *request.climate_metadata.warnings,
                        *request.climate_metadata.risks,
                    ]
                )
                if request.climate_metadata is not None
                else []
            ),
            economic_strengths=(
                list(request.economic_metadata.strengths)
                if request.economic_metadata is not None
                else []
            ),
            economic_weaknesses=(
                list(request.economic_metadata.weaknesses)
                if request.economic_metadata is not None
                else []
            ),
            precomputed_risks=(
                [
                    *(
                        _normalize_climate_messages(
                            [
                                *request.climate_metadata.warnings,
                                *request.climate_metadata.risks,
                            ]
                        )
                        if request.climate_metadata is not None
                        else []
                    ),
                    *(list(request.risk_metadata.risks) if request.risk_metadata is not None else []),
                    *(
                        list(request.economic_metadata.risks)
                        if request.economic_metadata is not None
                        else []
                    ),
                ]
            ),
        )

        strengths = _collect_strengths(
            context.breakdown,
            context.climate_strengths,
            context.economic_strengths,
        )
        weaknesses = _collect_weaknesses(
            context.penalties,
            context.climate_weaknesses,
            context.economic_weaknesses,
        )
        risk_assessment = self.risk_provider.score(
            RiskScoringRequest(
                breakdown={
                    key: _to_score_component(component)
                    for key, component in context.breakdown.items()
                },
                blockers=[_to_score_blocker(blocker) for blocker in context.blockers],
            )
        )
        risks = _dedupe_messages(
            [
                *context.precomputed_risks,
                *risk_assessment.risks,
            ]
        )
        additional_reasons = _collect_additional_reasons(
            context.reasons,
            strengths,
            weaknesses,
            risks,
        )

        return ExplanationOutput(
            short_explanation=_build_short_explanation(
                context.display_score,
                strengths,
                weaknesses,
                risks,
                context.climate_strengths,
                context.climate_weaknesses,
                context.climate_risks,
                context.reasons,
                context.blockers,
            ),
            detailed_explanation=_build_detailed_explanation(
                context.field_name,
                context.display_score,
                strengths,
                weaknesses,
                risks,
                context.climate_strengths,
                context.climate_weaknesses,
                context.climate_risks,
                additional_reasons,
                context.blockers,
            ),
            strengths=strengths,
            weaknesses=weaknesses,
            risks=risks,
            confidence_note=_build_confidence_note(context),
            metadata=AITraceMetadata(
                provider_name="rule_based",
                provider_version="v1",
                generated_at=datetime.now(timezone.utc),
                confidence=_build_confidence_value(context),
                debug_info={
                    "blocker_count": len(context.blockers),
                    "component_count": len(context.breakdown),
                    "climate_reason_count": len(context.climate_reasons),
                    "risk_count": len(risks),
                    "risk_provider_metadata": {
                        "provider_name": risk_assessment.provider_name,
                        "provider_version": risk_assessment.provider_version,
                        "generated_at": risk_assessment.generated_at.isoformat(),
                        "confidence": risk_assessment.confidence,
                        "debug_info": risk_assessment.debug_info,
                    },
                },
            ),
        )

    def explain_ranked_result(self, request: RankedExplanationRequest) -> FieldExplanation:
        """Build a structured explanation from a typed ranked explanation request."""

        return adapt_explanation_output(self.explain(build_explanation_input_from_ranked_request(request)))

    def explain_suitability(self, request: SuitabilityExplanationRequest) -> FieldExplanation:
        """Build a structured explanation from a typed suitability explanation request."""

        return adapt_explanation_output(self.explain(build_explanation_input_from_suitability_request(request)))

    def build_ranked_field_explanation(self, ranked_result: _SupportsRankedExplanation) -> FieldExplanation:
        """Compatibility wrapper for legacy ranked-result inputs."""

        return self.explain_ranked_result(
            RankedExplanationRequest(
                field_name=ranked_result.field_name,
                total_score=ranked_result.total_score,
                ranking_score=getattr(ranked_result, "ranking_score", ranked_result.total_score),
                breakdown=ranked_result.breakdown,
                blockers=ranked_result.blockers,
                reasons=ranked_result.reasons,
                penalties=ranked_result.result.penalties,
                climate_reasons=getattr(ranked_result, "climate_reasons", []),
                climate_warnings=getattr(ranked_result, "climate_warnings", []),
                climate_strengths=getattr(ranked_result, "climate_strengths", []),
                climate_weaknesses=getattr(ranked_result, "climate_weaknesses", []),
                climate_risks=getattr(ranked_result, "climate_risks", []),
                economic_strengths=getattr(ranked_result, "economic_strengths", []),
                economic_weaknesses=getattr(ranked_result, "economic_weaknesses", []),
                field_id=getattr(ranked_result, "field_id", None),
                crop_id=getattr(ranked_result, "crop_id", None),
                rank=getattr(ranked_result, "rank", None),
                economic_score=getattr(ranked_result, "economic_score", None),
                estimated_profit=getattr(ranked_result, "estimated_profit", None),
            )
        )

    def build_suitability_explanation(
        self,
        result: SuitabilityResult,
        field_obj: Field,
        rank: int = 1,
    ) -> FieldExplanation:
        """Compatibility wrapper for legacy suitability explanation inputs."""

        return self.explain_suitability(
            SuitabilityExplanationRequest(
                result=result,
                field_obj=field_obj,
                rank=rank,
            )
        )

    def generate_explanation(
        self,
        result: SuitabilityResult,
        field_obj: Field,
        crop: CropProfile,
    ) -> str:
        """Return the legacy explanation string used by existing API responses."""

        _ = crop
        explanation = self.explain(
            build_explanation_input_from_suitability_request(
                SuitabilityExplanationRequest(
                    result=result,
                    field_obj=field_obj,
                    crop=crop,
                )
            )
        )
        return explanation.detailed_explanation


def build_ranked_field_explanation(ranked_result: _SupportsRankedExplanation) -> FieldExplanation:
    """Compatibility wrapper that uses the default explanation provider."""

    return RuleBasedExplanationProvider().build_ranked_field_explanation(ranked_result)


def build_suitability_explanation(
    result: SuitabilityResult,
    field_obj: Field,
    rank: int = 1,
) -> FieldExplanation:
    """Compatibility wrapper that uses the default explanation provider."""

    return RuleBasedExplanationProvider().build_suitability_explanation(result, field_obj, rank=rank)


def generate_explanation(
    result: SuitabilityResult,
    field_obj: Field,
    crop: CropProfile,
) -> str:
    """Compatibility wrapper that uses the default explanation provider."""

    return RuleBasedExplanationProvider().generate_explanation(result, field_obj, crop)
