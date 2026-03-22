"""Rule-based suitability scoring provider."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from app.ai.contracts.suitability import SuitabilityProvider
from app.ai.providers.rule_based.climate import assess_climate_compatibility
from app.config import settings
from app.engines.scoring_config import RangeBand, SalinityBand, SuitabilityScoringConfig, load_scoring_config
from app.engines.scoring_types import (
    ScoreBlocker,
    ScoreComponent,
    ScorePenalty,
    ScoreStatus,
    SuitabilityResult,
)
from app.models.crop_profile import CropProfile
from app.models.field import Field
from app.models.soil_test import SoilTest
from app.schemas.weather_history import ClimateSummary


DRAINAGE_LEVELS = {"poor": 1, "moderate": 2, "good": 3, "excellent": 4}


@dataclass(frozen=True, slots=True)
class _FactorScore:
    """Internal normalized subscore used to build dimension scores."""

    ratio: float
    status: ScoreStatus
    reasons: list[str]


def _normalized_value(value: str | Enum | None) -> str | None:
    if isinstance(value, Enum):
        return value.value
    return value


def _round_points(value: float) -> float:
    return round(value, 2)


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _safe_soil_test_id(soil_test: SoilTest | None) -> int | str | UUID | None:
    if soil_test is None:
        return None
    soil_test_id = getattr(soil_test, "id", None)
    if isinstance(soil_test_id, (int, str, UUID)):
        return soil_test_id
    return None


def _dedupe_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for message in messages:
        if message not in seen:
            seen.add(message)
            ordered.append(message)
    return ordered


def _penalty_messages(component: ScoreComponent) -> list[str]:
    negative_markers = (
        "outside",
        "below",
        "insufficient",
        "risk",
        "exceeds",
        "limits",
        "lack",
        "unavailable",
        "missing",
        "not configured",
    )
    negative_reasons: list[str] = []
    for reason in component.reasons:
        normalized = reason.lower()
        if any(marker in normalized for marker in negative_markers):
            negative_reasons.append(reason)
    if negative_reasons:
        return _dedupe_messages(negative_reasons)
    if component.reasons:
        return [component.reasons[-1]]
    return [f"{component.label} reduced the score."]


def _status_from_ratio(ratio: float) -> ScoreStatus:
    if ratio >= 0.95:
        return ScoreStatus.IDEAL
    if ratio >= 0.6:
        return ScoreStatus.ACCEPTABLE
    return ScoreStatus.LIMITED


def _factor_from_range(
    value: float,
    band: RangeBand,
    ideal_message: str,
    acceptable_message: str,
    outside_message: str,
) -> _FactorScore:
    if band.ideal_min <= value <= band.ideal_max:
        return _FactorScore(1.0, ScoreStatus.IDEAL, [ideal_message])
    if band.acceptable_min <= value <= band.acceptable_max:
        if value < band.ideal_min:
            span = band.ideal_min - band.acceptable_min
            ratio = (value - band.acceptable_min) / span if span > 0 else 1.0
        else:
            span = band.acceptable_max - band.ideal_max
            ratio = (band.acceptable_max - value) / span if span > 0 else 1.0
        return _FactorScore(_clamp_ratio(ratio), ScoreStatus.ACCEPTABLE, [acceptable_message])
    return _FactorScore(0.0, ScoreStatus.LIMITED, [outside_message])


def _factor_from_upper_bound(
    value: float,
    band: SalinityBand,
    ideal_message: str,
    acceptable_message: str,
    outside_message: str,
) -> _FactorScore:
    if value <= band.ideal_max:
        return _FactorScore(1.0, ScoreStatus.IDEAL, [ideal_message])
    if value <= band.acceptable_max:
        span = band.acceptable_max - band.ideal_max
        ratio = (band.acceptable_max - value) / span if span > 0 else 1.0
        return _FactorScore(_clamp_ratio(ratio), ScoreStatus.ACCEPTABLE, [acceptable_message])
    return _FactorScore(0.0, ScoreStatus.LIMITED, [outside_message])


def _score_organic_matter(
    organic_matter_percent: float,
    preference: str | Enum | None,
    config: SuitabilityScoringConfig,
) -> _FactorScore:
    preference_key = _normalized_value(preference)
    if not preference_key:
        return _FactorScore(
            1.0,
            ScoreStatus.UNCONSTRAINED,
            ["Crop has no organic matter preference configured."],
        )

    band = config.thresholds.organic_matter_bands[preference_key]
    return _factor_from_range(
        organic_matter_percent,
        band,
        "Organic matter is within the crop preference range.",
        "Organic matter is acceptable for this crop.",
        "Organic matter is outside the crop preference range.",
    )


def _score_rooting_depth(
    soil_depth_cm: float | None,
    rooting_depth_cm: float | None,
    config: SuitabilityScoringConfig,
) -> _FactorScore:
    if rooting_depth_cm is None:
        return _FactorScore(
            1.0,
            ScoreStatus.UNCONSTRAINED,
            ["Crop has no rooting depth requirement configured."],
        )
    if soil_depth_cm is None:
        return _FactorScore(
            0.5,
            ScoreStatus.MISSING,
            ["Soil depth was not measured; awarded a conservative partial score."],
        )
    if soil_depth_cm >= rooting_depth_cm:
        return _FactorScore(
            1.0,
            ScoreStatus.IDEAL,
            ["Soil depth meets the crop rooting requirement."],
        )

    minimum_ratio = config.thresholds.rooting_depth_partial_ratio
    actual_ratio = soil_depth_cm / rooting_depth_cm if rooting_depth_cm > 0 else 1.0
    if actual_ratio < minimum_ratio:
        return _FactorScore(
            0.0,
            ScoreStatus.LIMITED,
            ["Soil depth is below the crop rooting requirement."],
        )

    scaled_ratio = (actual_ratio - minimum_ratio) / (1 - minimum_ratio)
    return _FactorScore(
        _clamp_ratio(scaled_ratio),
        ScoreStatus.ACCEPTABLE,
        ["Soil depth is below the crop rooting requirement."],
    )


def _score_salinity(
    ec: float | None,
    salinity_tolerance: str | Enum | None,
    config: SuitabilityScoringConfig,
) -> _FactorScore:
    tolerance_key = _normalized_value(salinity_tolerance)
    if not tolerance_key:
        return _FactorScore(
            1.0,
            ScoreStatus.UNCONSTRAINED,
            ["Crop has no salinity tolerance constraint configured."],
        )
    if ec is None:
        return _FactorScore(
            0.5,
            ScoreStatus.MISSING,
            ["Soil salinity was not measured; awarded a conservative partial score."],
        )

    band = config.thresholds.ec_bands[tolerance_key]
    return _factor_from_upper_bound(
        ec,
        band,
        "Soil salinity is acceptable for this crop.",
        "Soil salinity is marginal but still acceptable for this crop.",
        "Soil salinity exceeds the crop tolerance.",
    )


def score_soil_compatibility(
    crop: CropProfile,
    soil_test: SoilTest | None,
    config: SuitabilityScoringConfig,
) -> ScoreComponent:
    """Score the aggregate soil compatibility dimension."""

    key = "soil_compatibility"
    label = "Soil compatibility"
    weight = config.soil_compatibility
    if soil_test is None:
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=0.0,
            max_points=weight,
            status=ScoreStatus.BLOCKED,
            reasons=["No soil test available for suitability scoring."],
        )

    organic_matter = _score_organic_matter(
        soil_test.organic_matter_percent,
        crop.organic_matter_preference,
        config,
    )
    rooting_depth = _score_rooting_depth(
        soil_test.depth_cm,
        crop.rooting_depth_cm,
        config,
    )
    salinity = _score_salinity(soil_test.ec, crop.salinity_tolerance, config)

    weighted_ratio = (
        organic_matter.ratio * config.soil_subweights.organic_matter
        + rooting_depth.ratio * config.soil_subweights.rooting_depth
        + salinity.ratio * config.soil_subweights.salinity
    )
    reasons = _dedupe_messages(
        [
            *organic_matter.reasons,
            *rooting_depth.reasons,
            *salinity.reasons,
        ]
    )
    return ScoreComponent(
        key=key,
        label=label,
        weight=weight,
        awarded_points=_round_points(weight * weighted_ratio),
        max_points=weight,
        status=_status_from_ratio(weighted_ratio),
        reasons=reasons,
    )


def score_ph_compatibility(
    crop: CropProfile,
    soil_test: SoilTest | None,
    config: SuitabilityScoringConfig,
) -> ScoreComponent:
    """Score pH compatibility against ideal and tolerable crop bounds."""

    key = "ph_compatibility"
    label = "pH compatibility"
    weight = config.ph_compatibility
    if soil_test is None:
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=0.0,
            max_points=weight,
            status=ScoreStatus.BLOCKED,
            reasons=["No soil test available for suitability scoring."],
        )

    ph_value = soil_test.ph
    if crop.ideal_ph_min <= ph_value <= crop.ideal_ph_max:
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=weight,
            max_points=weight,
            status=ScoreStatus.IDEAL,
            reasons=["pH is within ideal range."],
        )

    if crop.tolerable_ph_min <= ph_value <= crop.tolerable_ph_max:
        if ph_value < crop.ideal_ph_min:
            span = crop.ideal_ph_min - crop.tolerable_ph_min
            ratio = (ph_value - crop.tolerable_ph_min) / span if span > 0 else 1.0
        else:
            span = crop.tolerable_ph_max - crop.ideal_ph_max
            ratio = (crop.tolerable_ph_max - ph_value) / span if span > 0 else 1.0
        ratio = _clamp_ratio(ratio)
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=_round_points(weight * ratio),
            max_points=weight,
            status=ScoreStatus.ACCEPTABLE,
            reasons=["pH is within tolerable range but outside ideal range."],
        )

    ph_gap = max(crop.tolerable_ph_min - ph_value, ph_value - crop.tolerable_ph_max)
    status = ScoreStatus.BLOCKED if ph_gap >= config.thresholds.ph_blocker_delta else ScoreStatus.LIMITED
    reasons = ["pH is outside the tolerable range."]
    if status is ScoreStatus.BLOCKED:
        reasons.append("Soil pH is far outside the crop tolerable range.")
    return ScoreComponent(
        key=key,
        label=label,
        weight=weight,
        awarded_points=0.0,
        max_points=weight,
        status=status,
        reasons=reasons,
    )


def score_drainage_compatibility(
    field_obj: Field,
    crop: CropProfile,
    config: SuitabilityScoringConfig,
) -> ScoreComponent:
    """Score field drainage against crop drainage requirements."""

    key = "drainage_compatibility"
    label = "Drainage compatibility"
    weight = config.drainage_compatibility
    field_level = DRAINAGE_LEVELS.get((_normalized_value(field_obj.drainage_quality) or "").lower(), 2)
    crop_level = DRAINAGE_LEVELS.get((_normalized_value(crop.drainage_requirement) or "").lower(), 2)
    if field_level >= crop_level:
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=weight,
            max_points=weight,
            status=ScoreStatus.IDEAL,
            reasons=["Field drainage meets the crop requirement."],
        )

    ratio = field_level / crop_level if crop_level > 0 else 1.0
    return ScoreComponent(
        key=key,
        label=label,
        weight=weight,
        awarded_points=_round_points(weight * ratio),
        max_points=weight,
        status=ScoreStatus.ACCEPTABLE if ratio >= 0.6 else ScoreStatus.LIMITED,
        reasons=["Drainage is below crop requirement."],
    )


def score_water_availability(
    field_obj: Field,
    crop: CropProfile,
    config: SuitabilityScoringConfig,
) -> ScoreComponent:
    """Score irrigation availability against crop water demand."""

    key = "water_availability_compatibility"
    label = "Water availability compatibility"
    weight = config.water_availability_compatibility
    water_requirement = _normalized_value(crop.water_requirement_level) or "medium"

    if field_obj.irrigation_available:
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=weight,
            max_points=weight,
            status=ScoreStatus.IDEAL,
            reasons=["Field has irrigation available."],
        )

    if water_requirement == "low":
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=weight,
            max_points=weight,
            status=ScoreStatus.IDEAL,
            reasons=["Crop has low water demand, so lack of irrigation is acceptable."],
        )

    if water_requirement == "medium":
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=_round_points(weight * 0.5),
            max_points=weight,
            status=ScoreStatus.ACCEPTABLE,
            reasons=["Field has no irrigation, which limits a medium water-demand crop."],
        )

    return ScoreComponent(
        key=key,
        label=label,
        weight=weight,
        awarded_points=0.0,
        max_points=weight,
        status=ScoreStatus.BLOCKED,
        reasons=["No irrigation available for a high water-demand crop."],
    )


def score_slope_compatibility(
    field_obj: Field,
    crop: CropProfile,
    config: SuitabilityScoringConfig,
) -> ScoreComponent:
    """Score field slope against crop slope tolerance."""

    key = "slope_compatibility"
    label = "Slope compatibility"
    weight = config.slope_compatibility
    if crop.slope_tolerance is None:
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=weight,
            max_points=weight,
            status=ScoreStatus.UNCONSTRAINED,
            reasons=["Crop has no slope tolerance configured."],
        )

    if field_obj.slope_percent <= crop.slope_tolerance:
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=weight,
            max_points=weight,
            status=ScoreStatus.IDEAL,
            reasons=["Field slope is within the crop tolerance."],
        )

    zero_threshold = crop.slope_tolerance * config.thresholds.slope_zero_ratio
    if zero_threshold <= crop.slope_tolerance or field_obj.slope_percent >= zero_threshold:
        return ScoreComponent(
            key=key,
            label=label,
            weight=weight,
            awarded_points=0.0,
            max_points=weight,
            status=ScoreStatus.LIMITED,
            reasons=["Field slope exceeds the crop tolerance."],
        )

    ratio = (zero_threshold - field_obj.slope_percent) / (zero_threshold - crop.slope_tolerance)
    ratio = _clamp_ratio(ratio)
    return ScoreComponent(
        key=key,
        label=label,
        weight=weight,
        awarded_points=_round_points(weight * ratio),
        max_points=weight,
        status=ScoreStatus.ACCEPTABLE,
        reasons=["Field slope exceeds the crop tolerance."],
    )


def evaluate_blockers(
    field_obj: Field,
    crop: CropProfile,
    soil_test: SoilTest | None,
    config: SuitabilityScoringConfig,
) -> list[ScoreBlocker]:
    """Return hard blockers that force the final score to zero."""

    blockers: list[ScoreBlocker] = []

    if soil_test is None:
        blockers.append(
            ScoreBlocker(
                code="missing_soil_test",
                dimension="soil_compatibility",
                message="No soil test available for suitability scoring.",
            )
        )

    water_requirement = _normalized_value(crop.water_requirement_level) or "medium"
    if not field_obj.irrigation_available and water_requirement == "high":
        blockers.append(
            ScoreBlocker(
                code="no_irrigation_high_water_crop",
                dimension="water_availability_compatibility",
                message="No irrigation available for a high water-demand crop.",
            )
        )

    if soil_test is not None:
        ph_value = soil_test.ph
        if ph_value < crop.tolerable_ph_min:
            ph_gap = crop.tolerable_ph_min - ph_value
        elif ph_value > crop.tolerable_ph_max:
            ph_gap = ph_value - crop.tolerable_ph_max
        else:
            ph_gap = 0.0

        if ph_gap >= config.thresholds.ph_blocker_delta:
            blockers.append(
                ScoreBlocker(
                    code="ph_far_outside_tolerable_range",
                    dimension="ph_compatibility",
                    message="Soil pH is far outside the crop tolerable range.",
                )
            )

    return blockers


def build_penalties(
    score_breakdown: dict[str, ScoreComponent],
    blockers: list[ScoreBlocker],
) -> list[ScorePenalty]:
    """Build informational penalties from lost points in non-blocked dimensions."""

    blocked_dimensions = {blocker.dimension for blocker in blockers}
    penalties: list[ScorePenalty] = []
    for key, component in score_breakdown.items():
        if key in blocked_dimensions or component.status is ScoreStatus.BLOCKED:
            continue
        points_lost = _round_points(component.max_points - component.awarded_points)
        if points_lost <= 0:
            continue
        for message in _penalty_messages(component):
            penalties.append(
                ScorePenalty(
                    dimension=key,
                    points_lost=points_lost,
                    message=message,
                )
            )
    return penalties


def normalize_total_score(raw_total: float, max_points: float) -> float:
    """Normalize raw points to a 0-100 suitability score."""

    if max_points <= 0:
        return 0.0
    return _round_points(max(0.0, min((raw_total / max_points) * 100, 100.0)))


def _legacy_weighted_score(component: ScoreComponent, requested_weight: float) -> float:
    if component.max_points <= 0:
        return 0.0
    return _round_points((component.awarded_points / component.max_points) * requested_weight)


class SuitabilityScorer:
    """Clean orchestration layer for the rule-based suitability engine."""

    def __init__(self, config: SuitabilityScoringConfig | None = None) -> None:
        self.config = config or load_scoring_config()

    def score_field(
        self,
        field_obj: Field,
        crop: CropProfile,
        soil_test: SoilTest | None,
        climate_summary: ClimateSummary | None = None,
    ) -> SuitabilityResult:
        """Score a single field for a selected crop."""

        climate_assessment = assess_climate_compatibility(crop, climate_summary, self.config)
        score_breakdown = {
            "soil_compatibility": score_soil_compatibility(crop, soil_test, self.config),
            "ph_compatibility": score_ph_compatibility(crop, soil_test, self.config),
            "drainage_compatibility": score_drainage_compatibility(field_obj, crop, self.config),
            "water_availability_compatibility": score_water_availability(field_obj, crop, self.config),
            "slope_compatibility": score_slope_compatibility(field_obj, crop, self.config),
            "climate_compatibility": climate_assessment.component,
        }

        blockers = evaluate_blockers(field_obj, crop, soil_test, self.config)
        penalties = build_penalties(score_breakdown, blockers)

        agronomic_keys = [key for key in score_breakdown if key != "climate_compatibility"]
        agronomic_raw_total = sum(score_breakdown[key].awarded_points for key in agronomic_keys)
        agronomic_max_points = sum(score_breakdown[key].max_points for key in agronomic_keys)
        agronomic_score = normalize_total_score(agronomic_raw_total, agronomic_max_points)

        climate_score = climate_assessment.climate_score
        if climate_score is None:
            total_score = agronomic_score
        else:
            total_weight = settings.AGRONOMIC_SCORE_WEIGHT + settings.CLIMATE_SCORE_WEIGHT
            total_score = _round_points(
                (
                    (agronomic_score * settings.AGRONOMIC_SCORE_WEIGHT)
                    + (climate_score * settings.CLIMATE_SCORE_WEIGHT)
                )
                / total_weight
            )
        if blockers:
            total_score = 0.0

        agronomic_confidence = 0.82 if soil_test is not None else 0.45
        if climate_assessment.confidence_score is None:
            confidence_score = agronomic_confidence
        elif climate_score is None:
            confidence_score = round(
                min(agronomic_confidence, climate_assessment.confidence_score),
                2,
            )
        else:
            confidence_score = round(
                (agronomic_confidence + climate_assessment.confidence_score) / 2.0,
                2,
            )
        if blockers:
            confidence_score = max(confidence_score, 0.88)

        reasons = _dedupe_messages(
            [
                reason
                for component in score_breakdown.values()
                for reason in component.reasons
            ]
            + climate_assessment.reasons
            + [blocker.message for blocker in blockers]
        )

        return SuitabilityResult(
            field_id=field_obj.id,
            crop_id=crop.id,
            soil_test_id=_safe_soil_test_id(soil_test),
            total_score=total_score,
            score_breakdown=score_breakdown,
            penalties=penalties,
            blockers=blockers,
            reasons=reasons,
            agronomic_score_value=agronomic_score,
            climate_score_value=climate_score,
            confidence_score=confidence_score,
            climate_reasons=climate_assessment.reasons,
            climate_warnings=climate_assessment.warnings,
            climate_penalties=[
                {
                    "dimension": penalty.dimension,
                    "code": penalty.code,
                    "message": penalty.message,
                    "points_lost": penalty.points_lost,
                }
                for penalty in climate_assessment.penalties
            ],
            climate_strengths=climate_assessment.strengths,
            climate_weaknesses=climate_assessment.weaknesses,
            climate_risks=climate_assessment.risks,
        )


class RuleBasedSuitabilityProvider(SuitabilityProvider):
    """Default suitability provider backed by deterministic agronomic rules."""

    def __init__(self, config: SuitabilityScoringConfig | None = None) -> None:
        self.scorer = SuitabilityScorer(config=config)

    def calculate_suitability(
        self,
        field_obj: Field,
        crop: CropProfile,
        soil_test: SoilTest | None,
        climate_summary: ClimateSummary | None = None,
    ) -> SuitabilityResult:
        return self.scorer.score_field(field_obj, crop, soil_test, climate_summary=climate_summary)


def calculate_suitability(
    field_obj: Field,
    crop: CropProfile,
    soil_test: SoilTest | None,
    climate_summary: ClimateSummary | None = None,
) -> SuitabilityResult:
    """Compatibility wrapper around the provider-backed scorer service."""

    return RuleBasedSuitabilityProvider().calculate_suitability(
        field_obj,
        crop,
        soil_test,
        climate_summary=climate_summary,
    )


def score_ph(ph_level: float, crop: CropProfile, weight: float) -> float:
    """Legacy helper retained for compatibility with older tests and callers."""

    config = load_scoring_config()
    soil = type("_Soil", (), {"ph": ph_level})()
    component = score_ph_compatibility(crop, soil, config)
    return _legacy_weighted_score(component, weight)


def score_drainage(field_drainage: str, crop_drainage: str, weight: float) -> float:
    """Legacy helper retained for compatibility with older tests and callers."""

    config = load_scoring_config()
    field_obj = type("_Field", (), {"drainage_quality": field_drainage})()
    crop = type("_Crop", (), {"drainage_requirement": crop_drainage})()
    component = score_drainage_compatibility(field_obj, crop, config)
    return _legacy_weighted_score(component, weight)


def score_irrigation(irrigation_available: bool, water_req: str, weight: float) -> float:
    """Legacy helper retained for compatibility with older tests and callers."""

    config = load_scoring_config()
    field_obj = type("_Field", (), {"irrigation_available": irrigation_available})()
    crop = type("_Crop", (), {"water_requirement_level": water_req})()
    component = score_water_availability(field_obj, crop, config)
    return _legacy_weighted_score(component, weight)


def score_slope(field_slope: float, max_slope: float, weight: float) -> float:
    """Legacy helper retained for compatibility with older tests and callers."""

    config = load_scoring_config()
    field_obj = type("_Field", (), {"slope_percent": field_slope})()
    crop = type("_Crop", (), {"slope_tolerance": max_slope})()
    component = score_slope_compatibility(field_obj, crop, config)
    return _legacy_weighted_score(component, weight)


def score_nutrient(actual: float, minimum: float, weight: float) -> float:
    """Legacy helper retained for compatibility with older tests and callers."""

    if minimum <= 0:
        return weight
    if actual >= minimum:
        return weight
    return max(0.0, weight * (actual / minimum))


def score_soil_texture(field_texture: str, preferred_textures: str, weight: float) -> float:
    """Legacy helper retained for compatibility with older tests and callers."""

    if not preferred_textures or not preferred_textures.strip():
        return weight
    textures = [texture.strip().lower() for texture in preferred_textures.split(",")]
    if field_texture.lower() in textures:
        return weight
    return 0.0
