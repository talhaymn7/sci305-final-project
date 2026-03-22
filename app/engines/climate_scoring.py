"""Deterministic climate scoring engine for field and crop compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.engines.scoring_config import SuitabilityScoringConfig, load_scoring_config
from app.engines.scoring_types import ScoreComponent, ScoreStatus
from app.schemas.weather_history import ClimateSummary
from app.services.climate_feature_builder import ClimateFeatures
from app.services.crop_climate_requirements import (
    CropClimateRequirements,
    resolve_crop_climate_requirements,
)


@dataclass(frozen=True, slots=True)
class ClimateScorePenalty:
    """Structured climate penalty surfaced alongside the aggregate climate score."""

    dimension: str
    code: str
    message: str
    points_lost: float


@dataclass(frozen=True, slots=True)
class ClimateFactorAssessment:
    """Assessment for one deterministic climate factor."""

    dimension: str
    ratio: float
    status: ScoreStatus
    reasons: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    missing: bool = False


@dataclass(frozen=True, slots=True)
class ClimateScoringInput:
    """Typed climate scoring input detached from ORM models."""

    climate_summary: ClimateSummary | None = None
    climate_features: ClimateFeatures | None = None
    crop_requirements: CropClimateRequirements | None = None

    def resolved_summary(self) -> ClimateSummary | None:
        if self.climate_summary is not None:
            return self.climate_summary
        if self.climate_features is not None:
            return self.climate_features.summary
        return None

    def resolved_requirements(self) -> CropClimateRequirements | None:
        if self.crop_requirements is not None:
            return self.crop_requirements
        if self.climate_features is not None:
            return self.climate_features.requirements
        return None


@dataclass(frozen=True, slots=True)
class ClimateAssessment:
    """Rich climate assessment surfaced to suitability, ranking, and explanations."""

    component: ScoreComponent
    climate_score: float | None
    confidence_score: float | None
    reasons: list[str] = field(default_factory=list)
    penalties: list[ClimateScorePenalty] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    factors: dict[str, ClimateFactorAssessment] = field(default_factory=dict)


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _round_points(value: float) -> float:
    return round(value, 2)


def _normalize_score(ratio: float) -> float:
    return _round_points(_clamp_ratio(ratio) * 100.0)


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


def _penalty_code(dimension: str, message: str) -> str:
    normalized = "".join(
        character if character.isalnum() else "_"
        for character in message.strip().lower()
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    return f"{dimension}_{normalized}"[:96]


def _status_from_ratio(ratio: float, *, missing: bool) -> ScoreStatus:
    if missing:
        return ScoreStatus.MISSING
    if ratio >= 0.95:
        return ScoreStatus.IDEAL
    if ratio >= 0.6:
        return ScoreStatus.ACCEPTABLE
    return ScoreStatus.LIMITED


def _missing_factor(message: str, missing_ratio: float, *, dimension: str) -> ClimateFactorAssessment:
    return ClimateFactorAssessment(
        dimension=dimension,
        ratio=missing_ratio,
        status=ScoreStatus.MISSING,
        reasons=[message],
        risks=[message],
        missing=True,
    )


def _temperature_factor(
    climate_summary: ClimateSummary,
    requirements: CropClimateRequirements,
    config: SuitabilityScoringConfig,
) -> ClimateFactorAssessment:
    avg_temp = climate_summary.avg_temp
    if avg_temp is None:
        return _missing_factor(
            "Average temperature is unavailable for the recent climate window.",
            config.thresholds.missing_climate_ratio,
            dimension="temperature",
        )
    if requirements.optimal_temp_min_c is None or requirements.optimal_temp_max_c is None:
        return _missing_factor(
            "Crop temperature requirement is not configured.",
            config.thresholds.missing_climate_ratio,
            dimension="temperature",
        )

    tolerable_min = requirements.tolerable_temp_min_c
    tolerable_max = requirements.tolerable_temp_max_c
    if tolerable_min is None:
        tolerable_min = requirements.optimal_temp_min_c - config.thresholds.temperature_buffer_c
    if tolerable_max is None:
        tolerable_max = requirements.optimal_temp_max_c + config.thresholds.temperature_buffer_c

    if requirements.optimal_temp_min_c <= avg_temp <= requirements.optimal_temp_max_c:
        return ClimateFactorAssessment(
            dimension="temperature",
            ratio=1.0,
            status=ScoreStatus.IDEAL,
            reasons=[
                "Recent temperatures stayed within the crop's preferred range.",
                "Temperature within optimal range.",
            ],
            strengths=[
                "Recent temperatures stayed within the crop's preferred range.",
                "Temperature within optimal range.",
            ],
        )

    if tolerable_min <= avg_temp <= tolerable_max:
        if avg_temp < requirements.optimal_temp_min_c:
            span = requirements.optimal_temp_min_c - tolerable_min
            ratio = (avg_temp - tolerable_min) / span if span > 0 else 1.0
            weakness = "Recent temperatures were cooler than the crop's preferred range."
        else:
            span = tolerable_max - requirements.optimal_temp_max_c
            ratio = (tolerable_max - avg_temp) / span if span > 0 else 1.0
            weakness = "Recent temperatures were warmer than the crop's preferred range."
        return ClimateFactorAssessment(
            dimension="temperature",
            ratio=max(0.35, _clamp_ratio(ratio)),
            status=ScoreStatus.ACCEPTABLE,
            reasons=[weakness],
            weaknesses=[weakness],
        )

    risk = "Recent temperatures fell outside the crop's tolerable range."
    return ClimateFactorAssessment(
        dimension="temperature",
        ratio=0.0,
        status=ScoreStatus.LIMITED,
        reasons=[risk],
        weaknesses=[risk],
        risks=[risk],
    )


def _acceptable_rainfall_band(
    preferred_min_mm: float | None,
    preferred_max_mm: float | None,
    config: SuitabilityScoringConfig,
) -> tuple[float | None, float | None]:
    acceptable_min = None
    acceptable_max = None
    if preferred_min_mm is not None:
        lower_ratio = (
            config.thresholds.rainfall_acceptable_lower_ratio
            / config.thresholds.rainfall_ideal_lower_ratio
        )
        acceptable_min = round(preferred_min_mm * lower_ratio, 2)
    if preferred_max_mm is not None:
        upper_ratio = (
            config.thresholds.rainfall_acceptable_upper_ratio
            / config.thresholds.rainfall_ideal_upper_ratio
        )
        acceptable_max = round(preferred_max_mm * upper_ratio, 2)
    return acceptable_min, acceptable_max


def _rainfall_factor(
    climate_summary: ClimateSummary,
    requirements: CropClimateRequirements,
    config: SuitabilityScoringConfig,
) -> ClimateFactorAssessment:
    total_rainfall = climate_summary.total_rainfall
    if total_rainfall is None:
        return _missing_factor(
            "Rainfall total is unavailable for the recent climate window.",
            config.thresholds.missing_climate_ratio,
            dimension="rainfall",
        )

    preferred_min = requirements.preferred_rainfall_min_mm
    preferred_max = requirements.preferred_rainfall_max_mm
    if preferred_min is None and preferred_max is None:
        return _missing_factor(
            "Crop rainfall requirement is not configured.",
            config.thresholds.missing_climate_ratio,
            dimension="rainfall",
        )

    acceptable_min, acceptable_max = _acceptable_rainfall_band(preferred_min, preferred_max, config)

    within_lower = preferred_min is None or total_rainfall >= preferred_min
    within_upper = preferred_max is None or total_rainfall <= preferred_max
    if within_lower and within_upper:
        return ClimateFactorAssessment(
            dimension="rainfall",
            ratio=1.0,
            status=ScoreStatus.IDEAL,
            reasons=[
                "Rainfall over the lookback period stayed within the crop's preferred range.",
            ],
            strengths=[
                "Rainfall over the lookback period stayed within the crop's preferred range.",
            ],
        )

    if acceptable_min is not None and total_rainfall < preferred_min:
        if total_rainfall >= acceptable_min:
            span = preferred_min - acceptable_min
            ratio = (total_rainfall - acceptable_min) / span if span > 0 else 1.0
            weakness = "Rainfall over the lookback period was below the crop's preferred range."
            return ClimateFactorAssessment(
                dimension="rainfall",
                ratio=max(0.35, _clamp_ratio(ratio)),
                status=ScoreStatus.ACCEPTABLE,
                reasons=[weakness, "Rainfall insufficient."],
                weaknesses=[weakness],
                risks=["Rainfall insufficient."],
            )
        risk = "Rainfall over the lookback period was materially below the crop's preferred range."
        return ClimateFactorAssessment(
            dimension="rainfall",
            ratio=0.0,
            status=ScoreStatus.LIMITED,
            reasons=[risk, "Rainfall insufficient."],
            weaknesses=[risk],
            risks=["Rainfall insufficient.", risk],
        )

    if acceptable_max is not None and total_rainfall > preferred_max:
        if total_rainfall <= acceptable_max:
            span = acceptable_max - preferred_max
            ratio = (acceptable_max - total_rainfall) / span if span > 0 else 1.0
            weakness = "Rainfall over the lookback period was above the crop's preferred range."
            return ClimateFactorAssessment(
                dimension="rainfall",
                ratio=max(0.35, _clamp_ratio(ratio)),
                status=ScoreStatus.ACCEPTABLE,
                reasons=[weakness],
                weaknesses=[weakness],
            )
        risk = "Rainfall over the lookback period was materially above the crop's preferred range."
        return ClimateFactorAssessment(
            dimension="rainfall",
            ratio=0.0,
            status=ScoreStatus.LIMITED,
            reasons=[risk],
            weaknesses=[risk],
            risks=[risk],
        )

    return _missing_factor(
        "Rainfall suitability could not be calculated from the current crop settings.",
        config.thresholds.missing_climate_ratio,
        dimension="rainfall",
    )


def _stress_factor(
    observed_days: int | None,
    tolerance_days: int | None,
    *,
    dimension: str,
    partial_multiplier: float,
    positive_message: str,
    risk_message: str,
    legacy_risk_message: str | None = None,
    config: SuitabilityScoringConfig,
) -> ClimateFactorAssessment:
    if observed_days is None:
        return _missing_factor(
            f"{positive_message.split()[1]}-day signal is unavailable for the recent climate window.",
            config.thresholds.missing_climate_ratio,
            dimension=dimension,
        )
    if tolerance_days is None:
        return _missing_factor(
            f"{positive_message.split()[1]} tolerance is not configured for this crop.",
            config.thresholds.missing_climate_ratio,
            dimension=dimension,
        )
    if observed_days <= tolerance_days:
        return ClimateFactorAssessment(
            dimension=dimension,
            ratio=1.0,
            status=ScoreStatus.IDEAL,
            reasons=[positive_message],
            strengths=[positive_message],
        )

    partial_limit = max(tolerance_days + 1, int(round(tolerance_days * partial_multiplier)))
    if observed_days <= partial_limit:
        span = partial_limit - tolerance_days
        ratio = (partial_limit - observed_days) / span if span > 0 else 0.0
        reasons = [risk_message]
        if legacy_risk_message is not None:
            reasons.append(legacy_risk_message)
        return ClimateFactorAssessment(
            dimension=dimension,
            ratio=max(0.2, _clamp_ratio(ratio)),
            status=ScoreStatus.ACCEPTABLE,
            reasons=reasons,
            weaknesses=[risk_message],
            risks=[legacy_risk_message or risk_message],
        )

    reasons = [risk_message]
    if legacy_risk_message is not None:
        reasons.append(legacy_risk_message)
    return ClimateFactorAssessment(
        dimension=dimension,
        ratio=0.0,
        status=ScoreStatus.LIMITED,
        reasons=reasons,
        weaknesses=[risk_message],
        risks=[legacy_risk_message or risk_message, risk_message],
    )


def _frost_factor(
    climate_summary: ClimateSummary,
    requirements: CropClimateRequirements,
    config: SuitabilityScoringConfig,
) -> ClimateFactorAssessment:
    return _stress_factor(
        climate_summary.frost_days,
        requirements.frost_tolerance_days,
        dimension="frost",
        partial_multiplier=config.thresholds.frost_tolerance_partial_multiplier,
        positive_message="Recent frost exposure stayed within the crop's tolerance.",
        risk_message="Elevated frost risk was observed in the recent climate window.",
        legacy_risk_message="High frost risk detected.",
        config=config,
    )


def _heat_factor(
    climate_summary: ClimateSummary,
    requirements: CropClimateRequirements,
    config: SuitabilityScoringConfig,
) -> ClimateFactorAssessment:
    return _stress_factor(
        climate_summary.heat_days,
        requirements.heat_tolerance_days,
        dimension="heat",
        partial_multiplier=config.thresholds.heat_tolerance_partial_multiplier,
        positive_message="Recent heat stress stayed within the crop's tolerance.",
        risk_message="Elevated heat stress was observed in the recent climate window.",
        legacy_risk_message="High heat risk detected.",
        config=config,
    )


def _coverage_adjusted_ratio(
    raw_ratio: float,
    climate_summary: ClimateSummary | None,
    config: SuitabilityScoringConfig,
) -> float:
    if climate_summary is None or climate_summary.coverage_ratio is None:
        return raw_ratio
    coverage_ratio = _clamp_ratio(climate_summary.coverage_ratio)
    baseline = config.thresholds.missing_climate_ratio
    return _clamp_ratio((raw_ratio * coverage_ratio) + (baseline * (1.0 - coverage_ratio)))


def _confidence_score(
    *,
    climate_summary: ClimateSummary | None,
    requirements: CropClimateRequirements,
    missing_factor_count: int,
) -> float:
    if climate_summary is None:
        return 0.52

    coverage_ratio = climate_summary.coverage_ratio if climate_summary.coverage_ratio is not None else 1.0
    confidence = 0.62 + (0.25 * _clamp_ratio(coverage_ratio))
    if requirements.source == "named_default":
        confidence -= 0.08
    elif requirements.source == "heuristic":
        confidence -= 0.14
    confidence -= 0.05 * missing_factor_count
    return round(max(0.35, min(confidence, 0.96)), 2)


def _build_penalties(
    component_weight: float,
    weighted_factors: list[tuple[str, ClimateFactorAssessment, float]],
) -> list[ClimateScorePenalty]:
    penalties: list[ClimateScorePenalty] = []
    for dimension, factor, subweight in weighted_factors:
        points_lost = round(component_weight * subweight * (1.0 - _clamp_ratio(factor.ratio)), 2)
        if points_lost <= 0:
            continue
        message = next(
            (
                candidate
                for candidate in [*factor.weaknesses, *factor.risks, *factor.reasons]
                if candidate.strip()
            ),
            f"{dimension.title()} compatibility reduced the climate score.",
        )
        penalties.append(
            ClimateScorePenalty(
                dimension=dimension,
                code=_penalty_code(dimension, message),
                message=message,
                points_lost=points_lost,
            )
        )
    return penalties


def assess_climate_input(
    scoring_input: ClimateScoringInput,
    config: SuitabilityScoringConfig | None = None,
) -> ClimateAssessment:
    """Score climate compatibility from typed climate inputs and crop requirements."""

    active_config = config or load_scoring_config()
    component_key = "climate_compatibility"
    component_label = "Climate compatibility"
    component_weight = active_config.climate_compatibility
    climate_summary = scoring_input.resolved_summary()
    requirements = scoring_input.resolved_requirements()

    if requirements is None:
        raise ValueError("Climate scoring requires crop climate requirements.")

    if climate_summary is None:
        reasons = [
            "Recent climate summary is unavailable; climate scoring used a conservative fallback.",
        ]
        warnings = ["Recent climate summary is unavailable for this field."]
        component = ScoreComponent(
            key=component_key,
            label=component_label,
            weight=component_weight,
            awarded_points=_round_points(component_weight * active_config.thresholds.missing_climate_ratio),
            max_points=component_weight,
            status=ScoreStatus.MISSING,
            reasons=reasons,
        )
        return ClimateAssessment(
            component=component,
            climate_score=None,
            confidence_score=0.52,
            reasons=reasons,
            penalties=[
                ClimateScorePenalty(
                    dimension="coverage",
                    code="coverage_missing_recent_climate_summary",
                    message=warnings[0],
                    points_lost=_round_points(
                        component_weight * (1.0 - active_config.thresholds.missing_climate_ratio)
                    ),
                )
            ],
            warnings=warnings,
            weaknesses=warnings,
            risks=warnings,
            factors={},
        )

    temperature = _temperature_factor(climate_summary, requirements, active_config)
    rainfall = _rainfall_factor(climate_summary, requirements, active_config)
    frost = _frost_factor(climate_summary, requirements, active_config)
    heat = _heat_factor(climate_summary, requirements, active_config)
    factors = {
        "temperature": temperature,
        "rainfall": rainfall,
        "frost": frost,
        "heat": heat,
    }
    weighted_factors = [
        ("temperature", temperature, active_config.climate_subweights.temperature),
        ("rainfall", rainfall, active_config.climate_subweights.rainfall),
        ("frost", frost, active_config.climate_subweights.frost),
        ("heat", heat, active_config.climate_subweights.heat),
    ]

    raw_ratio = (
        (temperature.ratio * active_config.climate_subweights.temperature)
        + (rainfall.ratio * active_config.climate_subweights.rainfall)
        + (frost.ratio * active_config.climate_subweights.frost)
        + (heat.ratio * active_config.climate_subweights.heat)
    )
    weighted_ratio = _coverage_adjusted_ratio(raw_ratio, climate_summary, active_config)
    missing_factor_count = sum(1 for factor in factors.values() if factor.missing)
    missing_only = missing_factor_count == len(factors)

    strengths = _dedupe_messages([message for factor in factors.values() for message in factor.strengths])
    weaknesses = _dedupe_messages([message for factor in factors.values() for message in factor.weaknesses])
    risks = _dedupe_messages([message for factor in factors.values() for message in factor.risks])
    reasons = _dedupe_messages([message for factor in factors.values() for message in factor.reasons])

    if requirements.source == "named_default":
        reasons.append("Crop climate targets were inferred from AgriMind default crop benchmarks.")
    elif requirements.source == "heuristic":
        reasons.append("Crop climate targets were inferred from partial crop settings.")

    if missing_only:
        component_reasons = [
            "Climate observations were incomplete for the recent scoring window.",
        ]
    elif weighted_ratio >= 0.95:
        component_reasons = strengths or reasons
    else:
        component_reasons = _dedupe_messages([*weaknesses, *risks, *reasons]) or reasons

    component = ScoreComponent(
        key=component_key,
        label=component_label,
        weight=component_weight,
        awarded_points=_round_points(component_weight * weighted_ratio),
        max_points=component_weight,
        status=_status_from_ratio(weighted_ratio, missing=missing_only),
        reasons=_dedupe_messages(component_reasons),
    )

    warnings = _dedupe_messages(
        [
            *risks,
            *(
                factor.reasons[0]
                for factor in factors.values()
                if factor.missing and factor.reasons
            ),
        ]
    )

    return ClimateAssessment(
        component=component,
        climate_score=_normalize_score(weighted_ratio),
        confidence_score=_confidence_score(
            climate_summary=climate_summary,
            requirements=requirements,
            missing_factor_count=missing_factor_count,
        ),
        reasons=_dedupe_messages(reasons),
        penalties=_build_penalties(component_weight, weighted_factors),
        warnings=warnings,
        strengths=strengths,
        weaknesses=weaknesses,
        risks=risks,
        factors=factors,
    )


def assess_climate_requirements(
    climate_summary_or_features: ClimateSummary | ClimateFeatures | None,
    crop_requirements: CropClimateRequirements,
    config: SuitabilityScoringConfig | None = None,
) -> ClimateAssessment:
    """Score climate compatibility from a summary/features object and explicit requirements."""

    if isinstance(climate_summary_or_features, ClimateFeatures):
        scoring_input = ClimateScoringInput(
            climate_features=climate_summary_or_features,
            crop_requirements=crop_requirements,
        )
    else:
        scoring_input = ClimateScoringInput(
            climate_summary=climate_summary_or_features,
            crop_requirements=crop_requirements,
        )
    return assess_climate_input(scoring_input, config=config)


def assess_climate_compatibility(
    crop,
    climate_summary: ClimateSummary | None,
    config: SuitabilityScoringConfig | None = None,
) -> ClimateAssessment:
    """Compatibility wrapper that resolves crop requirements from a crop profile-like object."""

    return assess_climate_input(
        ClimateScoringInput(
            climate_summary=climate_summary,
            crop_requirements=resolve_crop_climate_requirements(crop),
        ),
        config=config,
    )


def score_climate_compatibility(
    crop,
    climate_summary: ClimateSummary | None,
    config: SuitabilityScoringConfig | None = None,
) -> ScoreComponent:
    """Compatibility wrapper that returns only the climate score component."""

    return assess_climate_compatibility(crop, climate_summary, config=config).component


__all__ = [
    "ClimateAssessment",
    "ClimateFactorAssessment",
    "ClimateScorePenalty",
    "ClimateScoringInput",
    "assess_climate_compatibility",
    "assess_climate_input",
    "assess_climate_requirements",
    "score_climate_compatibility",
]
