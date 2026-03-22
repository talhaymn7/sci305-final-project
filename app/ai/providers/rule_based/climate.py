"""Rule-based climate scoring provider backed by the deterministic engine."""

from app.engines.climate_scoring import (
    ClimateAssessment,
    ClimateFactorAssessment,
    ClimateScorePenalty,
    ClimateScoringInput,
    assess_climate_compatibility,
    assess_climate_input,
    assess_climate_requirements,
    compute_climate_score,
    score_climate_compatibility,
)

__all__ = [
    "ClimateAssessment",
    "ClimateFactorAssessment",
    "ClimateScorePenalty",
    "ClimateScoringInput",
    "assess_climate_compatibility",
    "assess_climate_input",
    "assess_climate_requirements",
    "compute_climate_score",
    "score_climate_compatibility",
]
