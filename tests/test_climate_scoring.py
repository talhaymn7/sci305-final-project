from types import SimpleNamespace

from app.engines.climate_scoring import assess_climate_compatibility, score_climate_compatibility
from app.engines.scoring_config import load_scoring_config
from app.engines.scoring_types import ScoreStatus
from app.engines.suitability_engine import calculate_suitability
from app.schemas.weather_history import ClimateSummary


CONFIG = load_scoring_config()


def make_crop(**kwargs):
    values = {
        "id": 1,
        "crop_name": "Corn",
        "ideal_ph_min": 6.0,
        "ideal_ph_max": 6.8,
        "tolerable_ph_min": 5.8,
        "tolerable_ph_max": 7.0,
        "water_requirement_level": "medium",
        "drainage_requirement": "moderate",
        "salinity_tolerance": "moderate",
        "rooting_depth_cm": 100.0,
        "slope_tolerance": 8.0,
        "organic_matter_preference": "moderate",
        "optimal_temp_min_c": 18.0,
        "optimal_temp_max_c": 30.0,
        "rainfall_requirement_mm": 650.0,
        "frost_tolerance_days": 2,
        "heat_tolerance_days": 20,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def make_field(**kwargs):
    values = {
        "id": 1,
        "name": "Climate Parcel",
        "irrigation_available": True,
        "drainage_quality": "good",
        "slope_percent": 3.0,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def make_soil(**kwargs):
    values = {
        "id": 1,
        "ph": 6.5,
        "organic_matter_percent": 4.0,
        "depth_cm": 120.0,
        "ec": 1.2,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def make_climate_summary(**kwargs):
    values = {
        "avg_temp": 24.0,
        "total_rainfall": 650.0,
        "frost_days": 1,
        "heat_days": 10,
    }
    values.update(kwargs)
    return ClimateSummary(**values)


def test_temperature_within_optimal_range_gets_full_reason():
    component = score_climate_compatibility(make_crop(), make_climate_summary(avg_temp=24.0), CONFIG)

    assert component.awarded_points > 0
    assert "Temperature within optimal range." in component.reasons


def test_rainfall_insufficient_reduces_climate_score():
    component = score_climate_compatibility(
        make_crop(rainfall_requirement_mm=700.0),
        make_climate_summary(total_rainfall=300.0),
        CONFIG,
    )

    assert component.awarded_points < component.max_points
    assert "Rainfall insufficient." in component.reasons


def test_high_frost_risk_detected_reduces_climate_score():
    component = score_climate_compatibility(
        make_crop(frost_tolerance_days=2),
        make_climate_summary(frost_days=5),
        CONFIG,
    )

    assert component.awarded_points < component.max_points
    assert "High frost risk detected." in component.reasons


def test_high_heat_risk_detected_reduces_climate_score():
    component = score_climate_compatibility(
        make_crop(heat_tolerance_days=10),
        make_climate_summary(heat_days=18),
        CONFIG,
    )

    assert component.awarded_points < component.max_points
    assert "High heat risk detected." in component.reasons


def test_ideal_climate_summary_produces_high_component_and_climate_score():
    result = calculate_suitability(
        make_field(),
        make_crop(),
        make_soil(),
        climate_summary=make_climate_summary(),
    )

    assert "climate_compatibility" in result.score_breakdown
    assert result.climate_score >= 90
    assert "Temperature within optimal range." in result.reasons


def test_missing_climate_summary_returns_non_blocking_missing_component():
    result = calculate_suitability(make_field(), make_crop(), make_soil(), climate_summary=None)
    component = result.score_breakdown["climate_compatibility"]

    assert component.status is ScoreStatus.MISSING
    assert component.awarded_points == 10.0
    assert not any(blocker.dimension == "climate_compatibility" for blocker in result.blockers)


def test_missing_crop_climate_requirements_return_conservative_partial_score():
    component = score_climate_compatibility(
        make_crop(
            optimal_temp_min_c=None,
            optimal_temp_max_c=None,
            rainfall_requirement_mm=None,
            frost_tolerance_days=None,
            heat_tolerance_days=None,
        ),
            make_climate_summary(),
            CONFIG,
        )

    assert component.status in {ScoreStatus.ACCEPTABLE, ScoreStatus.IDEAL}
    assert 0 < component.awarded_points <= component.max_points
    assert "Crop climate targets were inferred from AgriMind default crop benchmarks." in component.reasons


def test_climate_assessment_returns_structured_penalties_and_warnings():
    assessment = assess_climate_compatibility(
        make_crop(frost_tolerance_days=1, heat_tolerance_days=4),
        make_climate_summary(total_rainfall=250.0, frost_days=3, heat_days=7),
        CONFIG,
    )

    assert assessment.climate_score is not None
    assert assessment.penalties
    assert any(penalty.dimension == "rainfall" for penalty in assessment.penalties)
    assert any(penalty.dimension == "frost" for penalty in assessment.penalties)
    assert "Rainfall insufficient." in assessment.warnings
    assert "High frost risk detected." in assessment.warnings
