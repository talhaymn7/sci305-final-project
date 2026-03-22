from datetime import date
from types import SimpleNamespace

from app.engines.climate_scoring import assess_climate_compatibility, compute_climate_score
from app.engines.explanation_engine import build_suitability_explanation
from app.engines.ranking_engine import rank_fields_for_crop
from app.engines.suitability_engine import calculate_suitability
from app.models.enums import WaterSourceType
from app.models.field import Field
from app.models.weather_history import WeatherHistory
from app.schemas.weather_history import ClimateSummary
from app.services.climate_feature_builder import build_climate_features
from app.services.climate_summary_service import get_climate_summary
from app.services.weather_service import WeatherService


def _create_weather_field(db, **overrides) -> Field:
    values = {
        "name": "Climate Logic Field",
        "location_name": "Decision Valley",
        "latitude": 39.1,
        "longitude": -94.5,
        "area_hectares": 12.0,
        "elevation_meters": 220.0,
        "slope_percent": 3.0,
        "irrigation_available": True,
        "water_source_type": WaterSourceType.WELL,
        "infrastructure_score": 70,
        "drainage_quality": "good",
    }
    values.update(overrides)
    field = Field(**values)
    db.add(field)
    db.flush()
    return field


def _add_weather_record(db, field_id: int, **overrides) -> WeatherHistory:
    values = {
        "field_id": field_id,
        "date": date(2026, 3, 10),
        "min_temp": 8.0,
        "max_temp": 28.0,
        "avg_temp": 18.0,
        "rainfall_mm": 5.0,
        "humidity": 60.0,
        "wind_speed": 5.0,
        "solar_radiation": 16.0,
        "et0": 3.0,
    }
    values.update(overrides)
    record = WeatherHistory(**values)
    db.add(record)
    return record


def _make_crop(**overrides):
    values = {
        "id": 1,
        "crop_name": "Corn",
        "ideal_ph_min": 6.0,
        "ideal_ph_max": 6.8,
        "tolerable_ph_min": 5.8,
        "tolerable_ph_max": 7.0,
        "water_requirement_level": "medium",
        "drainage_requirement": "moderate",
        "frost_sensitivity": "high",
        "heat_sensitivity": "medium",
        "salinity_tolerance": "moderate",
        "rooting_depth_cm": 120.0,
        "slope_tolerance": 8.0,
        "organic_matter_preference": "moderate",
        "optimal_temp_min_c": 18.0,
        "optimal_temp_max_c": 30.0,
        "preferred_rainfall_min_mm": 500.0,
        "preferred_rainfall_max_mm": 700.0,
        "rainfall_requirement_mm": 600.0,
        "frost_tolerance_days": 2,
        "heat_tolerance_days": 10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_field(field_id: int, name: str, **overrides):
    values = {
        "id": field_id,
        "name": name,
        "irrigation_available": True,
        "drainage_quality": "good",
        "slope_percent": 3.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_soil(**overrides):
    values = {
        "id": 1,
        "ph": 6.5,
        "organic_matter_percent": 4.0,
        "depth_cm": 120.0,
        "ec": 1.2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_climate_summary(**overrides):
    values = {
        "avg_temp": 24.0,
        "total_rainfall": 650.0,
        "frost_days": 1,
        "heat_days": 5,
        "coverage_ratio": 1.0,
    }
    values.update(overrides)
    return ClimateSummary(**values)


def test_climate_summary_aggregation_uses_latest_window_metrics(db):
    field = _create_weather_field(db)
    _add_weather_record(
        db,
        field.id,
        date=date(2026, 3, 7),
        min_temp=2.0,
        max_temp=20.0,
        avg_temp=11.0,
        rainfall_mm=50.0,
        et0=1.5,
    )
    _add_weather_record(
        db,
        field.id,
        date=date(2026, 3, 8),
        min_temp=7.0,
        max_temp=28.0,
        avg_temp=18.0,
        rainfall_mm=3.0,
        et0=2.5,
    )
    _add_weather_record(
        db,
        field.id,
        date=date(2026, 3, 9),
        min_temp=9.0,
        max_temp=30.0,
        avg_temp=20.0,
        rainfall_mm=5.0,
        et0=3.0,
    )
    _add_weather_record(
        db,
        field.id,
        date=date(2026, 3, 10),
        min_temp=11.0,
        max_temp=32.0,
        avg_temp=22.0,
        rainfall_mm=7.0,
        et0=3.5,
    )
    db.commit()

    summary = WeatherService(db).get_climate_summary(field.id, days=3)

    assert summary is not None
    assert summary.avg_temp == 20.0
    assert summary.min_temp_avg == 9.0
    assert summary.avg_min_temp == 9.0
    assert summary.max_temp_avg == 30.0
    assert summary.avg_max_temp == 30.0
    assert summary.total_rainfall == 15.0
    assert summary.avg_wind == 5.0
    assert summary.avg_solar == 16.0
    assert summary.total_et0 == 9.0
    assert summary.observation_days == 3
    assert summary.observation_days_count == 3
    assert summary.missing_days_count == 0
    assert summary.observation_start_date == date(2026, 3, 8)
    assert summary.observation_end_date == date(2026, 3, 10)


def test_climate_summary_counts_frost_days_from_subzero_minimums(db):
    field = _create_weather_field(db, name="Frost Counting Field")
    _add_weather_record(db, field.id, date=date(2026, 3, 7), min_temp=-2.0, max_temp=8.0, avg_temp=2.0)
    _add_weather_record(db, field.id, date=date(2026, 3, 8), min_temp=0.0, max_temp=10.0, avg_temp=4.0)
    _add_weather_record(db, field.id, date=date(2026, 3, 9), min_temp=-0.1, max_temp=11.0, avg_temp=5.0)
    _add_weather_record(db, field.id, date=date(2026, 3, 10), min_temp=3.0, max_temp=14.0, avg_temp=7.0)
    db.commit()

    summary = WeatherService(db).get_climate_summary(field.id, days=4)

    assert summary is not None
    assert summary.frost_days == 2
    assert summary.frost_days_count == 2


def test_climate_summary_counts_heat_days_above_threshold(db):
    field = _create_weather_field(db, name="Heat Counting Field")
    _add_weather_record(db, field.id, date=date(2026, 3, 7), max_temp=29.9, avg_temp=22.0)
    _add_weather_record(db, field.id, date=date(2026, 3, 8), max_temp=30.0, avg_temp=23.0)
    _add_weather_record(db, field.id, date=date(2026, 3, 9), max_temp=30.1, avg_temp=24.0)
    _add_weather_record(db, field.id, date=date(2026, 3, 10), max_temp=40.0, avg_temp=28.0)
    db.commit()

    summary = WeatherService(db).get_climate_summary(field.id, days=4)

    assert summary is not None
    assert summary.heat_threshold_c == 30.0
    assert summary.heat_days == 2
    assert summary.heat_days_count == 2


def test_get_climate_summary_service_wrapper_returns_sql_aggregated_summary(db):
    field = _create_weather_field(db, name="Wrapper Summary Field")
    _add_weather_record(db, field.id, date=date(2026, 3, 9), min_temp=-1.0, max_temp=31.0, avg_temp=15.0, rainfall_mm=4.0)
    _add_weather_record(db, field.id, date=date(2026, 3, 10), min_temp=2.0, max_temp=29.0, avg_temp=17.0, rainfall_mm=6.0)
    db.commit()

    summary = get_climate_summary(db, field.id, days=2)

    assert summary is not None
    assert summary.avg_temp == 16.0
    assert summary.total_rainfall == 10.0
    assert summary.frost_days == 1
    assert summary.heat_days == 1


def test_climate_score_calculation_penalizes_temperature_rainfall_frost_and_heat():
    assessment = assess_climate_compatibility(
        _make_crop(
            frost_tolerance_days=1,
            heat_tolerance_days=3,
            preferred_rainfall_min_mm=500.0,
            preferred_rainfall_max_mm=700.0,
        ),
        _make_climate_summary(
            avg_temp=34.0,
            total_rainfall=250.0,
            frost_days=4,
            heat_days=8,
        ),
    )

    assert assessment.climate_score is not None
    assert assessment.climate_score < 50.0
    assert {penalty.dimension for penalty in assessment.penalties} >= {
        "temperature",
        "rainfall",
        "frost",
        "heat",
    }
    assert any("rainfall" in warning.lower() for warning in assessment.warnings)
    assert any("frost" in warning.lower() for warning in assessment.warnings)
    assert any("heat" in warning.lower() for warning in assessment.warnings)


def test_compute_climate_score_returns_real_score_penalties_risks_and_reasons():
    assessment = compute_climate_score(
        _make_climate_summary(avg_temp=34.0, total_rainfall=250.0, frost_days=4, heat_days=8),
        _make_crop(frost_tolerance_days=1, heat_tolerance_days=3),
    )

    assert assessment.score is not None
    assert assessment.score < 50.0
    assert assessment.penalties
    assert assessment.risks
    assert assessment.reasons


def test_build_climate_features_service_wrapper_returns_structured_features(db):
    field = _create_weather_field(db, name="Wrapper Feature Field")
    db.add_all(
        [
            WeatherHistory(
                field_id=field.id,
                date=date(2026, 3, 9),
                min_temp=-1.0,
                max_temp=31.0,
                avg_temp=15.0,
                rainfall_mm=4.0,
                humidity=60.0,
                wind_speed=4.0,
                solar_radiation=14.0,
                et0=2.0,
            ),
            WeatherHistory(
                field_id=field.id,
                date=date(2026, 3, 10),
                min_temp=2.0,
                max_temp=29.0,
                avg_temp=17.0,
                rainfall_mm=6.0,
                humidity=62.0,
                wind_speed=5.0,
                solar_radiation=16.0,
                et0=2.5,
            ),
        ]
    )
    from app.models.crop_profile import CropProfile
    from app.models.enums import (
        CropDrainageRequirement,
        CropPreferenceLevel,
        CropSensitivityLevel,
        WaterRequirementLevel,
    )

    crop = CropProfile(
        crop_name="Corn",
        scientific_name="Zea mays",
        ideal_ph_min=6.0,
        ideal_ph_max=6.8,
        tolerable_ph_min=5.8,
        tolerable_ph_max=7.2,
        water_requirement_level=WaterRequirementLevel.MEDIUM,
        drainage_requirement=CropDrainageRequirement.MODERATE,
        frost_sensitivity=CropSensitivityLevel.HIGH,
        heat_sensitivity=CropSensitivityLevel.MEDIUM,
        salinity_tolerance=CropPreferenceLevel.MODERATE,
        rooting_depth_cm=120.0,
        slope_tolerance=8.0,
        optimal_temp_min_c=18.0,
        optimal_temp_max_c=30.0,
        preferred_rainfall_min_mm=20.0,
        preferred_rainfall_max_mm=40.0,
        frost_tolerance_days=1,
        heat_tolerance_days=4,
        organic_matter_preference=CropPreferenceLevel.MODERATE,
    )
    db.add(crop)
    db.commit()

    features = build_climate_features(db, field.id, crop.id, days=2)

    assert features.field_id == field.id
    assert features.crop_id == crop.id
    assert features.summary is not None
    assert features.requirements.optimal_temp_min_c == 18.0
    assert features.observation_days_count == 2


def test_ranking_output_contains_climate_score_and_climate_reasons():
    field_a = _make_field(1, "Climate Fit")
    field_b = _make_field(2, "Climate Stress")
    crop = _make_crop()
    soil_map = {
        1: _make_soil(id=1),
        2: _make_soil(id=2),
    }
    climate_map = {
        1: _make_climate_summary(avg_temp=24.0, total_rainfall=650.0, frost_days=1, heat_days=5),
        2: _make_climate_summary(avg_temp=35.0, total_rainfall=250.0, frost_days=6, heat_days=16),
    }

    ranked = rank_fields_for_crop(
        [field_a, field_b],
        crop,
        soil_map,
        climate_summaries=climate_map,
    )

    assert [entry.field_id for entry in ranked.ranked_fields] == [1, 2]
    assert ranked.ranked_fields[0].climate_score is not None
    assert ranked.ranked_fields[1].climate_score is not None
    assert ranked.ranked_fields[0].climate_score > ranked.ranked_fields[1].climate_score
    assert ranked.ranked_fields[0].climate_reasons
    assert any("frost" in message.lower() for message in ranked.ranked_fields[1].climate_warnings)
    assert any("heat" in message.lower() for message in ranked.ranked_fields[1].climate_warnings)


def test_explanation_output_includes_climate_related_reasoning():
    result = calculate_suitability(
        _make_field(1, "Explained Field"),
        _make_crop(),
        _make_soil(),
        climate_summary=_make_climate_summary(
            avg_temp=24.0,
            total_rainfall=300.0,
            frost_days=5,
            heat_days=18,
        ),
    )

    explanation = build_suitability_explanation(result, _make_field(1, "Explained Field"))

    assert "Average temperature is within the crop's ideal range." in explanation.strengths
    assert "Recent rainfall is below the crop's preferred threshold." in explanation.weaknesses
    assert "Frost risk is elevated in the recent climate window." in explanation.risks
    assert "Recent heat stress reduces suitability." in explanation.risks
    assert "Climate:" in explanation.detailed_explanation
