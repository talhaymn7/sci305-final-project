from datetime import date, datetime, timedelta, timezone

import pytest

from app.ai.contracts.explanation import RankedExplanationRequest, build_explanation_input_from_ranked_request
from app.ai.contracts.risk import RiskScoringRequest
from app.ai.contracts.yield_prediction import build_yield_prediction_input_from_entities
from app.ai.features.explanation import build_explanation_input
from app.ai.features.loaders import assemble_feature_bundle
from app.ai.features.risk import build_risk_input, build_risk_input_from_suitability_result
from app.ai.features.types import (
    ClimateFeatureSummary,
    CropFeatureSummary,
    FeatureSummaryBundle,
    FieldFeatureSummary,
    SoilFeatureSummary,
)
from app.ai.features.yield_prediction import build_yield_prediction_input
from app.ai.orchestration.ranking import RankingOrchestrator
from app.ai.providers.rule_based.suitability import RuleBasedSuitabilityProvider
from app.models.crop_price import CropPrice
from app.models.crop_profile import CropProfile
from app.models.enums import (
    CropDrainageRequirement,
    CropPreferenceLevel,
    CropSensitivityLevel,
    FieldAspect,
    WaterRequirementLevel,
    WaterSourceType,
)
from app.models.field import Field
from app.models.input_cost import InputCost
from app.models.soil_test import SoilTest
from app.models.weather_history import WeatherHistory
from app.services.errors import NotFoundError
from app.services.weather_service import WeatherService


def _build_field(**kwargs) -> Field:
    values = {
        "name": "North Parcel",
        "location_name": "North",
        "latitude": 39.0,
        "longitude": -94.0,
        "area_hectares": 18.0,
        "elevation_meters": 1020.0,
        "slope_percent": 3.2,
        "aspect": FieldAspect.SOUTH,
        "irrigation_available": True,
        "water_source_type": WaterSourceType.WELL,
        "infrastructure_score": 78,
        "drainage_quality": "good",
    }
    values.update(kwargs)
    return Field(**values)


def _build_crop(**kwargs) -> CropProfile:
    values = {
        "crop_name": "Corn",
        "scientific_name": "Zea mays",
        "ideal_ph_min": 6.0,
        "ideal_ph_max": 6.8,
        "tolerable_ph_min": 5.5,
        "tolerable_ph_max": 7.5,
        "water_requirement_level": WaterRequirementLevel.HIGH,
        "drainage_requirement": CropDrainageRequirement.MODERATE,
        "frost_sensitivity": CropSensitivityLevel.HIGH,
        "heat_sensitivity": CropSensitivityLevel.MEDIUM,
        "salinity_tolerance": CropPreferenceLevel.MODERATE,
        "rooting_depth_cm": 150.0,
        "slope_tolerance": 8.0,
        "optimal_temp_min_c": 18.0,
        "optimal_temp_max_c": 30.0,
        "rainfall_requirement_mm": 550.0,
        "frost_tolerance_days": 4,
        "heat_tolerance_days": 18,
        "organic_matter_preference": CropPreferenceLevel.MODERATE,
        "crop_price": CropPrice(price_per_ton=210.0),
        "input_cost": InputCost(fertilizer_cost=240.0, water_cost=165.0, labor_cost=130.0),
    }
    values.update(kwargs)
    return CropProfile(**values)


def _build_soil(field_id: int, **kwargs) -> SoilTest:
    values = {
        "field_id": field_id,
        "ph": 6.4,
        "ec": 0.9,
        "organic_matter_percent": 3.7,
        "nitrogen_ppm": 68.0,
        "phosphorus_ppm": 29.0,
        "potassium_ppm": 235.0,
        "calcium_ppm": 1700.0,
        "magnesium_ppm": 220.0,
        "texture_class": "loamy",
        "drainage_class": "good",
        "depth_cm": 145.0,
        "water_holding_capacity": 24.0,
    }
    values.update(kwargs)
    return SoilTest(**values)


def _add_weather_window(db, field_id: int, *, latest_date: date, days: int = 14) -> None:
    for offset in range(days):
        day = latest_date - timedelta(days=offset)
        db.add(
            WeatherHistory(
                field_id=field_id,
                date=day,
                min_temp=11.0,
                max_temp=27.0,
                avg_temp=19.0,
                rainfall_mm=14.5,
                humidity=56.0,
                wind_speed=3.8,
                solar_radiation=18.0,
                et0=4.6,
            )
        )


def test_assemble_feature_bundle_raises_not_found_for_missing_field_or_crop(db):
    crop = _build_crop()
    field = _build_field()
    db.add_all([crop, field])
    db.commit()

    with pytest.raises(NotFoundError, match="Field with id 999 not found"):
        assemble_feature_bundle(db, field_id=999, crop_id=crop.id)

    with pytest.raises(NotFoundError, match="Crop with id 999 not found"):
        assemble_feature_bundle(db, field_id=field.id, crop_id=999)


def test_assemble_feature_bundle_uses_latest_soil_and_optional_climate(db):
    field = _build_field()
    crop = _build_crop()
    db.add_all([field, crop])
    db.flush()

    older_sample = datetime.now(timezone.utc) - timedelta(days=10)
    newer_sample = datetime.now(timezone.utc)
    db.add_all(
        [
            _build_soil(field.id, ph=6.4, sample_date=older_sample),
            _build_soil(field.id, ph=5.2, sample_date=newer_sample),
        ]
    )
    _add_weather_window(db, field.id, latest_date=date(2026, 3, 15))
    db.commit()

    bundle = assemble_feature_bundle(db, field.id, crop.id)

    assert bundle.soil_test is not None
    assert bundle.soil_test.ph == 5.2
    assert bundle.summary.soil is not None
    assert bundle.summary.soil.ph == 5.2
    assert bundle.summary.soil.calcium_ppm == 1700.0
    assert bundle.summary.soil.magnesium_ppm == 220.0
    assert bundle.summary.climate is not None
    assert bundle.summary.climate.avg_temp == 19.0
    assert bundle.summary.climate.min_observed_temp == 11.0
    assert bundle.summary.climate.max_observed_temp == 27.0
    assert bundle.summary.climate.total_rainfall == 203.0
    assert bundle.summary.climate.avg_humidity == 56.0
    assert bundle.summary.climate.avg_wind_speed == 3.8
    assert bundle.summary.climate.avg_solar_radiation == 18.0
    assert bundle.summary.climate.weather_record_count == 14


def test_assemble_feature_bundle_returns_normalized_summary_types(db):
    field = _build_field()
    crop = _build_crop()
    db.add_all([field, crop])
    db.flush()
    db.add(_build_soil(field.id))
    db.commit()

    bundle = assemble_feature_bundle(db, field.id, crop.id)

    assert isinstance(bundle.summary, FeatureSummaryBundle)
    assert isinstance(bundle.summary.field, FieldFeatureSummary)
    assert isinstance(bundle.summary.crop, CropFeatureSummary)
    assert isinstance(bundle.summary.soil, SoilFeatureSummary)
    assert bundle.summary.climate is None or isinstance(bundle.summary.climate, ClimateFeatureSummary)
    assert not hasattr(bundle.summary.field, "_sa_instance_state")
    assert not hasattr(bundle.summary.crop, "_sa_instance_state")
    assert not hasattr(bundle.summary.soil, "_sa_instance_state")


def test_build_yield_prediction_input_matches_entity_mapping_and_handles_missing_optional_data(db):
    field = _build_field()
    crop = _build_crop()
    db.add_all([field, crop])
    db.flush()
    soil = _build_soil(field.id)
    db.add(soil)
    db.commit()

    built = build_yield_prediction_input(db, field.id, crop.id)
    manual = build_yield_prediction_input_from_entities(field, crop, soil_test=soil, climate_summary=None)

    assert built == manual
    assert built.soil is not None
    assert built.soil.calcium_ppm == 1700.0
    assert built.soil.magnesium_ppm == 220.0
    assert built.crop.ideal_ph_min == 6.0
    assert built.crop.ideal_ph_max == 6.8
    assert built.crop.frost_sensitivity == "high"
    assert built.crop.heat_sensitivity == "medium"
    assert built.climate is None


def test_build_risk_input_matches_manual_suitability_path(db):
    field = _build_field()
    crop = _build_crop()
    db.add_all([field, crop])
    db.flush()
    soil = _build_soil(field.id)
    db.add(soil)
    _add_weather_window(db, field.id, latest_date=date(2026, 3, 15))
    db.commit()

    weather_service = WeatherService(db)
    suitability_result = RuleBasedSuitabilityProvider().calculate_suitability(
        field,
        crop,
        soil,
        climate_summary=weather_service.get_climate_summary(field.id),
    )

    built = build_risk_input(db, field.id, crop.id)
    manual = build_risk_input_from_suitability_result(suitability_result)

    assert isinstance(built, RiskScoringRequest)
    assert built.breakdown == manual.breakdown
    assert built.blockers == manual.blockers


def test_build_explanation_input_maps_enriched_ranked_result(db):
    field = _build_field()
    crop = _build_crop()
    db.add_all([field, crop])
    db.flush()
    soil = _build_soil(field.id)
    db.add(soil)
    _add_weather_window(db, field.id, latest_date=date(2026, 3, 15))
    db.commit()

    climate_lookup = {field.id: WeatherService(db).get_climate_summary(field.id)}
    ranking = RankingOrchestrator().rank_fields_for_crop(
        [field],
        crop,
        {field.id: soil},
        climate_summaries=climate_lookup,
    )
    entry = ranking.ranked_fields[0]

    built = build_explanation_input(entry)
    expected = build_explanation_input_from_ranked_request(
        RankedExplanationRequest(
            field_name=entry.field_name,
            total_score=entry.total_score,
            ranking_score=entry.ranking_score,
            breakdown=entry.breakdown,
            blockers=entry.blockers,
            reasons=entry.reasons,
            penalties=entry.result.penalties,
            climate_reasons=entry.climate_reasons,
            climate_warnings=entry.climate_warnings,
            climate_strengths=entry.climate_strengths,
            climate_weaknesses=entry.climate_weaknesses,
            climate_risks=entry.climate_risks,
            economic_strengths=entry.economic_strengths,
            economic_weaknesses=entry.economic_weaknesses,
            field_id=entry.field_id,
            crop_id=entry.crop_id,
            crop_name=entry.feature_context.crop.crop_name,
            rank=entry.rank,
            economic_score=entry.economic_score,
            estimated_profit=entry.estimated_profit,
            feature_context=entry.feature_context,
        )
    )

    assert built == expected
    assert built.feature_context is entry.feature_context
    assert built.ranking_result.crop_name == crop.crop_name
    assert built.feature_context is not None
    assert built.feature_context.field.field_id == field.id
    assert built.feature_context.crop.crop_id == crop.id
