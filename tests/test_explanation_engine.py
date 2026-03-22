from types import SimpleNamespace

from app.engines.explanation_engine import (
    build_ranked_field_explanation,
    build_suitability_explanation,
    generate_explanation,
)
from app.engines.ranking_engine import rank_fields_for_crop
from app.engines.suitability_engine import calculate_suitability
from app.schemas.weather_history import ClimateSummary
from app.services.economic_service import EconomicAssessment


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
        "slope_tolerance": 10.0,
        "organic_matter_preference": "moderate",
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def make_field(**kwargs):
    values = {
        "id": 1,
        "name": "North Parcel",
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
        "ec": 1.5,
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


def make_economic_assessment(**kwargs):
    values = {
        "estimated_revenue": 26000.0,
        "estimated_cost": 9000.0,
        "estimated_profit": 17000.0,
        "reasons": ["High profit due to high yield and low cost."],
        "strengths": ["High profit due to high yield and low cost."],
        "weaknesses": [],
    }
    values.update(kwargs)
    return EconomicAssessment(**values)


def _normalized_explanation_payload(explanation) -> dict[str, object]:
    payload = explanation.model_dump()
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("generated_at", None)
        debug_info = metadata.get("debug_info")
        if isinstance(debug_info, dict):
            risk_provider_metadata = debug_info.get("risk_provider_metadata")
            if isinstance(risk_provider_metadata, dict):
                risk_provider_metadata.pop("generated_at", None)
    return payload


def test_ranked_field_explanation_surfaces_positive_strengths():
    field_obj = make_field()
    crop = make_crop(
        optimal_temp_min_c=18.0,
        optimal_temp_max_c=30.0,
        rainfall_requirement_mm=650.0,
        frost_tolerance_days=2,
        heat_tolerance_days=20,
    )
    soil_map = {field_obj.id: make_soil()}
    climate_map = {field_obj.id: make_climate_summary()}

    ranked = rank_fields_for_crop([field_obj], crop, soil_map, climate_summaries=climate_map)
    explanation = build_ranked_field_explanation(ranked.ranked_fields[0])

    assert "ranked highly" in explanation.short_explanation.lower()
    assert "Average temperature is within the crop's ideal range." in explanation.strengths
    assert "pH is within ideal range." in explanation.strengths
    assert "Field has irrigation available." in explanation.strengths
    assert "Climate:" in explanation.detailed_explanation
    assert "Strengths:" in explanation.detailed_explanation
    assert explanation.risks == []


def test_explanation_collects_weaknesses_from_penalties():
    field_obj = make_field(drainage_quality="poor", slope_percent=15.0)
    crop = make_crop(
        drainage_requirement="good",
        slope_tolerance=10.0,
        optimal_temp_min_c=18.0,
        optimal_temp_max_c=30.0,
        rainfall_requirement_mm=650.0,
        frost_tolerance_days=2,
        heat_tolerance_days=20,
    )
    soil_map = {field_obj.id: make_soil()}
    climate_map = {field_obj.id: make_climate_summary()}

    ranked = rank_fields_for_crop([field_obj], crop, soil_map, climate_summaries=climate_map)
    explanation = build_ranked_field_explanation(ranked.ranked_fields[0])

    assert "Drainage is below crop requirement." in explanation.weaknesses
    assert "Field slope exceeds the crop tolerance." in explanation.weaknesses
    assert "Weaknesses:" in explanation.detailed_explanation
    assert "Drainage is below crop requirement." in explanation.detailed_explanation


def test_blocked_result_surfaces_risks_and_negative_short_explanation():
    field_obj = make_field(irrigation_available=False)
    crop = make_crop(
        water_requirement_level="high",
        optimal_temp_min_c=18.0,
        optimal_temp_max_c=30.0,
        rainfall_requirement_mm=650.0,
        frost_tolerance_days=2,
        heat_tolerance_days=20,
    )
    soil_map = {field_obj.id: make_soil()}
    climate_map = {field_obj.id: make_climate_summary()}

    ranked = rank_fields_for_crop([field_obj], crop, soil_map, climate_summaries=climate_map)
    explanation = build_ranked_field_explanation(ranked.ranked_fields[0])

    assert "No irrigation available for a high water-demand crop." in explanation.risks
    assert "not suitable because" in explanation.short_explanation.lower()
    assert "Blockers:" in explanation.detailed_explanation


def test_missing_soil_data_becomes_a_risk_without_crashing():
    field_obj = make_field()
    crop = make_crop()
    result = calculate_suitability(field_obj, crop, None)

    explanation = build_suitability_explanation(result, field_obj)

    assert "No soil test available for suitability scoring." in explanation.risks
    assert explanation.short_explanation.startswith("This field is not suitable")


def test_ranked_and_suitability_adapters_produce_equivalent_explanations():
    field_obj = make_field()
    crop = make_crop(
        optimal_temp_min_c=18.0,
        optimal_temp_max_c=30.0,
        rainfall_requirement_mm=650.0,
        frost_tolerance_days=2,
        heat_tolerance_days=20,
    )
    soil = make_soil()
    climate_summary = make_climate_summary()
    result = calculate_suitability(field_obj, crop, soil, climate_summary=climate_summary)
    ranked = rank_fields_for_crop(
        [field_obj],
        crop,
        {field_obj.id: soil},
        climate_summaries={field_obj.id: climate_summary},
    )

    from_ranked = build_ranked_field_explanation(ranked.ranked_fields[0])
    from_suitability = build_suitability_explanation(result, field_obj)

    assert _normalized_explanation_payload(from_ranked) == _normalized_explanation_payload(from_suitability)


def test_generate_explanation_returns_legacy_detailed_string():
    field_obj = make_field(irrigation_available=False)
    crop = make_crop(water_requirement_level="high")
    result = calculate_suitability(field_obj, crop, make_soil())

    explanation = generate_explanation(result, field_obj, crop)

    assert explanation == build_suitability_explanation(result, field_obj).detailed_explanation
    assert "Field 'North Parcel'" in explanation
    assert "Blockers:" in explanation


def test_explanation_surfaces_climate_weaknesses_when_present():
    field_obj = make_field()
    crop = make_crop(
        optimal_temp_min_c=18.0,
        optimal_temp_max_c=30.0,
        rainfall_requirement_mm=650.0,
        frost_tolerance_days=2,
        heat_tolerance_days=20,
    )
    soil = make_soil()
    climate_summary = make_climate_summary(total_rainfall=250.0, frost_days=5, heat_days=28)

    result = calculate_suitability(field_obj, crop, soil, climate_summary=climate_summary)
    explanation = build_suitability_explanation(result, field_obj)

    assert "Recent rainfall is below the crop's preferred threshold." in explanation.weaknesses
    assert "Frost risk is elevated in the recent climate window." in explanation.risks
    assert "Recent heat stress reduces suitability." in explanation.risks
    assert "Climate:" in explanation.detailed_explanation


def test_ranked_explanation_surfaces_economic_strengths_and_weaknesses():
    field_obj = make_field()
    crop = make_crop()
    ranked = rank_fields_for_crop(
        [field_obj],
        crop,
        {field_obj.id: make_soil()},
        economic_assessments={
            field_obj.id: make_economic_assessment(
                reasons=[
                    "High profit due to high yield and low cost.",
                    "Low profitability due to irrigation cost.",
                ],
                strengths=["High profit due to high yield and low cost."],
                weaknesses=["Low profitability due to irrigation cost."],
            )
        },
    )

    explanation = build_ranked_field_explanation(ranked.ranked_fields[0])

    assert "High profit due to high yield and low cost." in explanation.strengths
    assert "Low profitability due to irrigation cost." in explanation.weaknesses
