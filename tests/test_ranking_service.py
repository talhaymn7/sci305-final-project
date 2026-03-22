from datetime import datetime, timedelta, timezone

import pytest

from app.ai.contracts.explanation import ExplanationOutput
from app.ai.contracts.metadata import AITraceMetadata
from app.ai.registry import AIProviderRegistry
from app.config import settings
from app.models.crop_price import CropPrice
from app.models.crop_profile import CropProfile
from app.models.enums import (
    CropDrainageRequirement,
    CropPreferenceLevel,
    CropSensitivityLevel,
    WaterRequirementLevel,
)
from app.models.field import Field
from app.models.input_cost import InputCost
from app.models.soil_test import SoilTest
from app.models.weather_history import WeatherHistory
from app.services.errors import NotFoundError
from app.services.ranking_service import get_ranked_fields_response


def make_crop(**kwargs) -> CropProfile:
    values = {
        "crop_name": "Corn",
        "scientific_name": "Zea mays",
        "ideal_ph_min": 6.0,
        "ideal_ph_max": 6.8,
        "tolerable_ph_min": 5.8,
        "tolerable_ph_max": 7.0,
        "water_requirement_level": WaterRequirementLevel.MEDIUM,
        "drainage_requirement": CropDrainageRequirement.MODERATE,
        "frost_sensitivity": CropSensitivityLevel.HIGH,
        "heat_sensitivity": CropSensitivityLevel.MEDIUM,
        "salinity_tolerance": CropPreferenceLevel.MODERATE,
        "rooting_depth_cm": 120.0,
        "slope_tolerance": 8.0,
        "optimal_temp_min_c": 18.0,
        "optimal_temp_max_c": 30.0,
        "rainfall_requirement_mm": 650.0,
        "frost_tolerance_days": 2,
        "heat_tolerance_days": 20,
        "organic_matter_preference": CropPreferenceLevel.MODERATE,
        "crop_price": CropPrice(price_per_ton=210.0),
        "input_cost": InputCost(fertilizer_cost=240.0, water_cost=165.0, labor_cost=130.0),
    }
    values.update(kwargs)
    return CropProfile(**values)


def make_field(name: str, **kwargs) -> Field:
    values = {
        "name": name,
        "location_name": f"{name} Zone",
        "latitude": 39.0,
        "longitude": -94.0,
        "area_hectares": 10.0,
        "slope_percent": 3.0,
        "irrigation_available": True,
        "drainage_quality": "good",
    }
    values.update(kwargs)
    return Field(**values)


def make_soil(field_id: int, **kwargs) -> SoilTest:
    values = {
        "field_id": field_id,
        "ph": 6.5,
        "organic_matter_percent": 4.0,
        "nitrogen_ppm": 50.0,
        "phosphorus_ppm": 35.0,
        "potassium_ppm": 220.0,
        "texture_class": "loamy",
        "depth_cm": 120.0,
        "ec": 1.2,
    }
    values.update(kwargs)
    return SoilTest(**values)


def make_weather(field_id: int, **kwargs) -> WeatherHistory:
    values = {
        "field_id": field_id,
        "date": datetime(2025, 12, 31, tzinfo=timezone.utc).date(),
        "min_temp": 10.0,
        "max_temp": 28.0,
        "avg_temp": 20.0,
        "rainfall_mm": 650.0,
        "humidity": 60.0,
        "wind_speed": 12.0,
        "solar_radiation": 18.0,
        "et0": 3.5,
    }
    values.update(kwargs)
    return WeatherHistory(**values)


def test_get_ranked_fields_response_ranks_all_fields_when_no_filter(db):
    crop = make_crop()
    field_a = make_field("Field Alpha", drainage_quality="excellent")
    field_b = make_field("Field Beta", irrigation_available=False, slope_percent=9.0)
    db.add_all([crop, field_a, field_b])
    db.commit()

    db.add_all([
        make_soil(field_a.id, ph=6.4, depth_cm=140.0, ec=1.0),
        make_soil(field_b.id, ph=5.1, depth_cm=60.0, ec=3.0, organic_matter_percent=1.0),
    ])
    db.commit()

    response = get_ranked_fields_response(db, crop_id=crop.id)

    assert response.total_fields_evaluated == 2
    assert response.crop.crop_name == "Corn"
    assert len(response.ranked_results) == 2
    assert response.schema_version == "ranking.v2"
    assert response.ranked_results[0].field_name == "Field Alpha"
    assert response.ranked_results[0].estimated_profit is not None
    assert response.ranked_results[0].estimated_revenue is not None
    assert response.ranked_results[0].estimated_cost is not None
    assert response.ranked_results[0].predicted_yield is not None
    assert response.ranked_results[0].predicted_yield_min is not None
    assert response.ranked_results[0].predicted_yield_max is not None
    assert response.ranked_results[0].predicted_yield_range is not None
    assert response.ranked_results[0].confidence_score is not None
    assert response.ranked_results[0].ranking_score == response.ranked_results[0].total_score
    assert response.ranked_results[0].metadata.provider_name == "rule_based"
    assert response.ranked_results[0].provider_metadata.explanation_provider.provider_name == "rule_based"
    assert response.ranked_results[0].explanation.short_explanation


def test_get_ranked_fields_response_respects_field_filter(db):
    crop = make_crop()
    field_a = make_field("Field Alpha")
    field_b = make_field("Field Beta")
    db.add_all([crop, field_a, field_b])
    db.commit()

    db.add_all([
        make_soil(field_a.id),
        make_soil(field_b.id),
    ])
    db.commit()

    response = get_ranked_fields_response(db, crop_id=crop.id, field_ids=[field_b.id])

    assert response.total_fields_evaluated == 1
    assert [entry.field_id for entry in response.ranked_results] == [field_b.id]


def test_get_ranked_fields_response_uses_latest_soil_test(db):
    crop = make_crop()
    field_obj = make_field("Field Alpha")
    db.add_all([crop, field_obj])
    db.commit()

    older_sample = datetime.now(timezone.utc) - timedelta(days=10)
    newer_sample = datetime.now(timezone.utc)
    db.add_all([
        make_soil(field_obj.id, ph=6.4, sample_date=older_sample),
        make_soil(field_obj.id, ph=4.9, sample_date=newer_sample),
    ])
    db.commit()

    response = get_ranked_fields_response(db, crop_id=crop.id, field_ids=[field_obj.id])

    assert response.ranked_results[0].total_score == 0.0
    assert any(
        blocker.code == "ph_far_outside_tolerable_range"
        for blocker in response.ranked_results[0].blockers
    )


def test_get_ranked_fields_response_raises_when_no_fields_found(db):
    crop = make_crop()
    db.add(crop)
    db.commit()

    with pytest.raises(NotFoundError, match="No fields found for ranking"):
        get_ranked_fields_response(db, crop_id=crop.id)


def test_get_ranked_fields_response_uses_climate_summary_for_ranking(db):
    crop = make_crop()
    field_a = make_field("Climate Alpha")
    field_b = make_field("Climate Beta")
    db.add_all([crop, field_a, field_b])
    db.commit()

    db.add_all([
        make_soil(field_a.id),
        make_soil(field_b.id),
        make_weather(field_a.id, avg_temp=24.0, rainfall_mm=650.0, min_temp=8.0, max_temp=30.0),
        make_weather(field_b.id, avg_temp=35.0, rainfall_mm=250.0, min_temp=-3.0, max_temp=40.0),
    ])
    db.commit()

    response = get_ranked_fields_response(db, crop_id=crop.id)

    assert [entry.field_name for entry in response.ranked_results] == ["Climate Alpha", "Climate Beta"]
    assert "climate_compatibility" in response.ranked_results[0].breakdown
    assert response.ranked_results[0].climate_score is not None
    assert response.ranked_results[0].climate_strengths
    assert response.ranked_results[0].climate_reasons
    assert isinstance(response.ranked_results[0].climate_warnings, list)
    assert response.ranked_results[1].climate_risks
    assert any("rainfall" in message.lower() or "frost" in message.lower() for message in response.ranked_results[1].weaknesses)
    assert response.ranked_results[0].total_score > response.ranked_results[1].total_score


def test_get_ranked_fields_response_uses_profitability_in_ranking_score(db):
    crop = make_crop()
    field_a = make_field("Profit Alpha", area_hectares=8.0)
    field_b = make_field("Profit Beta", area_hectares=22.0)
    db.add_all([crop, field_a, field_b])
    db.commit()

    db.add_all([
        make_soil(field_a.id, ph=6.5, depth_cm=120.0),
        make_soil(field_b.id, ph=6.5, depth_cm=120.0),
        make_weather(field_a.id, avg_temp=20.0, rainfall_mm=650.0),
        make_weather(field_b.id, avg_temp=20.0, rainfall_mm=650.0),
    ])
    db.commit()

    response = get_ranked_fields_response(db, crop_id=crop.id)

    assert response.ranked_results[0].field_name == "Profit Beta"
    assert response.ranked_results[0].economic_score >= response.ranked_results[1].economic_score


def test_get_ranked_fields_response_uses_registry_explanation_provider(db, monkeypatch):
    class _RegistryExplanationProvider:
        def __init__(self) -> None:
            self.requests = []

        def explain(self, request):
            self.requests.append(request)
            return ExplanationOutput(
                short_explanation="Registry explanation summary",
                detailed_explanation="Registry explanation detail.",
                strengths=["Strong registry-provided signal."],
                weaknesses=["No additional weakness recorded."],
                risks=["Registry-provided risk note."],
                confidence_note="Deterministic test confidence note.",
                metadata=AITraceMetadata(
                    provider_name="registry-test",
                    provider_version="v1",
                    generated_at=datetime.now(timezone.utc),
                    confidence=0.81,
                    debug_info={"source": "test"},
                ),
            )

    class _RecordingRegistry:
        def __init__(self, provider: _RegistryExplanationProvider) -> None:
            self._delegate = AIProviderRegistry(settings)
            self.provider = provider
            self.calls = 0
            self.settings = self._delegate.settings

        def get_explanation_provider(self):
            self.calls += 1
            return self.provider

        def get_suitability_provider(self):
            return self._delegate.get_suitability_provider()

        def get_ranking_augmentation_provider(self):
            return self._delegate.get_ranking_augmentation_provider()

    crop = make_crop()
    field_obj = make_field("Registry Field")
    db.add_all([crop, field_obj])
    db.commit()
    db.add(make_soil(field_obj.id))
    db.commit()

    provider = _RegistryExplanationProvider()
    registry = _RecordingRegistry(provider)
    monkeypatch.setattr("app.services.ranking_service.get_ai_provider_registry", lambda: registry)

    response = get_ranked_fields_response(db, crop_id=crop.id, field_ids=[field_obj.id])

    assert registry.calls == 1
    assert provider.requests
    assert response.ranked_results[0].explanation.short_explanation == "Registry explanation summary"
    assert response.ranked_results[0].explanation.risks == ["Registry-provided risk note."]
    assert response.ranked_results[0].strengths == ["Strong registry-provided signal."]
    assert response.ranked_results[0].metadata.provider_name == settings.AI_RANKING_AUGMENTATION_PROVIDER
    assert response.ranked_results[0].provider_metadata.explanation_provider.provider_name == "registry-test"
