# tests/test_hemp_prescription.py
import pytest
from pydantic import ValidationError
from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult
from app.schemas.hemp_cycle import CycleRecord, AdaptivePrescriptionRequest


def test_request_requires_area_dekar():
    with pytest.raises(ValidationError):
        HempPrescriptionRequest(slope_percent=5.0)  # missing area_dekar


def test_request_soil_chemistry_optional():
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)
    assert req.ph is None
    assert req.nitrogen_ppm is None


def test_request_defaults():
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)
    assert req.variety == "other"
    assert req.irrigation_type == "none"
    assert req.previous_crop == "other"
    assert req.first_hemp_season is False


def test_request_valid_variety():
    req = HempPrescriptionRequest(area_dekar=30.0, slope_percent=3.0, variety="narli")
    assert req.variety == "narli"


def test_request_invalid_variety():
    with pytest.raises(ValidationError):
        HempPrescriptionRequest(area_dekar=30.0, slope_percent=3.0, variety="unknown_variety")


def test_request_valid_irrigation_type():
    req = HempPrescriptionRequest(area_dekar=30.0, slope_percent=3.0, irrigation_type="drip")
    assert req.irrigation_type == "drip"


def test_result_has_confidence():
    result = HempPrescriptionResult(
        rec_nitrogen_kg_dekar=8.0, rec_phosphorus_kg_dekar=3.0,
        rec_potassium_kg_dekar=6.0, rec_irrigation_mm_week=5.0,
        expected_yield_ton_dekar=0.5, suitable=True, provider="xgboost",
    )
    assert result.confidence == 1.0  # default


def test_result_confidence_range():
    with pytest.raises(ValidationError):
        HempPrescriptionResult(
            rec_nitrogen_kg_dekar=8.0, rec_phosphorus_kg_dekar=3.0,
            rec_potassium_kg_dekar=6.0, rec_irrigation_mm_week=5.0,
            expected_yield_ton_dekar=0.5, suitable=True, provider="x",
            confidence=1.5,  # out of range
        )


def test_cycle_record_requires_all_fields():
    with pytest.raises(ValidationError):
        CycleRecord(cycle_id="c1")  # missing most fields


def test_cycle_record_valid():
    cycle = CycleRecord(
        cycle_id="c1",
        predicted_yield_ton_dekar=0.5, rec_nitrogen_kg_dekar=8.0,
        rec_phosphorus_kg_dekar=3.0, rec_potassium_kg_dekar=6.0,
        rec_irrigation_mm_week=5.0, applied_nitrogen_kg_dekar=7.5,
        applied_phosphorus_kg_dekar=3.0, applied_potassium_kg_dekar=6.0,
        applied_irrigation_mm_season=90.0, actual_yield_ton_dekar=0.52,
        actual_moisture_percent=12.0, cycle_days=120,
    )
    assert cycle.cycle_id == "c1"


def test_adaptive_request_no_history():
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)
    body = AdaptivePrescriptionRequest(request=req)
    assert body.cycle_history == []


def test_adaptive_request_with_history():
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)
    cycle = CycleRecord(
        cycle_id="c1", predicted_yield_ton_dekar=0.5, rec_nitrogen_kg_dekar=8.0,
        rec_phosphorus_kg_dekar=3.0, rec_potassium_kg_dekar=6.0,
        rec_irrigation_mm_week=5.0, applied_nitrogen_kg_dekar=7.5,
        applied_phosphorus_kg_dekar=3.0, applied_potassium_kg_dekar=6.0,
        applied_irrigation_mm_season=90.0, actual_yield_ton_dekar=0.55,
        actual_moisture_percent=12.0, cycle_days=120,
    )
    body = AdaptivePrescriptionRequest(request=req, cycle_history=[cycle])
    assert len(body.cycle_history) == 1


# ── Cold-start provider tests ──────────────────────────────────────────────────
from app.ai.providers.ml.hemp_prescription import (
    HempPrescriptionProvider, _ph_penalty, _temp_penalty,
    ROTATION_N_CREDIT, IRRIGATION_EFFICIENCY, VARIETY_YIELD_FACTOR,
)


def _base_request(**kwargs) -> HempPrescriptionRequest:
    defaults = dict(
        area_dekar=50.0, slope_percent=5.0, ph=6.5, nitrogen_ppm=15.0,
        phosphorus_ppm=20.0, potassium_ppm=150.0, organic_matter_percent=1.0,
        ec=1.5, avg_temp=20.0, seasonal_rainfall_mm=380.0,
        avg_humidity=58.0, avg_solar_radiation=16.0,
    )
    defaults.update(kwargs)
    return HempPrescriptionRequest(**defaults)


def test_ph_penalty_optimal():
    assert _ph_penalty(6.5) == 1.0


def test_ph_penalty_below_optimal():
    penalty = _ph_penalty(5.0)
    assert 0.0 < penalty < 1.0


def test_ph_penalty_above_optimal():
    penalty = _ph_penalty(8.0)
    assert 0.0 < penalty < 1.0


def test_temp_penalty_optimal():
    assert _temp_penalty(20.0) == 1.0


def test_temp_penalty_too_cold():
    assert _temp_penalty(5.0) < 1.0


def test_rotation_n_credit_legume():
    assert ROTATION_N_CREDIT["legume"] > 0.0


def test_rotation_n_credit_hemp_negative():
    assert ROTATION_N_CREDIT["hemp"] < 0.0


def test_irrigation_efficiency_drip_highest():
    assert IRRIGATION_EFFICIENCY["drip"] > IRRIGATION_EFFICIENCY["sprinkler"]
    assert IRRIGATION_EFFICIENCY["sprinkler"] > IRRIGATION_EFFICIENCY["flood"]
    assert IRRIGATION_EFFICIENCY["none"] == 0.0


def test_provider_rule_based_no_irrigation():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req = _base_request(irrigation_type="none")
    result = provider.predict(req)
    assert result.rec_irrigation_mm_week == 0.0


def test_provider_rule_based_drip_prescribes_irrigation():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req = _base_request(irrigation_type="drip", seasonal_rainfall_mm=200.0)
    result = provider.predict(req)
    assert result.rec_irrigation_mm_week > 0.0


def test_provider_rule_based_imputes_missing_soil():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)  # no soil chemistry
    result = provider.predict(req)
    assert result.suitable is True
    assert result.confidence < 0.75  # lower confidence when imputed


def test_provider_rule_based_legume_rotation_reduces_n():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req_other = _base_request(previous_crop="other")
    req_legume = _base_request(previous_crop="legume")
    n_other = provider.predict(req_other).rec_nitrogen_kg_dekar
    n_legume = provider.predict(req_legume).rec_nitrogen_kg_dekar
    assert n_legume < n_other


def test_provider_narli_variety_yields_more_than_other():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req_narli = _base_request(variety="narli")
    req_other = _base_request(variety="other")
    y_narli = provider.predict(req_narli).expected_yield_ton_dekar
    y_other = provider.predict(req_other).expected_yield_ton_dekar
    assert y_narli > y_other


def test_n_prescription_in_dekar_range():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req = _base_request(nitrogen_ppm=10.0, organic_matter_percent=1.0)
    result = provider.predict(req)
    assert 0.0 <= result.rec_nitrogen_kg_dekar <= 12.0


def test_yield_in_dekar_range():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req = _base_request()
    result = provider.predict(req)
    assert 0.0 <= result.expected_yield_ton_dekar <= 1.0
