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
