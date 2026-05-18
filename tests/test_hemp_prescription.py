"""Tests for the hemp prescription engine and API endpoint."""

import pytest

from app.engines.hemp_prescription_engine import compute_hemp_prescription
from app.schemas.hemp_prescription import HempPrescriptionRequest


# ── Shared fixtures ────────────────────────────────────────────────────────────

GOOD_INPUTS = {
    "ph": 6.5,
    "nitrogen_ppm": 30.0,
    "phosphorus_ppm": 10.0,
    "potassium_ppm": 80.0,
    "organic_matter_percent": 2.5,
    "ec": 0.8,
    "drainage_class": "good",
    "texture_class": "loam",
    "area_hectares": 15.0,
    "slope_percent": 4.0,
    "irrigation_available": True,
    "elevation_meters": 250.0,
    "avg_temp": 20.0,
    "seasonal_rainfall_mm": 350.0,
    "avg_humidity": 60.0,
    "avg_solar_radiation": 17.0,
}


def make_request(**overrides) -> HempPrescriptionRequest:
    return HempPrescriptionRequest(**{**GOOD_INPUTS, **overrides})


# ── Engine unit tests (no model files needed — uses rule-based fallback) ───────

class TestHempPrescriptionEngine:
    def test_suitable_field_returns_non_negative_prescriptions(self):
        result = compute_hemp_prescription(make_request())
        assert result.suitable is True
        assert result.rec_nitrogen_kg_ha >= 0.0
        assert result.rec_phosphorus_kg_ha >= 0.0
        assert result.rec_potassium_kg_ha >= 0.0
        assert result.rec_irrigation_mm_week >= 0.0
        assert result.expected_yield_ton_ha > 0.0

    def test_high_soil_nutrients_produce_zero_prescription(self):
        # Very high N/P/K in soil → no supplemental fertilizer needed
        result = compute_hemp_prescription(
            make_request(nitrogen_ppm=40.0, phosphorus_ppm=20.0, potassium_ppm=35.0)
        )
        assert result.rec_nitrogen_kg_ha == 0.0
        assert result.rec_phosphorus_kg_ha == 0.0
        assert result.rec_potassium_kg_ha == 0.0

    def test_no_irrigation_available_returns_zero_irrigation(self):
        result = compute_hemp_prescription(make_request(irrigation_available=False))
        assert result.rec_irrigation_mm_week == 0.0

    def test_high_rainfall_reduces_irrigation_prescription(self):
        low_rain = compute_hemp_prescription(make_request(seasonal_rainfall_mm=150.0))
        high_rain = compute_hemp_prescription(make_request(seasonal_rainfall_mm=600.0))
        assert low_rain.rec_irrigation_mm_week > high_rain.rec_irrigation_mm_week

    def test_ph_too_low_fires_blocker(self):
        result = compute_hemp_prescription(make_request(ph=4.5))
        assert result.suitable is False
        assert len(result.blockers) > 0
        assert result.rec_nitrogen_kg_ha == 0.0

    def test_ph_too_high_fires_blocker(self):
        result = compute_hemp_prescription(make_request(ph=8.5))
        assert result.suitable is False

    def test_excessive_slope_fires_blocker(self):
        result = compute_hemp_prescription(make_request(slope_percent=25.0))
        assert result.suitable is False
        assert any("slope" in b.lower() for b in result.blockers)

    def test_high_ec_fires_blocker(self):
        result = compute_hemp_prescription(make_request(ec=5.0))
        assert result.suitable is False

    def test_extreme_temp_fires_blocker(self):
        result = compute_hemp_prescription(make_request(avg_temp=2.0))
        assert result.suitable is False

    def test_poor_drainage_generates_note(self):
        result = compute_hemp_prescription(make_request(drainage_class="poor"))
        assert result.suitable is True
        assert any("drainage" in n.lower() for n in result.notes)

    def test_low_forecast_yield_generates_note(self):
        # Poor pH + cold temp + poor drainage → low yield → note fires
        result = compute_hemp_prescription(
            make_request(ph=5.2, avg_temp=12.0, drainage_class="poor")
        )
        if result.suitable:
            assert any("yield" in n.lower() or "3 t/ha" in n for n in result.notes)

    def test_provider_field_is_populated(self):
        result = compute_hemp_prescription(make_request())
        assert result.provider in ("xgboost", "rule_based")


# ── API endpoint tests ─────────────────────────────────────────────────────────

class TestHempPrescriptionEndpoint:
    def test_valid_request_returns_200(self, client):
        response = client.post("/api/v1/hemp/prescription", json=GOOD_INPUTS)
        assert response.status_code == 200

    def test_response_contains_all_target_fields(self, client):
        response = client.post("/api/v1/hemp/prescription", json=GOOD_INPUTS)
        data = response.json()
        expected_keys = {
            "rec_nitrogen_kg_ha", "rec_phosphorus_kg_ha", "rec_potassium_kg_ha",
            "rec_irrigation_mm_week", "expected_yield_ton_ha",
            "suitable", "blockers", "notes", "provider",
        }
        assert expected_keys.issubset(data.keys())

    def test_invalid_drainage_class_returns_422(self, client):
        bad = {**GOOD_INPUTS, "drainage_class": "excellent_plus"}
        response = client.post("/api/v1/hemp/prescription", json=bad)
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self, client):
        bad = {k: v for k, v in GOOD_INPUTS.items() if k != "ph"}
        response = client.post("/api/v1/hemp/prescription", json=bad)
        assert response.status_code == 422

    def test_blocker_scenario_returns_suitable_false(self, client):
        blocked = {**GOOD_INPUTS, "ph": 4.0}
        response = client.post("/api/v1/hemp/prescription", json=blocked)
        assert response.status_code == 200
        assert response.json()["suitable"] is False
        assert len(response.json()["blockers"]) > 0

    def test_all_prescription_values_non_negative(self, client):
        response = client.post("/api/v1/hemp/prescription", json=GOOD_INPUTS)
        data = response.json()
        for field in ("rec_nitrogen_kg_ha", "rec_phosphorus_kg_ha",
                      "rec_potassium_kg_ha", "rec_irrigation_mm_week",
                      "expected_yield_ton_ha"):
            assert data[field] >= 0.0, f"{field} was negative"
