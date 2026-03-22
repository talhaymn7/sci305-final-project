from datetime import date

from app.models.weather_history import WeatherHistory


def test_get_recommendation_returns_climate_aware_payload(client, db, created_field):
    soil_response = client.post(
        "/api/v1/soil-tests/",
        json={
            "field_id": created_field["id"],
            "sample_date": "2026-03-15T10:00:00Z",
            "ph": 6.5,
            "ec": 0.8,
            "nitrogen_ppm": 45.0,
            "phosphorus_ppm": 30.0,
            "potassium_ppm": 200.0,
            "calcium_ppm": 1700.0,
            "magnesium_ppm": 210.0,
            "organic_matter_percent": 3.5,
            "texture_class": "loamy",
            "drainage_class": "good",
            "depth_cm": 30.0,
            "water_holding_capacity": 21.5,
            "notes": "Baseline sample",
        },
    )
    assert soil_response.status_code == 201

    crop = client.post(
        "/api/v1/crops/",
        json={
            "crop_name": "Corn",
            "scientific_name": "Zea mays",
            "ideal_ph_min": 6.0,
            "ideal_ph_max": 6.8,
            "tolerable_ph_min": 5.8,
            "tolerable_ph_max": 7.0,
            "water_requirement_level": "medium",
            "drainage_requirement": "moderate",
            "frost_sensitivity": "high",
            "heat_sensitivity": "medium",
            "salinity_tolerance": "moderate",
            "rooting_depth_cm": 100.0,
            "slope_tolerance": 10.0,
            "optimal_temp_min_c": 18.0,
            "optimal_temp_max_c": 30.0,
            "preferred_rainfall_min_mm": 500.0,
            "preferred_rainfall_max_mm": 700.0,
            "frost_tolerance_days": 1,
            "heat_tolerance_days": 1,
            "organic_matter_preference": "moderate",
        },
    ).json()

    db.add_all(
        [
            WeatherHistory(
                field_id=created_field["id"],
                date=date(2026, 3, 1),
                min_temp=-2.0,
                max_temp=36.0,
                avg_temp=15.0,
                rainfall_mm=2.0,
                humidity=51.0,
                wind_speed=4.5,
                solar_radiation=15.0,
                et0=2.7,
            ),
            WeatherHistory(
                field_id=created_field["id"],
                date=date(2026, 3, 2),
                min_temp=-1.0,
                max_temp=37.0,
                avg_temp=16.0,
                rainfall_mm=1.0,
                humidity=53.0,
                wind_speed=4.8,
                solar_radiation=16.0,
                et0=2.9,
            ),
            WeatherHistory(
                field_id=created_field["id"],
                date=date(2026, 3, 3),
                min_temp=0.0,
                max_temp=38.0,
                avg_temp=17.0,
                rainfall_mm=0.0,
                humidity=55.0,
                wind_speed=5.0,
                solar_radiation=17.0,
                et0=3.0,
            ),
        ]
    )
    db.commit()

    response = client.get(f"/api/v1/recommendation/{created_field['id']}/{crop['id']}?climate_days=7")

    assert response.status_code == 200
    payload = response.json()
    assert payload["field_id"] == created_field["id"]
    assert payload["crop_id"] == crop["id"]
    assert payload["score"] == payload["total_score"]
    assert isinstance(payload["agronomic_score"], float)
    assert isinstance(payload["climate_score"], float)
    assert isinstance(payload["reasons"], list)
    assert isinstance(payload["climate_reasons"], list)
    assert isinstance(payload["climate_warnings"], list)
    assert isinstance(payload["climate_strengths"], list)
    assert isinstance(payload["climate_weaknesses"], list)
    assert isinstance(payload["climate_risks"], list)
    assert isinstance(payload["strengths"], list)
    assert isinstance(payload["weaknesses"], list)
    assert isinstance(payload["risks"], list)
    assert isinstance(payload["blockers"], list)
    assert "Climate:" in payload["explanation"]
    assert any("rainfall" in message.lower() for message in payload["climate_reasons"])
    assert any("frost" in message.lower() for message in payload["climate_warnings"])
    assert any("heat" in message.lower() for message in payload["climate_warnings"])


def test_get_recommendation_returns_404_for_missing_crop(client, created_field):
    response = client.get(f"/api/v1/recommendation/{created_field['id']}/9999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Crop not found"
