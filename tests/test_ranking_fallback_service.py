from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import Boolean, Column, Date, DateTime, Float, MetaData, String, Table, Text, create_engine
from sqlalchemy.orm import sessionmaker

from app.services.ranking_service import get_ranked_fields_response


def _build_legacy_like_schema(metadata: MetaData) -> tuple[Table, Table, Table]:
    fields = Table(
        "fields",
        metadata,
        Column("id", String, primary_key=True),
        Column("name", Text, nullable=False),
        Column("location_name", Text),
        Column("latitude", Float),
        Column("longitude", Float),
        Column("area_hectares", Float),
        Column("elevation_meters", Float),
        Column("slope_percent", Float),
        Column("aspect", Text),
        Column("irrigation_available", Boolean, nullable=False),
        Column("water_source_type", Text),
        Column("infrastructure_score", Float),
        Column("notes", Text),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
    )
    crop_profiles = Table(
        "crop_profiles",
        metadata,
        Column("id", String, primary_key=True),
        Column("crop_name", Text, nullable=False),
        Column("scientific_name", Text),
        Column("ideal_ph_min", Float),
        Column("ideal_ph_max", Float),
        Column("tolerable_ph_min", Float),
        Column("tolerable_ph_max", Float),
        Column("water_requirement_level", Text),
        Column("drainage_requirement", Text),
        Column("frost_sensitivity", Text),
        Column("heat_sensitivity", Text),
        Column("salinity_tolerance", Text),
        Column("rooting_depth_cm", Float),
        Column("slope_tolerance", Text),
        Column("organic_matter_preference", Text),
        Column("notes", Text),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
    )
    soil_tests = Table(
        "soil_tests",
        metadata,
        Column("id", String, primary_key=True),
        Column("field_id", String, nullable=False),
        Column("sample_date", Date, nullable=False),
        Column("ph", Float),
        Column("ec", Float),
        Column("organic_matter_percent", Float),
        Column("nitrogen_ppm", Float),
        Column("phosphorus_ppm", Float),
        Column("potassium_ppm", Float),
        Column("calcium_ppm", Float),
        Column("magnesium_ppm", Float),
        Column("texture_class", Text),
        Column("drainage_class", Text),
        Column("depth_cm", Float),
        Column("water_holding_capacity", Float),
        Column("notes", Text),
        Column("created_at", DateTime, nullable=False),
    )
    return fields, crop_profiles, soil_tests


def _build_legacy_weather_history(metadata: MetaData) -> Table:
    return Table(
        "weather_history",
        metadata,
        Column("id", String, primary_key=True),
        Column("field_id", String, nullable=False),
        Column("weather_date", Date, nullable=False),
        Column("min_temp", Float),
        Column("max_temp", Float),
        Column("avg_temp", Float),
        Column("rainfall_mm", Float),
        Column("humidity", Float),
        Column("wind_speed", Float),
        Column("solar_radiation", Float),
        Column("created_at", DateTime, nullable=False),
    )


def test_ranking_service_falls_back_to_legacy_schema_support():
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    fields, crop_profiles, soil_tests = _build_legacy_like_schema(metadata)
    metadata.create_all(engine)
    SessionTesting = sessionmaker(bind=engine)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    with SessionTesting() as session:
        session.execute(fields.insert(), [
            {
                "id": "field-1",
                "name": "Legacy Alpha",
                "location_name": "Zone A",
                "latitude": 39.1,
                "longitude": -94.5,
                "area_hectares": 16.0,
                "elevation_meters": 220.0,
                "slope_percent": 2.0,
                "aspect": "south",
                "irrigation_available": True,
                "water_source_type": "well",
                "infrastructure_score": 75.0,
                "notes": "legacy alpha",
                "created_at": now,
                "updated_at": now,
            },
            {
                "id": "field-2",
                "name": "Legacy Beta",
                "location_name": "Zone B",
                "latitude": 39.3,
                "longitude": -94.8,
                "area_hectares": 12.0,
                "elevation_meters": 180.0,
                "slope_percent": 8.0,
                "aspect": "west",
                "irrigation_available": False,
                "water_source_type": "none",
                "infrastructure_score": 61.0,
                "notes": "legacy beta",
                "created_at": now,
                "updated_at": now,
            },
        ])
        session.execute(crop_profiles.insert().values(
            id="crop-1",
            crop_name="Corn",
            scientific_name="Zea mays",
            ideal_ph_min=6.0,
            ideal_ph_max=6.8,
            tolerable_ph_min=5.8,
            tolerable_ph_max=7.2,
            water_requirement_level="high",
            drainage_requirement="moderate",
            frost_sensitivity="high",
            heat_sensitivity="medium",
            salinity_tolerance="moderate",
            rooting_depth_cm=150.0,
            slope_tolerance="6.0",
            organic_matter_preference="moderate",
            notes="legacy crop",
            created_at=now,
            updated_at=now,
        ))
        session.execute(soil_tests.insert().values(
            id="soil-1",
            field_id="field-1",
            sample_date=date(2026, 3, 20),
            ph=6.4,
            ec=0.9,
            organic_matter_percent=3.8,
            nitrogen_ppm=48.0,
            phosphorus_ppm=24.0,
            potassium_ppm=220.0,
            calcium_ppm=1700.0,
            magnesium_ppm=190.0,
            texture_class="loam",
            drainage_class="good",
            depth_cm=125.0,
            water_holding_capacity=23.0,
            notes="legacy soil",
            created_at=now,
        ))
        session.commit()

        response = get_ranked_fields_response(session, crop_id="crop-1")

        assert response.total_fields_evaluated == 2
        assert response.crop.id == "crop-1"
        assert response.ranked_results[0].field_id == "field-1"
        assert response.ranked_results[0].ranking_score >= response.ranked_results[1].ranking_score
        assert any(blocker.code == "missing_soil_test" for blocker in response.ranked_results[1].blockers)


def test_ranking_service_fallback_uses_weather_history_climate_signals():
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    fields, crop_profiles, soil_tests = _build_legacy_like_schema(metadata)
    weather_history = _build_legacy_weather_history(metadata)
    metadata.create_all(engine)
    SessionTesting = sessionmaker(bind=engine)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    with SessionTesting() as session:
        session.execute(fields.insert(), [
            {
                "id": "field-1",
                "name": "Legacy Alpha",
                "location_name": "Zone A",
                "latitude": 39.1,
                "longitude": -94.5,
                "area_hectares": 16.0,
                "elevation_meters": 220.0,
                "slope_percent": 2.0,
                "aspect": "south",
                "irrigation_available": True,
                "water_source_type": "well",
                "infrastructure_score": 75.0,
                "notes": "legacy alpha",
                "created_at": now,
                "updated_at": now,
            },
            {
                "id": "field-2",
                "name": "Legacy Beta",
                "location_name": "Zone B",
                "latitude": 39.3,
                "longitude": -94.8,
                "area_hectares": 12.0,
                "elevation_meters": 180.0,
                "slope_percent": 2.0,
                "aspect": "west",
                "irrigation_available": True,
                "water_source_type": "none",
                "infrastructure_score": 61.0,
                "notes": "legacy beta",
                "created_at": now,
                "updated_at": now,
            },
        ])
        session.execute(crop_profiles.insert().values(
            id="crop-1",
            crop_name="Corn",
            scientific_name="Zea mays",
            ideal_ph_min=6.0,
            ideal_ph_max=6.8,
            tolerable_ph_min=5.8,
            tolerable_ph_max=7.2,
            water_requirement_level="high",
            drainage_requirement="moderate",
            frost_sensitivity="high",
            heat_sensitivity="medium",
            salinity_tolerance="moderate",
            rooting_depth_cm=150.0,
            slope_tolerance="6.0",
            organic_matter_preference="moderate",
            notes="legacy crop",
            created_at=now,
            updated_at=now,
        ))
        session.execute(soil_tests.insert(), [
            {
                "id": "soil-1",
                "field_id": "field-1",
                "sample_date": date(2026, 3, 20),
                "ph": 6.4,
                "ec": 0.9,
                "organic_matter_percent": 3.8,
                "nitrogen_ppm": 48.0,
                "phosphorus_ppm": 24.0,
                "potassium_ppm": 220.0,
                "calcium_ppm": 1700.0,
                "magnesium_ppm": 190.0,
                "texture_class": "loam",
                "drainage_class": "good",
                "depth_cm": 125.0,
                "water_holding_capacity": 23.0,
                "notes": "legacy soil",
                "created_at": now,
            },
            {
                "id": "soil-2",
                "field_id": "field-2",
                "sample_date": date(2026, 3, 20),
                "ph": 6.4,
                "ec": 0.9,
                "organic_matter_percent": 3.8,
                "nitrogen_ppm": 48.0,
                "phosphorus_ppm": 24.0,
                "potassium_ppm": 220.0,
                "calcium_ppm": 1700.0,
                "magnesium_ppm": 190.0,
                "texture_class": "loam",
                "drainage_class": "good",
                "depth_cm": 125.0,
                "water_holding_capacity": 23.0,
                "notes": "legacy soil",
                "created_at": now,
            },
        ])
        session.execute(weather_history.insert(), [
            {
                "id": "weather-1",
                "field_id": "field-1",
                "weather_date": date(2026, 3, 20),
                "min_temp": 9.0,
                "max_temp": 28.0,
                "avg_temp": 21.0,
                "rainfall_mm": 95.0,
                "humidity": 60.0,
                "wind_speed": 4.5,
                "solar_radiation": 18.0,
                "created_at": now,
            },
            {
                "id": "weather-2",
                "field_id": "field-2",
                "weather_date": date(2026, 3, 20),
                "min_temp": -4.0,
                "max_temp": 39.0,
                "avg_temp": 34.0,
                "rainfall_mm": 15.0,
                "humidity": 40.0,
                "wind_speed": 7.5,
                "solar_radiation": 21.0,
                "created_at": now,
            },
        ])
        session.commit()

        response = get_ranked_fields_response(session, crop_id="crop-1")

        assert [entry.field_id for entry in response.ranked_results] == ["field-1", "field-2"]
        assert response.ranked_results[0].climate_score is not None
        assert response.ranked_results[0].climate_strengths
        assert isinstance(response.ranked_results[0].climate_warnings, list)
        assert response.ranked_results[1].climate_risks
