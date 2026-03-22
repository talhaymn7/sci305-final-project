from datetime import date

from app.models.crop_profile import CropProfile
from app.models.enums import (
    CropDrainageRequirement,
    CropPreferenceLevel,
    CropSensitivityLevel,
    WaterRequirementLevel,
)
from app.models.field import Field
from app.models.weather_history import WeatherHistory
from app.services.climate_feature_builder import ClimateFeatureBuilder, ClimateObservation


def test_build_summary_computes_recent_climate_aggregates():
    builder = ClimateFeatureBuilder()

    summary = builder.build_summary(
        [
            ClimateObservation(
                observation_date=date(2026, 3, 1),
                min_temp=-1.0,
                max_temp=16.0,
                avg_temp=8.0,
                rainfall_mm=12.0,
                humidity=64.0,
                wind_speed=4.0,
                solar_radiation=14.0,
            ),
            ClimateObservation(
                observation_date=date(2026, 3, 2),
                min_temp=7.0,
                max_temp=36.0,
                avg_temp=21.0,
                rainfall_mm=4.5,
                humidity=58.0,
                wind_speed=6.0,
                solar_radiation=19.0,
            ),
        ],
        lookback_days=30,
        heat_day_threshold=35.0,
    )

    assert summary is not None
    assert summary.avg_temp == 14.5
    assert summary.min_observed_temp == -1.0
    assert summary.max_observed_temp == 36.0
    assert summary.total_rainfall == 16.5
    assert summary.avg_min_temp == 3.0
    assert summary.avg_max_temp == 26.0
    assert summary.avg_humidity == 61.0
    assert summary.avg_wind_speed == 5.0
    assert summary.avg_solar_radiation == 16.5
    assert summary.frost_days == 1
    assert summary.heat_days == 1
    assert summary.weather_record_count == 2
    assert summary.observation_days_count == 2
    assert summary.missing_days_count == 28
    assert summary.coverage_ratio == 0.0667


def test_observation_from_mapping_supports_weather_date_column():
    observation = ClimateFeatureBuilder.observation_from_mapping(
        {
            "weather_date": date(2026, 3, 20),
            "min_temp": 5,
            "max_temp": 15,
            "avg_temp": 10,
            "rainfall_mm": 2.5,
            "humidity": 70,
            "wind_speed": 3.5,
            "solar_radiation": 12.5,
        },
        date_column_name="weather_date",
    )

    assert observation is not None
    assert observation.observation_date == date(2026, 3, 20)
    assert observation.avg_temp == 10.0


def test_build_climate_summary_reads_persisted_weather_history(db):
    field = Field(
        name="Builder Field",
        location_name="Builder Zone",
        latitude=39.0,
        longitude=-94.0,
        area_hectares=10.0,
        slope_percent=2.0,
        irrigation_available=True,
        drainage_quality="good",
    )
    db.add(field)
    db.flush()
    db.add_all(
        [
            WeatherHistory(
                field_id=field.id,
                date=date(2026, 3, 1),
                min_temp=4.0,
                max_temp=18.0,
                avg_temp=11.0,
                rainfall_mm=3.0,
                humidity=58.0,
                wind_speed=4.0,
                solar_radiation=14.0,
                et0=2.0,
            ),
            WeatherHistory(
                field_id=field.id,
                date=date(2026, 3, 3),
                min_temp=-2.0,
                max_temp=37.0,
                avg_temp=17.0,
                rainfall_mm=6.0,
                humidity=62.0,
                wind_speed=6.0,
                solar_radiation=18.0,
                et0=3.0,
            ),
        ]
    )
    db.commit()

    summary = ClimateFeatureBuilder(db).build_climate_summary(field.id, days=7)

    assert summary is not None
    assert summary.avg_temp == 14.0
    assert summary.avg_min_temp == 1.0
    assert summary.avg_max_temp == 27.5
    assert summary.total_rainfall == 9.0
    assert summary.total_et0 == 5.0
    assert summary.observation_days_count == 2
    assert summary.missing_days_count == 5


def test_build_climate_features_returns_structured_scoring_inputs(db):
    field = Field(
        name="Feature Field",
        location_name="Feature Zone",
        latitude=39.0,
        longitude=-94.0,
        area_hectares=10.0,
        slope_percent=2.0,
        irrigation_available=True,
        drainage_quality="good",
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
        preferred_rainfall_min_mm=80.0,
        preferred_rainfall_max_mm=140.0,
        frost_tolerance_days=1,
        heat_tolerance_days=4,
        organic_matter_preference=CropPreferenceLevel.MODERATE,
    )
    db.add_all([field, crop])
    db.flush()
    db.add_all(
        [
            WeatherHistory(
                field_id=field.id,
                date=date(2026, 3, 1),
                min_temp=-1.0,
                max_temp=36.0,
                avg_temp=16.0,
                rainfall_mm=20.0,
                humidity=60.0,
                wind_speed=4.0,
                solar_radiation=14.0,
                et0=2.0,
            ),
            WeatherHistory(
                field_id=field.id,
                date=date(2026, 3, 2),
                min_temp=1.0,
                max_temp=38.0,
                avg_temp=19.0,
                rainfall_mm=15.0,
                humidity=61.0,
                wind_speed=5.0,
                solar_radiation=15.0,
                et0=2.4,
            ),
        ]
    )
    db.commit()

    features = ClimateFeatureBuilder(db).build_climate_features(field.id, crop.id, days=7)

    assert features.field_id == field.id
    assert features.crop_id == crop.id
    assert features.summary is not None
    assert features.requirements.optimal_temp_min_c == 18.0
    assert features.avg_temp_gap_to_ideal_c == 0.5
    assert features.rainfall_gap_to_preferred_mm == 45.0
    assert features.frost_excess_days == 0
    assert features.heat_excess_days == 0
    assert features.temperature_within_ideal_range is False
    assert features.rainfall_within_preferred_range is False
