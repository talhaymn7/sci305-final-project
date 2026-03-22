"""Load field-scoped AI inputs and normalize them into shared feature summaries.

Example:
```python
from app.ai.features.loaders import assemble_feature_bundle

bundle = assemble_feature_bundle(db, field_id=1, crop_id=2)
print(bundle.summary.field.name)
print(bundle.summary.soil.ph if bundle.summary.soil is not None else None)
```
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.ai.features.types import (
    ClimateFeatureSummary,
    CropFeatureSummary,
    FeatureAssemblyBundle,
    FeatureSummaryBundle,
    FieldFeatureSummary,
    SoilFeatureSummary,
)
from app.models.crop_profile import CropProfile
from app.models.field import Field
from app.models.soil_test import SoilTest
from app.schemas.weather_history import ClimateSummary
from app.services.crop_climate_requirements import resolve_crop_climate_requirements
from app.services.crop_service import get_crop
from app.services.errors import NotFoundError
from app.services.field_service import get_field
from app.services.soil_service import get_latest_soil_test_for_field
from app.services.weather_service import WeatherService


def assemble_feature_bundle(
    db: Session,
    field_id: int,
    crop_id: int,
    *,
    weather_service: WeatherService | None = None,
) -> FeatureAssemblyBundle:
    """Load ORM entities once and return raw plus normalized feature summaries."""

    field_obj = get_field(db, field_id)
    if field_obj is None:
        raise NotFoundError(f"Field with id {field_id} not found")

    crop = get_crop(db, crop_id)
    if crop is None:
        raise NotFoundError(f"Crop with id {crop_id} not found")

    soil_test = get_latest_soil_test_for_field(db, field_id)
    weather = weather_service or WeatherService(db)
    climate_summary = weather.get_climate_summary(field_id)

    return FeatureAssemblyBundle(
        field_obj=field_obj,
        crop=crop,
        soil_test=soil_test,
        climate_summary=climate_summary,
        summary=build_feature_summary_bundle(
            field_obj,
            crop,
            soil_test=soil_test,
            climate_summary=climate_summary,
        ),
    )


def build_feature_summary_bundle(
    field_obj: Field,
    crop: CropProfile,
    *,
    soil_test: SoilTest | None = None,
    climate_summary: ClimateSummary | None = None,
) -> FeatureSummaryBundle:
    """Normalize raw entities into transport-friendly AI feature summaries."""

    return FeatureSummaryBundle(
        field=_build_field_summary(field_obj),
        soil=_build_soil_summary(soil_test),
        crop=_build_crop_summary(crop),
        climate=_build_climate_summary(climate_summary),
    )


def _build_field_summary(field_obj: Field) -> FieldFeatureSummary:
    return FieldFeatureSummary(
        field_id=getattr(field_obj, "id", None),
        name=str(getattr(field_obj, "name", "")),
        area_hectares=getattr(field_obj, "area_hectares", None),
        slope_percent=getattr(field_obj, "slope_percent", None),
        irrigation_available=getattr(field_obj, "irrigation_available", None),
        drainage_quality=getattr(field_obj, "drainage_quality", None),
        elevation_meters=getattr(field_obj, "elevation_meters", None),
        infrastructure_score=getattr(field_obj, "infrastructure_score", None),
        water_source_type=_enum_value(getattr(field_obj, "water_source_type", None)),
        aspect=_enum_value(getattr(field_obj, "aspect", None)),
    )


def _build_soil_summary(soil_test: SoilTest | None) -> SoilFeatureSummary | None:
    if soil_test is None:
        return None

    return SoilFeatureSummary(
        soil_test_id=getattr(soil_test, "id", None),
        ph=getattr(soil_test, "ph", None),
        ec=getattr(soil_test, "ec", None),
        organic_matter_percent=getattr(soil_test, "organic_matter_percent", None),
        nitrogen_ppm=getattr(soil_test, "nitrogen_ppm", None),
        phosphorus_ppm=getattr(soil_test, "phosphorus_ppm", None),
        potassium_ppm=getattr(soil_test, "potassium_ppm", None),
        calcium_ppm=getattr(soil_test, "calcium_ppm", None),
        magnesium_ppm=getattr(soil_test, "magnesium_ppm", None),
        depth_cm=getattr(soil_test, "depth_cm", None),
        water_holding_capacity=getattr(soil_test, "water_holding_capacity", None),
        texture_class=getattr(soil_test, "texture_class", None),
        drainage_class=getattr(soil_test, "drainage_class", None),
        sample_date=getattr(soil_test, "sample_date", None),
    )


def _build_crop_summary(crop: CropProfile) -> CropFeatureSummary:
    climate_requirements = resolve_crop_climate_requirements(crop)
    return CropFeatureSummary(
        crop_id=getattr(crop, "id", None),
        crop_name=str(getattr(crop, "crop_name", "")),
        ideal_ph_min=getattr(crop, "ideal_ph_min", None),
        ideal_ph_max=getattr(crop, "ideal_ph_max", None),
        water_requirement_level=_enum_value(getattr(crop, "water_requirement_level", None)),
        drainage_requirement=_enum_value(getattr(crop, "drainage_requirement", None)),
        frost_sensitivity=_enum_value(getattr(crop, "frost_sensitivity", None)),
        heat_sensitivity=_enum_value(getattr(crop, "heat_sensitivity", None)),
        salinity_tolerance=_enum_value(getattr(crop, "salinity_tolerance", None)),
        rooting_depth_cm=getattr(crop, "rooting_depth_cm", None),
        slope_tolerance=getattr(crop, "slope_tolerance", None),
        optimal_temp_min_c=climate_requirements.optimal_temp_min_c,
        optimal_temp_max_c=climate_requirements.optimal_temp_max_c,
        rainfall_requirement_mm=getattr(crop, "rainfall_requirement_mm", None),
        frost_tolerance_days=climate_requirements.frost_tolerance_days,
        heat_tolerance_days=climate_requirements.heat_tolerance_days,
        organic_matter_preference=_enum_value(getattr(crop, "organic_matter_preference", None)),
        tolerable_temp_min_c=climate_requirements.tolerable_temp_min_c,
        tolerable_temp_max_c=climate_requirements.tolerable_temp_max_c,
        preferred_rainfall_min_mm=climate_requirements.preferred_rainfall_min_mm,
        preferred_rainfall_max_mm=climate_requirements.preferred_rainfall_max_mm,
    )


def _build_climate_summary(climate_summary: ClimateSummary | None) -> ClimateFeatureSummary | None:
    if climate_summary is None:
        return None

    return ClimateFeatureSummary(
        avg_temp=climate_summary.avg_temp,
        avg_min_temp=climate_summary.avg_min_temp,
        avg_max_temp=climate_summary.avg_max_temp,
        total_rainfall=climate_summary.total_rainfall,
        frost_days=climate_summary.frost_days,
        heat_days=climate_summary.heat_days,
        min_observed_temp=climate_summary.min_observed_temp,
        max_observed_temp=climate_summary.max_observed_temp,
        avg_humidity=climate_summary.avg_humidity,
        avg_wind_speed=climate_summary.avg_wind_speed,
        avg_solar_radiation=climate_summary.avg_solar_radiation,
        total_et0=climate_summary.total_et0,
        weather_record_count=climate_summary.weather_record_count,
        observation_days_count=climate_summary.observation_days_count,
        missing_days_count=climate_summary.missing_days_count,
        lookback_days=climate_summary.lookback_days,
        heat_threshold_c=climate_summary.heat_threshold_c,
        coverage_ratio=climate_summary.coverage_ratio,
    )


def _enum_value(value: object | None) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", value)
