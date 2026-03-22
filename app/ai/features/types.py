"""Shared normalized feature summaries for AI input assembly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.models.crop_profile import CropProfile
from app.models.field import Field
from app.models.soil_test import SoilTest
from app.schemas.weather_history import ClimateSummary


@dataclass(slots=True)
class FieldFeatureSummary:
    """Normalized field summary shared across AI feature builders."""

    field_id: int | str | UUID | None
    name: str
    area_hectares: float | None
    slope_percent: float | None
    irrigation_available: bool | None
    drainage_quality: str | None
    elevation_meters: float | None
    infrastructure_score: int | None
    water_source_type: str | None
    aspect: str | None


@dataclass(slots=True)
class SoilFeatureSummary:
    """Normalized latest-soil summary shared across AI feature builders."""

    soil_test_id: int | str | UUID | None
    ph: float | None
    ec: float | None
    organic_matter_percent: float | None
    nitrogen_ppm: float | None
    phosphorus_ppm: float | None
    potassium_ppm: float | None
    calcium_ppm: float | None
    magnesium_ppm: float | None
    depth_cm: float | None
    water_holding_capacity: float | None
    texture_class: str | None
    drainage_class: str | None
    sample_date: datetime | None


@dataclass(slots=True)
class CropFeatureSummary:
    """Normalized crop summary shared across AI feature builders."""

    crop_id: int | str | UUID | None
    crop_name: str
    ideal_ph_min: float | None
    ideal_ph_max: float | None
    water_requirement_level: str | None
    drainage_requirement: str | None
    frost_sensitivity: str | None
    heat_sensitivity: str | None
    salinity_tolerance: str | None
    rooting_depth_cm: float | None
    slope_tolerance: float | None
    optimal_temp_min_c: float | None
    optimal_temp_max_c: float | None
    rainfall_requirement_mm: float | None
    frost_tolerance_days: int | None
    heat_tolerance_days: int | None
    organic_matter_preference: str | None
    tolerable_temp_min_c: float | None = None
    tolerable_temp_max_c: float | None = None
    preferred_rainfall_min_mm: float | None = None
    preferred_rainfall_max_mm: float | None = None


@dataclass(slots=True)
class ClimateFeatureSummary:
    """Normalized climate summary shared across AI feature builders."""

    avg_temp: float | None
    avg_min_temp: float | None
    avg_max_temp: float | None
    total_rainfall: float | None
    frost_days: int | None
    heat_days: int | None
    min_observed_temp: float | None = None
    max_observed_temp: float | None = None
    avg_humidity: float | None = None
    avg_wind_speed: float | None = None
    avg_solar_radiation: float | None = None
    total_et0: float | None = None
    weather_record_count: int | None = None
    observation_days_count: int | None = None
    missing_days_count: int | None = None
    lookback_days: int | None = None
    heat_threshold_c: float | None = None
    coverage_ratio: float | None = None


@dataclass(slots=True)
class FeatureSummaryBundle:
    """Collection of normalized summaries used to assemble provider inputs."""

    field: FieldFeatureSummary
    soil: SoilFeatureSummary | None
    crop: CropFeatureSummary
    climate: ClimateFeatureSummary | None


@dataclass(slots=True)
class FeatureAssemblyBundle:
    """Loaded raw entities plus their normalized summaries."""

    field_obj: Field
    crop: CropProfile
    soil_test: SoilTest | None
    climate_summary: ClimateSummary | None
    summary: FeatureSummaryBundle
