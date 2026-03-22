"""Pydantic schemas for weather history input and climate summaries."""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class WeatherHistoryBase(BaseModel):
    """Shared weather-history attributes and validation rules."""

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )

    field_id: int | str | UUID | None = Field(default=None)
    date: date_type | None = Field(
        default=None,
        validation_alias=AliasChoices("date", "weather_date"),
    )
    min_temp: float | None = None
    max_temp: float | None = None
    avg_temp: float | None = None
    rainfall_mm: float | None = Field(default=None, ge=0)
    humidity: float | None = Field(default=None, ge=0, le=100)
    wind_speed: float | None = Field(default=None, ge=0)
    solar_radiation: float | None = Field(default=None, ge=0)
    et0: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_temperature_order(self) -> "WeatherHistoryBase":
        """Ensure min, average, and max temperatures follow a logical order."""

        if self.min_temp is None or self.avg_temp is None or self.max_temp is None:
            return self
        if self.min_temp > self.avg_temp or self.avg_temp > self.max_temp:
            raise ValueError("Temperature values must satisfy min_temp <= avg_temp <= max_temp.")
        return self


class WeatherHistoryCreate(WeatherHistoryBase):
    """Schema used when creating a weather history record."""

    field_id: int | str | UUID
    date: date_type
    min_temp: float
    max_temp: float
    avg_temp: float
    rainfall_mm: float = Field(..., ge=0)
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0)


class WeatherHistoryRead(BaseModel):
    """Schema returned for weather-history read operations."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    field_id: int | str | UUID
    date: date_type
    min_temp: float
    max_temp: float
    avg_temp: float
    rainfall_mm: float
    humidity: float
    wind_speed: float
    solar_radiation: float | None
    et0: float | None
    created_at: datetime


class ClimateSummary(BaseModel):
    """Aggregated climate metrics for a field over a recent time window."""

    model_config = ConfigDict(extra="forbid")

    field_id: int | str | UUID | None = None
    avg_avg_temp: float | None = None
    avg_temp: float | None = None
    avg_min_temp: float | None = None
    avg_max_temp: float | None = None
    min_observed_temp: float | None = None
    max_observed_temp: float | None = None
    total_rainfall_mm: float | None = None
    total_rainfall: float | None = None
    avg_humidity: float | None = None
    avg_wind_speed: float | None = None
    avg_solar_radiation: float | None = None
    total_et0: float | None = Field(default=None, ge=0)
    frost_days_count: int = 0
    frost_days: int = 0
    heat_days_count: int = 0
    heat_days: int = 0
    weather_record_count: int = 0
    observation_days_count: int = 0
    missing_days_count: int | None = Field(default=None, ge=0)
    lookback_days: int | None = None
    heat_threshold_c: float | None = None
    observation_start_date: date_type | None = None
    observation_end_date: date_type | None = None
    coverage_ratio: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def normalize_counts(self) -> "ClimateSummary":
        """Backfill derived count fields for backwards-compatible callers."""

        if self.avg_avg_temp is None and self.avg_temp is not None:
            self.avg_avg_temp = self.avg_temp
        if self.avg_temp is None and self.avg_avg_temp is not None:
            self.avg_temp = self.avg_avg_temp
        if self.total_rainfall_mm is None and self.total_rainfall is not None:
            self.total_rainfall_mm = self.total_rainfall
        if self.total_rainfall is None and self.total_rainfall_mm is not None:
            self.total_rainfall = self.total_rainfall_mm
        if self.frost_days_count <= 0 and self.frost_days > 0:
            self.frost_days_count = self.frost_days
        if self.frost_days <= 0 and self.frost_days_count > 0:
            self.frost_days = self.frost_days_count
        if self.heat_days_count <= 0 and self.heat_days > 0:
            self.heat_days_count = self.heat_days
        if self.heat_days <= 0 and self.heat_days_count > 0:
            self.heat_days = self.heat_days_count
        if self.observation_days_count <= 0 and self.weather_record_count > 0:
            self.observation_days_count = self.weather_record_count
        if self.weather_record_count <= 0 and self.observation_days_count > 0:
            self.weather_record_count = self.observation_days_count
        if self.lookback_days is not None and self.missing_days_count is None:
            self.missing_days_count = max(self.lookback_days - self.observation_days_count, 0)
        if self.lookback_days and self.coverage_ratio is None:
            self.coverage_ratio = round(
                min(self.observation_days_count / self.lookback_days, 1.0),
                4,
            )
        return self
