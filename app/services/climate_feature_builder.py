"""Build reusable climate summaries and scoring inputs from weather history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.schemas.weather_history import ClimateSummary
from app.services.climate_summary_service import ClimateSummaryService
from app.services.crop_climate_requirements import (
    CropClimateRequirements,
    resolve_crop_climate_requirements,
)
from app.services.crop_service import get_crop
from app.services.errors import NotFoundError


@dataclass(frozen=True, slots=True)
class ClimateObservation:
    """Normalized daily weather observation used by the climate summary builder."""

    observation_date: date
    min_temp: float | None = None
    max_temp: float | None = None
    avg_temp: float | None = None
    rainfall_mm: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    solar_radiation: float | None = None
    et0: float | None = None


@dataclass(frozen=True, slots=True)
class ClimateFeatures:
    """Transport-friendly climate feature payload used by scoring and AI layers."""

    field_id: int | str | UUID
    crop_id: int | str | UUID
    summary: ClimateSummary | None
    requirements: CropClimateRequirements
    avg_temp_gap_to_ideal_c: float | None
    rainfall_gap_to_preferred_mm: float | None
    frost_excess_days: int | None
    heat_excess_days: int | None
    temperature_within_ideal_range: bool | None
    rainfall_within_preferred_range: bool | None
    observation_days_count: int
    missing_days_count: int | None
    requirement_source: str


class ClimateFeatureBuilder:
    """Build climate summaries and feature bundles without exposing ORM rows."""

    def __init__(
        self,
        db: Session | None = None,
        *,
        summary_service: ClimateSummaryService | None = None,
    ) -> None:
        self.db = db
        self.summary_service = summary_service or (
            ClimateSummaryService(db) if db is not None else None
        )

    def build_climate_summary(
        self,
        field_id: int | str | UUID,
        *,
        days: int = 30,
        heat_threshold_c: float | None = None,
    ) -> ClimateSummary | None:
        """Return a persisted climate summary for a field and lookback window."""

        if self.summary_service is None:
            raise ValueError("ClimateFeatureBuilder requires a database session for build_climate_summary().")
        return self.summary_service.get_field_summary(
            field_id,
            days=days,
            heat_threshold_c=heat_threshold_c,
        )

    def build_climate_summaries(
        self,
        field_ids: Sequence[int | str | UUID],
        *,
        days: int = 30,
        heat_threshold_c: float | None = None,
    ) -> dict[int | str | UUID, ClimateSummary | None]:
        """Return persisted climate summaries keyed by field id."""

        if self.summary_service is None:
            raise ValueError("ClimateFeatureBuilder requires a database session for build_climate_summaries().")
        return self.summary_service.get_field_summaries(
            field_ids,
            days=days,
            heat_threshold_c=heat_threshold_c,
        )

    def build_climate_features(
        self,
        field_id: int | str | UUID,
        crop_id: int | str | UUID,
        *,
        days: int = 30,
        heat_threshold_c: float | None = None,
    ) -> ClimateFeatures:
        """Build normalized climate inputs for deterministic scoring and AI use."""

        if self.db is None:
            raise ValueError("ClimateFeatureBuilder requires a database session for build_climate_features().")

        crop = get_crop(self.db, crop_id)  # type: ignore[arg-type]
        if crop is None:
            raise NotFoundError(f"Crop with id {crop_id} not found")

        summary = self.build_climate_summary(
            field_id,
            days=days,
            heat_threshold_c=heat_threshold_c,
        )
        requirements = resolve_crop_climate_requirements(crop)

        return ClimateFeatures(
            field_id=field_id,
            crop_id=crop_id,
            summary=summary,
            requirements=requirements,
            avg_temp_gap_to_ideal_c=_gap_to_range(
                summary.avg_temp if summary is not None else None,
                requirements.optimal_temp_min_c,
                requirements.optimal_temp_max_c,
            ),
            rainfall_gap_to_preferred_mm=_gap_to_range(
                summary.total_rainfall if summary is not None else None,
                requirements.preferred_rainfall_min_mm,
                requirements.preferred_rainfall_max_mm,
            ),
            frost_excess_days=_excess_days(
                summary.frost_days if summary is not None else None,
                requirements.frost_tolerance_days,
            ),
            heat_excess_days=_excess_days(
                summary.heat_days if summary is not None else None,
                requirements.heat_tolerance_days,
            ),
            temperature_within_ideal_range=_within_range(
                summary.avg_temp if summary is not None else None,
                requirements.optimal_temp_min_c,
                requirements.optimal_temp_max_c,
            ),
            rainfall_within_preferred_range=_within_range(
                summary.total_rainfall if summary is not None else None,
                requirements.preferred_rainfall_min_mm,
                requirements.preferred_rainfall_max_mm,
            ),
            observation_days_count=(
                summary.observation_days_count
                if summary is not None
                else 0
            ),
            missing_days_count=summary.missing_days_count if summary is not None else days,
            requirement_source=requirements.source,
        )

    def build_summary(
        self,
        observations: Sequence[ClimateObservation],
        *,
        lookback_days: int,
        heat_day_threshold: float,
    ) -> ClimateSummary | None:
        """Return a climate summary for the supplied in-memory weather observations."""

        if not observations:
            return None

        sorted_observations = sorted(observations, key=lambda item: item.observation_date)
        avg_temp_values = [
            observation.avg_temp
            if observation.avg_temp is not None
            else _average_from_min_max(observation.min_temp, observation.max_temp)
            for observation in sorted_observations
        ]
        avg_temp_values = [value for value in avg_temp_values if value is not None]
        min_temp_values = [observation.min_temp for observation in sorted_observations if observation.min_temp is not None]
        max_temp_values = [observation.max_temp for observation in sorted_observations if observation.max_temp is not None]
        rainfall_values = [
            float(observation.rainfall_mm or 0.0)
            for observation in sorted_observations
            if observation.rainfall_mm is not None
        ]
        humidity_values = [observation.humidity for observation in sorted_observations if observation.humidity is not None]
        wind_speed_values = [observation.wind_speed for observation in sorted_observations if observation.wind_speed is not None]
        solar_values = [
            observation.solar_radiation
            for observation in sorted_observations
            if observation.solar_radiation is not None
        ]
        et0_values = [float(observation.et0 or 0.0) for observation in sorted_observations if observation.et0 is not None]
        observation_days_count = len({observation.observation_date for observation in sorted_observations})
        coverage_ratio = (
            round(min(observation_days_count / lookback_days, 1.0), 4)
            if lookback_days > 0
            else None
        )

        return ClimateSummary(
            avg_temp=_round_optional(_average(avg_temp_values)),
            avg_min_temp=_round_optional(_average(min_temp_values)),
            avg_max_temp=_round_optional(_average(max_temp_values)),
            min_observed_temp=_round_optional(min(min_temp_values) if min_temp_values else None),
            max_observed_temp=_round_optional(max(max_temp_values) if max_temp_values else None),
            total_rainfall=_round_optional(sum(rainfall_values) if rainfall_values else 0.0),
            avg_humidity=_round_optional(_average(humidity_values)),
            avg_wind_speed=_round_optional(_average(wind_speed_values)),
            avg_solar_radiation=_round_optional(_average(solar_values)),
            total_et0=_round_optional(sum(et0_values) if et0_values else 0.0),
            frost_days=sum(
                1
                for observation in sorted_observations
                if observation.min_temp is not None and observation.min_temp < 0
            ),
            heat_days=sum(
                1
                for observation in sorted_observations
                if observation.max_temp is not None and observation.max_temp > heat_day_threshold
            ),
            weather_record_count=observation_days_count,
            observation_days_count=observation_days_count,
            missing_days_count=max(lookback_days - observation_days_count, 0),
            lookback_days=lookback_days,
            heat_threshold_c=heat_day_threshold,
            observation_start_date=sorted_observations[0].observation_date,
            observation_end_date=sorted_observations[-1].observation_date,
            coverage_ratio=coverage_ratio,
        )

    @staticmethod
    def observation_from_mapping(
        row: Mapping[str, Any],
        *,
        date_column_name: str = "date",
    ) -> ClimateObservation | None:
        """Normalize a weather-history row into an in-memory climate observation."""

        observation_date = row.get(date_column_name)
        if not isinstance(observation_date, date):
            return None
        return ClimateObservation(
            observation_date=observation_date,
            min_temp=_coerce_optional_float(row.get("min_temp")),
            max_temp=_coerce_optional_float(row.get("max_temp")),
            avg_temp=_coerce_optional_float(row.get("avg_temp")),
            rainfall_mm=_coerce_optional_float(row.get("rainfall_mm")),
            humidity=_coerce_optional_float(row.get("humidity")),
            wind_speed=_coerce_optional_float(row.get("wind_speed")),
            solar_radiation=_coerce_optional_float(row.get("solar_radiation")),
            et0=_coerce_optional_float(row.get("et0")),
        )


def _average(values: Sequence[float | None]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _average_from_min_max(min_temp: float | None, max_temp: float | None) -> float | None:
    if min_temp is None or max_temp is None:
        return None
    return (float(min_temp) + float(max_temp)) / 2.0


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 2)


def _gap_to_range(
    value: float | None,
    range_min: float | None,
    range_max: float | None,
) -> float | None:
    if value is None or (range_min is None and range_max is None):
        return None
    if range_min is not None and value < range_min:
        return round(range_min - value, 2)
    if range_max is not None and value > range_max:
        return round(value - range_max, 2)
    return 0.0


def _within_range(
    value: float | None,
    range_min: float | None,
    range_max: float | None,
) -> bool | None:
    if value is None or (range_min is None and range_max is None):
        return None
    if range_min is not None and value < range_min:
        return False
    if range_max is not None and value > range_max:
        return False
    return True


def _excess_days(observed_days: int | None, tolerance_days: int | None) -> int | None:
    if observed_days is None or tolerance_days is None:
        return None
    return max(observed_days - tolerance_days, 0)


def build_climate_summary(
    db: Session,
    field_id: int | str | UUID,
    *,
    days: int = 30,
    heat_threshold_c: float | None = None,
) -> ClimateSummary | None:
    """Build a persisted climate summary for one field."""

    return ClimateFeatureBuilder(db).build_climate_summary(
        field_id,
        days=days,
        heat_threshold_c=heat_threshold_c,
    )


def build_climate_features(
    db: Session,
    field_id: int | str | UUID,
    crop_id: int | str | UUID,
    *,
    days: int = 30,
    heat_threshold_c: float | None = None,
) -> ClimateFeatures:
    """Build normalized climate features for one field and crop."""

    return ClimateFeatureBuilder(db).build_climate_features(
        field_id,
        crop_id,
        days=days,
        heat_threshold_c=heat_threshold_c,
    )
