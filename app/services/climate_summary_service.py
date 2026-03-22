"""Typed SQL climate aggregation over recent weather history windows."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, case, distinct, func, literal, or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.reflection import reflect_tables, table_has_columns
from app.models.weather_history import WeatherHistory
from app.schemas.weather_history import ClimateSummary


def _round_optional(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 2)


class ClimateSummaryService:
    """Aggregate recent field climate signals from weather history."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_field_summary(
        self,
        field_id: int | str | UUID,
        *,
        days: int | None = None,
        heat_threshold_c: float | None = None,
    ) -> ClimateSummary | None:
        """Return one field-scoped climate summary anchored to its latest weather date."""

        summaries = self.get_field_summaries(
            [field_id],
            days=days,
            heat_threshold_c=heat_threshold_c,
        )
        return summaries.get(field_id)

    def get_field_summaries(
        self,
        field_ids: Iterable[int | str | UUID],
        *,
        days: int | None = None,
        heat_threshold_c: float | None = None,
    ) -> dict[int | str | UUID, ClimateSummary | None]:
        """Return climate summaries for multiple fields using shared SQL aggregation."""

        requested_field_ids = list(dict.fromkeys(field_ids))
        if not requested_field_ids:
            return {}

        resolved_days = days or settings.CLIMATE_LOOKBACK_DAYS
        resolved_heat_threshold = heat_threshold_c or settings.HEAT_DAY_THRESHOLD
        weather_table, field_column, date_column = self._get_weather_table_context()

        normalized_to_original = {
            self._normalize_identifier(field_column, field_id): field_id
            for field_id in requested_field_ids
        }
        latest_dates = self._load_latest_dates(
            weather_table,
            field_column,
            date_column,
            list(normalized_to_original.keys()),
        )
        if not latest_dates:
            return {field_id: None for field_id in requested_field_ids}

        window_lookup = {
            normalized_field_id: (
                latest_date - timedelta(days=resolved_days - 1),
                latest_date,
            )
            for normalized_field_id, latest_date in latest_dates.items()
        }
        rows = self._load_aggregate_rows(
            weather_table,
            field_column,
            date_column,
            window_lookup,
            heat_threshold_c=resolved_heat_threshold,
        )

        summaries = {field_id: None for field_id in requested_field_ids}
        for row in rows:
            normalized_field_id = row["field_id"]
            original_field_id = normalized_to_original.get(normalized_field_id, normalized_field_id)
            summaries[original_field_id] = self._summary_from_row(
                field_id=original_field_id,
                row=row,
                lookback_days=resolved_days,
                heat_threshold_c=resolved_heat_threshold,
            )
        return summaries

    def _get_weather_table_context(self):
        if table_has_columns(self.db, "weather_history", "date"):
            weather_table = WeatherHistory.__table__
            date_column_name = "date"
        else:
            weather_table = reflect_tables(self.db, "weather_history")["weather_history"]
            date_column_name = "date" if "date" in weather_table.c else "weather_date"
        return weather_table, weather_table.c.field_id, weather_table.c[date_column_name]

    def _load_latest_dates(
        self,
        weather_table,
        field_column,
        date_column,
        field_ids: list[int | str | UUID],
    ) -> dict[int | str | UUID, date]:
        latest_date_rows = self.db.execute(
            select(
                field_column.label("field_id"),
                func.max(date_column).label("latest_date"),
            )
            .where(field_column.in_(field_ids))
            .group_by(field_column)
        ).mappings().all()
        return {
            row["field_id"]: row["latest_date"]
            for row in latest_date_rows
            if isinstance(row["latest_date"], date)
        }

    def _load_aggregate_rows(
        self,
        weather_table,
        field_column,
        date_column,
        window_lookup: dict[int | str | UUID, tuple[date, date]],
        *,
        heat_threshold_c: float,
    ) -> list[dict[str, Any]]:
        if not window_lookup:
            return []

        clauses = [
            and_(
                field_column == field_id,
                date_column >= start_date,
                date_column <= end_date,
            )
            for field_id, (start_date, end_date) in window_lookup.items()
        ]
        where_clause = clauses[0] if len(clauses) == 1 else or_(*clauses)

        min_temp_column = weather_table.c.get("min_temp")
        max_temp_column = weather_table.c.get("max_temp")
        avg_temp_column = weather_table.c.get("avg_temp")
        rainfall_column = weather_table.c.get("rainfall_mm")
        humidity_column = weather_table.c.get("humidity")
        wind_speed_column = weather_table.c.get("wind_speed")
        solar_column = weather_table.c.get("solar_radiation")
        et0_column = weather_table.c.get("et0")

        if avg_temp_column is not None:
            if min_temp_column is not None and max_temp_column is not None:
                avg_temp_source = func.coalesce(
                    avg_temp_column,
                    (min_temp_column + max_temp_column) / 2.0,
                )
            else:
                avg_temp_source = avg_temp_column
        elif min_temp_column is not None and max_temp_column is not None:
            avg_temp_source = (min_temp_column + max_temp_column) / 2.0
        else:
            avg_temp_source = literal(None)

        aggregate_query = select(
            field_column.label("field_id"),
            func.avg(avg_temp_source).label("avg_temp"),
            self._avg_or_null(min_temp_column).label("avg_min_temp"),
            self._avg_or_null(max_temp_column).label("avg_max_temp"),
            self._min_or_null(min_temp_column).label("min_observed_temp"),
            self._max_or_null(max_temp_column).label("max_observed_temp"),
            self._sum_or_null(rainfall_column).label("total_rainfall"),
            self._avg_or_null(humidity_column).label("avg_humidity"),
            self._avg_or_null(wind_speed_column).label("avg_wind_speed"),
            self._avg_or_null(solar_column).label("avg_solar_radiation"),
            self._sum_or_null(et0_column).label("total_et0"),
            self._sum_case(min_temp_column, threshold=0.0, operator="lt").label("frost_days"),
            self._sum_case(max_temp_column, threshold=heat_threshold_c, operator="gt").label("heat_days"),
            func.count(distinct(date_column)).label("observation_days_count"),
            func.min(date_column).label("observation_start_date"),
            func.max(date_column).label("observation_end_date"),
        ).where(where_clause).group_by(field_column)

        return [dict(row) for row in self.db.execute(aggregate_query).mappings().all()]

    @staticmethod
    def _avg_or_null(column):
        if column is None:
            return literal(None)
        return func.avg(column)

    @staticmethod
    def _min_or_null(column):
        if column is None:
            return literal(None)
        return func.min(column)

    @staticmethod
    def _max_or_null(column):
        if column is None:
            return literal(None)
        return func.max(column)

    @staticmethod
    def _sum_or_null(column):
        if column is None:
            return literal(None)
        return func.sum(func.coalesce(column, 0.0))

    @staticmethod
    def _sum_case(column, *, threshold: float, operator: str):
        if column is None:
            return literal(0)
        if operator == "lt":
            condition = column < threshold
        else:
            condition = column > threshold
        return func.sum(case((condition, 1), else_=0))

    @staticmethod
    def _normalize_identifier(column, identifier: int | str | UUID) -> int | str | UUID:
        if not isinstance(identifier, str):
            return identifier

        try:
            python_type = column.type.python_type
        except (AttributeError, NotImplementedError):
            return identifier

        if python_type is int:
            try:
                return int(identifier)
            except ValueError:
                return identifier
        if python_type is UUID:
            try:
                return UUID(identifier)
            except ValueError:
                return identifier
        return identifier

    def _summary_from_row(
        self,
        *,
        field_id: int | str | UUID,
        row: dict[str, Any],
        lookback_days: int,
        heat_threshold_c: float,
    ) -> ClimateSummary:
        observation_days_count = int(row.get("observation_days_count") or 0)
        return ClimateSummary(
            field_id=field_id,
            avg_avg_temp=_round_optional(row.get("avg_temp")),
            avg_temp=_round_optional(row.get("avg_temp")),
            min_temp_avg=_round_optional(row.get("avg_min_temp")),
            avg_min_temp=_round_optional(row.get("avg_min_temp")),
            max_temp_avg=_round_optional(row.get("avg_max_temp")),
            avg_max_temp=_round_optional(row.get("avg_max_temp")),
            min_observed_temp=_round_optional(row.get("min_observed_temp")),
            max_observed_temp=_round_optional(row.get("max_observed_temp")),
            total_rainfall_mm=_round_optional(row.get("total_rainfall")),
            total_rainfall=_round_optional(row.get("total_rainfall")),
            avg_humidity=_round_optional(row.get("avg_humidity")),
            avg_wind=_round_optional(row.get("avg_wind_speed")),
            avg_wind_speed=_round_optional(row.get("avg_wind_speed")),
            avg_solar=_round_optional(row.get("avg_solar_radiation")),
            avg_solar_radiation=_round_optional(row.get("avg_solar_radiation")),
            total_et0=_round_optional(row.get("total_et0")),
            frost_days_count=int(row.get("frost_days") or 0),
            frost_days=int(row.get("frost_days") or 0),
            heat_days_count=int(row.get("heat_days") or 0),
            heat_days=int(row.get("heat_days") or 0),
            weather_record_count=observation_days_count,
            observation_days=observation_days_count,
            observation_days_count=observation_days_count,
            missing_days_count=max(lookback_days - observation_days_count, 0),
            lookback_days=lookback_days,
            heat_threshold_c=round(float(heat_threshold_c), 2),
            observation_start_date=row.get("observation_start_date"),
            observation_end_date=row.get("observation_end_date"),
            coverage_ratio=(
                round(min(observation_days_count / lookback_days, 1.0), 4)
                if lookback_days > 0
                else None
            ),
        )


def get_climate_summary(
    db: Session,
    field_id: int | str | UUID,
    days: int = 30,
    *,
    heat_threshold_c: float | None = None,
) -> ClimateSummary | None:
    """Return one field-scoped climate summary using SQL aggregation."""

    return ClimateSummaryService(db).get_field_summary(
        field_id,
        days=days,
        heat_threshold_c=heat_threshold_c,
    )


def get_climate_summaries(
    db: Session,
    field_ids: Iterable[int | str | UUID],
    days: int = 30,
    *,
    heat_threshold_c: float | None = None,
) -> dict[int | str | UUID, ClimateSummary | None]:
    """Return multiple field-scoped climate summaries using shared SQL aggregation."""

    return ClimateSummaryService(db).get_field_summaries(
        field_ids,
        days=days,
        heat_threshold_c=heat_threshold_c,
    )
