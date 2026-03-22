"""Duplicate-safe writer for normalized weather history ingestion records."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
import logging
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, Date, DateTime, Float, MetaData, String, Table, Uuid, insert, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.reflection import table_has_columns
from app.ingestion.deduplication import WeatherHistoryDuplicateStrategy, collect_existing_keys, deduplicate_prepared_rows
from app.ingestion.services.pipeline import NormalizedRecordWriter
from app.ingestion.types import NormalizedRecord, PersistResult, PreparedRow
from app.models.weather_history import WeatherHistory
from app.models.mixins import utc_now
from app.services.errors import ServiceValidationError


logger = logging.getLogger(__name__)


LEGACY_WEATHER_HISTORY_METADATA = MetaData()
LEGACY_WEATHER_HISTORY_TABLE = Table(
    "weather_history",
    LEGACY_WEATHER_HISTORY_METADATA,
    Column("id", Uuid(as_uuid=True), primary_key=True, nullable=False),
    Column("field_id", Uuid(as_uuid=True), nullable=True),
    Column("weather_date", Date, nullable=False),
    Column("min_temp", Float, nullable=True),
    Column("max_temp", Float, nullable=True),
    Column("avg_temp", Float, nullable=True),
    Column("rainfall_mm", Float, nullable=True),
    Column("humidity", Float, nullable=True),
    Column("wind_speed", Float, nullable=True),
    Column("solar_radiation", Float, nullable=True),
    Column("et0", Float, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


class WeatherHistoryIngestionWriter(NormalizedRecordWriter):
    """Persist normalized weather rows while skipping field/day duplicates."""

    def write(
        self,
        db: Session,
        records: Sequence[NormalizedRecord],
        *,
        data_source,
        ingestion_run,
    ) -> PersistResult:
        """Insert non-duplicate weather rows into the declared weather_history table."""

        _ = (data_source, ingestion_run)
        if not records:
            return PersistResult(
                records_inserted=0,
                records_skipped=0,
                skipped_is_successful=True,
                metadata_json={"target_table": "weather_history", "duplicate_count": 0},
            )

        weather_history_table, date_column_name = self._resolve_target_table(db)
        duplicate_strategy = WeatherHistoryDuplicateStrategy(date_column_name=date_column_name)

        prepared_rows = [
            PreparedRow(
                source_identifier=record.source_identifier,
                record_type=record.record_type,
                values=self._build_insert_row(weather_history_table, record, date_column_name=date_column_name),
            )
            for record in records
        ]

        field_ids = list({row.values["field_id"] for row in prepared_rows})
        weather_dates = list({row.values[date_column_name] for row in prepared_rows})

        existing_rows = db.execute(
            select(weather_history_table.c.field_id, getattr(weather_history_table.c, date_column_name))
            .where(weather_history_table.c.field_id.in_(field_ids))
            .where(getattr(weather_history_table.c, date_column_name).in_(weather_dates))
        ).mappings()
        deduplication_result = deduplicate_prepared_rows(
            prepared_rows,
            existing_keys=collect_existing_keys(existing_rows, duplicate_strategy),
            strategy=duplicate_strategy,
        )
        insert_rows = [dict(row.values) for row in deduplication_result.unique_rows]
        duplicate_count = len(deduplication_result.skipped_records)

        if duplicate_count:
            logger.info("Skipping %s duplicate weather_history rows", duplicate_count)

        if insert_rows:
            try:
                db.execute(insert(weather_history_table), insert_rows)
                db.commit()
            except SQLAlchemyError:
                db.rollback()
                raise

        return PersistResult(
            records_inserted=len(insert_rows),
            records_skipped=duplicate_count,
            skipped_is_successful=True,
            skipped_records=deduplication_result.skipped_records,
            metadata_json={
                "target_table": "weather_history",
                "date_column": date_column_name,
                "duplicate_count": duplicate_count,
                "input_record_count": len(prepared_rows),
            },
        )

    @staticmethod
    def _resolve_target_table(db: Session) -> tuple[Table, str]:
        if table_has_columns(db, "weather_history", "weather_date"):
            return LEGACY_WEATHER_HISTORY_TABLE, "weather_date"
        return WeatherHistory.__table__, "date"

    def _build_insert_row(
        self,
        weather_history_table: Table,
        record: NormalizedRecord,
        *,
        date_column_name: str,
    ) -> dict[str, Any]:
        weather_date = record.values.get("weather_date") or record.values.get("date")
        if not isinstance(weather_date, date):
            raise ServiceValidationError(f"Record '{record.source_identifier}' is missing a valid weather_date")

        row = {
            "field_id": self._normalize_identifier(weather_history_table.c.field_id, record.values.get("field_id")),
            date_column_name: weather_date,
            "min_temp": self._coerce_optional_float(record.values.get("min_temp")),
            "max_temp": self._coerce_optional_float(record.values.get("max_temp")),
            "avg_temp": self._coerce_optional_float(record.values.get("avg_temp")),
            "rainfall_mm": self._coerce_optional_float(record.values.get("rainfall_mm")),
            "humidity": self._coerce_optional_float(record.values.get("humidity")),
            "wind_speed": self._coerce_optional_float(record.values.get("wind_speed")),
            "solar_radiation": self._coerce_optional_float(record.values.get("solar_radiation")),
            "et0": self._coerce_optional_float(record.values.get("et0")),
        }
        if "id" in weather_history_table.c and date_column_name == "weather_date":
            row["id"] = uuid4()
        if "created_at" in weather_history_table.c:
            row["created_at"] = utc_now()
        return row

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _normalize_identifier(column, identifier: Any) -> Any:
        if identifier is None:
            raise ServiceValidationError("Normalized weather record is missing field_id")
        if not isinstance(identifier, str):
            return identifier
        try:
            python_type = column.type.python_type
        except (AttributeError, NotImplementedError):
            return identifier
        try:
            if python_type is int:
                return int(identifier)
            if python_type is UUID:
                return UUID(identifier)
        except (TypeError, ValueError) as exc:
            raise ServiceValidationError(
                f"Normalized weather record has an invalid field_id '{identifier}' for the target schema"
            ) from exc
        return identifier
