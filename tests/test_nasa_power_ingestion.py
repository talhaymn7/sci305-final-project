from __future__ import annotations

from datetime import date
from uuid import uuid4

import httpx
from sqlalchemy import Column, Date, DateTime, Float, MetaData, String, Table, Uuid, UniqueConstraint, create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.ingestion.clients.nasa_power import NASAPowerAPIClient, NASAPowerFetchResult
from app.ingestion.runners.run_nasa_power import execute_nasa_power_ingestion
from app.ingestion.services.weather_history_writer import WeatherHistoryIngestionWriter
from app.ingestion.transformers.nasa_power_weather import NASAPowerWeatherTransformer
from app.ingestion.types import NormalizedRecord, RawPayloadEnvelope
from app.models.enums import (
    DataSourceType,
    IngestionPayloadType,
    IngestionRunStatus,
    IngestionRunType,
)
from app.models.field import Field
from app.models.ingestion import DataSource, IngestionRun, RawIngestionPayload
from app.models.weather_history import WeatherHistory


class _StaticNASAAPIClient:
    def __init__(self, payloads: list[dict[str, object]] | dict[str, object], *, error_on_calls: set[int] | None = None) -> None:
        self.payloads = payloads if isinstance(payloads, list) else [payloads]
        self.error_on_calls = error_on_calls or set()
        self.calls: list[dict[str, object]] = []

    def fetch_daily_weather(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        base_url: str | None = None,
    ) -> NASAPowerFetchResult:
        call_index = len(self.calls)
        self.calls.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "base_url": base_url,
            }
        )
        if call_index in self.error_on_calls:
            raise RuntimeError(f"fetch failed for call {call_index}")

        payload = self.payloads[min(call_index, len(self.payloads) - 1)]
        parameter_names = tuple(payload["properties"]["parameter"].keys())
        return NASAPowerFetchResult(
            payload=payload,
            parameter_names=parameter_names,
            attempted_parameter_sets=(parameter_names,),
        )


def _build_nasa_payload() -> dict[str, object]:
    return {
        "properties": {
            "parameter": {
                "T2M_MIN": {"20250101": 10.0, "20250102": 11.0},
                "T2M_MAX": {"20250101": 20.0, "20250102": 21.0},
                "T2M": {"20250101": 15.0, "20250102": 16.0},
                "PRECTOTCORR": {"20250101": 3.2, "20250102": 0.0},
                "RH2M": {"20250101": 71.0, "20250102": 69.0},
                "WS2M": {"20250101": 4.6, "20250102": 5.1},
                "ALLSKY_SFC_SW_DWN": {"20250101": 12.3, "20250102": -999.0},
            }
        }
    }


def _build_empty_nasa_payload() -> dict[str, object]:
    return {
        "properties": {
            "parameter": {
                "T2M_MIN": {"20260320": -999.0, "20260321": -999.0},
                "T2M_MAX": {"20260320": -999.0, "20260321": -999.0},
                "T2M": {"20260320": -999.0, "20260321": -999.0},
                "PRECTOTCORR": {"20260320": -999.0, "20260321": -999.0},
                "RH2M": {"20260320": -999.0, "20260321": -999.0},
                "WS2M": {"20260320": -999.0, "20260321": -999.0},
                "ALLSKY_SFC_SW_DWN": {"20260320": -999.0, "20260321": -999.0},
            }
        }
    }


def _build_field(name: str = "North Block") -> Field:
    return Field(
        name=name,
        location_name="Springfield",
        latitude=39.7817,
        longitude=-89.6501,
        area_hectares=12.5,
        elevation_meters=182.0,
        slope_percent=2.4,
        irrigation_available=True,
        infrastructure_score=80,
        drainage_quality="good",
        notes="NASA ingestion test field",
    )


def _build_runtime_records() -> tuple[DataSource, IngestionRun]:
    data_source = DataSource(
        source_name="NASA POWER",
        source_type=DataSourceType.API,
        base_url="https://power.example.test",
        is_active=True,
    )
    ingestion_run = IngestionRun(
        data_source_id=uuid4(),
        run_type=IngestionRunType.INCREMENTAL,
    )
    return data_source, ingestion_run


def test_nasa_power_api_client_fetches_daily_weather_with_expected_query():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=_build_nasa_payload())

    client = NASAPowerAPIClient(transport=httpx.MockTransport(handler))

    result = client.fetch_daily_weather(
        latitude=39.7817,
        longitude=-89.6501,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
    )

    assert result.payload["properties"]["parameter"]["T2M"]["20250101"] == 15.0
    assert result.parameter_names == (
        "T2M",
        "T2M_MIN",
        "T2M_MAX",
        "PRECTOTCORR",
        "RH2M",
        "WS2M",
        "ALLSKY_SFC_SW_DWN",
    )
    assert captured["params"]["start"] == "20250101"
    assert captured["params"]["end"] == "20250102"
    assert captured["params"]["community"] == settings.NASA_POWER_COMMUNITY
    assert captured["params"]["time-standard"] == settings.NASA_POWER_TIME_STANDARD
    assert captured["params"]["parameters"] == "T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN"


def test_nasa_power_api_client_falls_back_when_primary_parameter_set_is_rejected():
    request_params: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_params.append(str(request.url.params["parameters"]))
        if "PRECTOTCORR" in str(request.url.params["parameters"]):
            return httpx.Response(422, json={"detail": "parameter PRECTOTCORR is not supported"})
        payload = _build_nasa_payload()
        payload["properties"]["parameter"]["PRECTOT"] = {"20250101": 3.2, "20250102": 0.0}
        return httpx.Response(200, json=payload)

    client = NASAPowerAPIClient(transport=httpx.MockTransport(handler))

    result = client.fetch_daily_weather(
        latitude=39.7817,
        longitude=-89.6501,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
    )

    assert len(request_params) == 2
    assert request_params[0] == "T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN"
    assert request_params[1] == "T2M,T2M_MIN,T2M_MAX,PRECTOT,RH2M,WS2M,ALLSKY_SFC_SW_DWN"
    assert result.parameter_names[3] == "PRECTOT"


def test_nasa_power_weather_transformer_maps_daily_rows():
    transformer = NASAPowerWeatherTransformer()
    data_source, ingestion_run = _build_runtime_records()

    records = transformer.transform(
        RawPayloadEnvelope(
            payload_type=IngestionPayloadType.JSON,
            source_identifier="field-1:2025-01-01:2025-01-02",
            raw_json={
                "field": {
                    "id": "field-1",
                    "name": "North Block",
                    "latitude": 39.7817,
                    "longitude": -89.6501,
                },
                "response": _build_nasa_payload(),
            },
        ),
        data_source=data_source,
        ingestion_run=ingestion_run,
    )

    assert len(records) == 2
    assert records[0].values["field_id"] == "field-1"
    assert records[0].values["weather_date"] == date(2025, 1, 1)
    assert records[0].values["rainfall_mm"] == 3.2
    assert records[1].values["solar_radiation"] is None
    assert "et0" not in records[1].values


def test_nasa_power_weather_transformer_omits_all_missing_days():
    transformer = NASAPowerWeatherTransformer()
    data_source, ingestion_run = _build_runtime_records()

    records = transformer.transform(
        RawPayloadEnvelope(
            payload_type=IngestionPayloadType.JSON,
            source_identifier="field-1:2026-03-20:2026-03-21",
            raw_json={
                "field": {
                    "id": "field-1",
                    "name": "North Block",
                    "latitude": 39.7817,
                    "longitude": -89.6501,
                },
                "response": _build_empty_nasa_payload(),
            },
        ),
        data_source=data_source,
        ingestion_run=ingestion_run,
    )

    assert records == []


def test_weather_history_writer_skips_existing_and_in_batch_duplicates(db):
    field = _build_field()
    db.add(field)
    db.commit()
    db.refresh(field)

    db.add(
        WeatherHistory(
            field_id=field.id,
            date=date(2025, 1, 1),
            min_temp=8.0,
            max_temp=18.0,
            avg_temp=13.0,
            rainfall_mm=1.1,
            humidity=65.0,
            wind_speed=3.2,
            solar_radiation=10.0,
            et0=2.0,
        )
    )
    db.commit()

    data_source, ingestion_run = _build_runtime_records()
    writer = WeatherHistoryIngestionWriter()
    result = writer.write(
        db,
        [
            NormalizedRecord(
                record_type="weather_history",
                source_identifier=f"{field.id}:2025-01-01",
                values={
                    "field_id": str(field.id),
                    "weather_date": date(2025, 1, 1),
                    "min_temp": 9.0,
                    "max_temp": 19.0,
                    "avg_temp": 14.0,
                    "rainfall_mm": 0.0,
                    "humidity": 60.0,
                    "wind_speed": 4.0,
                    "solar_radiation": 11.0,
                    "et0": 2.1,
                },
                payload_type=IngestionPayloadType.JSON,
            ),
            NormalizedRecord(
                record_type="weather_history",
                source_identifier=f"{field.id}:2025-01-02",
                values={
                    "field_id": str(field.id),
                    "weather_date": date(2025, 1, 2),
                    "min_temp": 10.0,
                    "max_temp": 20.0,
                    "avg_temp": 15.0,
                    "rainfall_mm": 2.0,
                    "humidity": 61.0,
                    "wind_speed": 4.1,
                    "solar_radiation": 12.0,
                    "et0": 2.4,
                },
                payload_type=IngestionPayloadType.JSON,
            ),
            NormalizedRecord(
                record_type="weather_history",
                source_identifier=f"{field.id}:2025-01-02:dup",
                values={
                    "field_id": str(field.id),
                    "weather_date": date(2025, 1, 2),
                    "min_temp": 10.0,
                    "max_temp": 20.0,
                    "avg_temp": 15.0,
                    "rainfall_mm": 2.0,
                    "humidity": 61.0,
                    "wind_speed": 4.1,
                    "solar_radiation": 12.0,
                    "et0": 2.4,
                },
                payload_type=IngestionPayloadType.JSON,
            ),
        ],
        data_source=data_source,
        ingestion_run=ingestion_run,
    )

    assert result.records_inserted == 1
    assert result.records_skipped == 2
    assert result.skipped_is_successful is True
    assert len(result.skipped_records) == 2
    assert result.skipped_records[0].stage == "deduplication"
    assert result.metadata_json["date_column"] == "date"
    assert db.query(WeatherHistory).count() == 2


def test_weather_history_writer_supports_weather_date_schema_without_reflection():
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    weather_history_table = Table(
        "weather_history",
        metadata,
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
        UniqueConstraint("field_id", "weather_date", name="weather_field_date_unique"),
    )
    metadata.create_all(engine)
    SessionTesting = sessionmaker(bind=engine)
    field_id = uuid4()
    data_source, ingestion_run = _build_runtime_records()

    with SessionTesting() as session:
        writer = WeatherHistoryIngestionWriter()
        first_result = writer.write(
            session,
            [
                NormalizedRecord(
                    record_type="weather_history",
                    source_identifier=f"{field_id}:2025-01-01",
                    values={
                        "field_id": str(field_id),
                        "weather_date": date(2025, 1, 1),
                        "min_temp": 9.0,
                        "max_temp": 19.0,
                        "avg_temp": 14.0,
                        "rainfall_mm": 0.5,
                        "humidity": 60.0,
                        "wind_speed": 4.0,
                        "solar_radiation": 11.0,
                    },
                    payload_type=IngestionPayloadType.JSON,
                )
            ],
            data_source=data_source,
            ingestion_run=ingestion_run,
        )
        second_result = writer.write(
            session,
            [
                NormalizedRecord(
                    record_type="weather_history",
                    source_identifier=f"{field_id}:2025-01-01:dup",
                    values={
                        "field_id": str(field_id),
                        "weather_date": date(2025, 1, 1),
                        "min_temp": 9.0,
                        "max_temp": 19.0,
                        "avg_temp": 14.0,
                        "rainfall_mm": 0.5,
                        "humidity": 60.0,
                        "wind_speed": 4.0,
                        "solar_radiation": 11.0,
                    },
                    payload_type=IngestionPayloadType.JSON,
                )
            ],
            data_source=data_source,
            ingestion_run=ingestion_run,
        )
        rows = session.execute(
            select(weather_history_table).order_by(weather_history_table.c.weather_date.asc())
        ).mappings().all()

    assert first_result.records_inserted == 1
    assert first_result.metadata_json["date_column"] == "weather_date"
    assert second_result.records_inserted == 0
    assert second_result.records_skipped == 1
    assert len(rows) == 1
    assert rows[0]["field_id"] == field_id
    assert rows[0]["weather_date"] == date(2025, 1, 1)


def test_execute_nasa_power_ingestion_writes_to_legacy_weather_history_schema():
    engine = create_engine("sqlite:///:memory:")
    schema_metadata = MetaData()
    fields_table = Table(
        "fields",
        schema_metadata,
        Column("id", Uuid(as_uuid=True), primary_key=True, nullable=False),
        Column("name", String(255), nullable=False),
        Column("latitude", Float, nullable=True),
        Column("longitude", Float, nullable=True),
    )
    weather_history_table = Table(
        "weather_history",
        schema_metadata,
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
        UniqueConstraint("field_id", "weather_date", name="weather_field_date_unique"),
    )
    schema_metadata.create_all(engine)
    DataSource.__table__.create(bind=engine, checkfirst=True)
    IngestionRun.__table__.create(bind=engine, checkfirst=True)
    RawIngestionPayload.__table__.create(bind=engine, checkfirst=True)
    SessionTesting = sessionmaker(bind=engine)
    field_id = uuid4()
    api_client = _StaticNASAAPIClient(_build_nasa_payload())

    with SessionTesting() as session:
        session.execute(
            fields_table.insert().values(
                id=field_id,
                name="Legacy North Block",
                latitude=39.7817,
                longitude=-89.6501,
            )
        )
        session.commit()

        first_result = execute_nasa_power_ingestion(
            session,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
            run_type=IngestionRunType.BACKFILL,
            api_client=api_client,
        )
        second_result = execute_nasa_power_ingestion(
            session,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
            run_type=IngestionRunType.BACKFILL,
            api_client=api_client,
        )

        weather_rows = session.execute(
            select(weather_history_table).order_by(weather_history_table.c.weather_date.asc())
        ).mappings().all()

        assert first_result.status == IngestionRunStatus.SUCCEEDED
        assert first_result.records_inserted == 2
        assert second_result.status == IngestionRunStatus.SUCCEEDED
        assert second_result.records_inserted == 0
        assert second_result.records_skipped == 2
        assert session.query(IngestionRun).count() == 2
        assert session.query(RawIngestionPayload).count() == 2
        assert len(weather_rows) == 2
        assert [row["weather_date"] for row in weather_rows] == [date(2025, 1, 1), date(2025, 1, 2)]


def test_execute_nasa_power_ingestion_persists_run_payloads_and_weather_rows(db):
    field = _build_field()
    db.add(field)
    db.commit()
    db.refresh(field)

    api_client = _StaticNASAAPIClient(_build_nasa_payload())

    first_result = execute_nasa_power_ingestion(
        db,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
        run_type=IngestionRunType.BACKFILL,
        api_client=api_client,
    )

    assert first_result.status == IngestionRunStatus.SUCCEEDED
    assert first_result.records_fetched == 1
    assert first_result.records_inserted == 2
    assert first_result.records_skipped == 0
    assert first_result.metadata_json["field_target_count"] == 1
    assert db.query(DataSource).count() == 1
    assert db.query(IngestionRun).count() == 1
    assert db.query(RawIngestionPayload).count() == 1
    assert db.query(WeatherHistory).count() == 2
    assert api_client.calls[0]["base_url"] == settings.NASA_POWER_BASE_URL

    second_result = execute_nasa_power_ingestion(
        db,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
        run_type=IngestionRunType.BACKFILL,
        api_client=api_client,
    )

    assert second_result.status == IngestionRunStatus.SUCCEEDED
    assert second_result.records_inserted == 0
    assert second_result.records_skipped == 2
    assert db.query(DataSource).count() == 1
    assert db.query(IngestionRun).count() == 2
    assert db.query(RawIngestionPayload).count() == 2
    assert db.query(WeatherHistory).count() == 2


def test_execute_nasa_power_ingestion_skips_failed_field_fetches_without_failing_run(db):
    primary_field = _build_field("North Block")
    secondary_field = _build_field("South Block")
    secondary_field.latitude = 40.5
    secondary_field.longitude = -90.1
    db.add_all([primary_field, secondary_field])
    db.commit()

    api_client = _StaticNASAAPIClient(_build_nasa_payload(), error_on_calls={1})

    result = execute_nasa_power_ingestion(
        db,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
        run_type=IngestionRunType.BACKFILL,
        api_client=api_client,
    )

    raw_payloads = db.query(RawIngestionPayload).order_by(RawIngestionPayload.id.asc()).all()

    assert result.status == IngestionRunStatus.PARTIAL
    assert result.records_fetched == 2
    assert result.records_inserted == 2
    assert result.records_skipped == 1
    assert len(raw_payloads) == 2
    assert "fetch_error" in raw_payloads[1].raw_json


def test_nasa_power_ingestion_client_shifts_default_window_when_recent_data_is_empty():
    api_client = _StaticNASAAPIClient([_build_empty_nasa_payload(), _build_nasa_payload()])
    from app.ingestion.clients.nasa_power import NASAPowerIngestionClient
    from app.ingestion.services.field_targets import FieldCoordinateTarget

    client = NASAPowerIngestionClient(
        api_client=api_client,
        field_targets=[
            FieldCoordinateTarget(
                field_id="field-1",
                field_name="North Block",
                latitude=39.7817,
                longitude=-89.6501,
            )
        ],
        end_date=date(2026, 3, 21),
        default_lookback_days=2,
        max_window_shifts=10,
    )

    resolved_start_date, resolved_end_date = client.resolve_date_range()

    assert resolved_end_date < date(2026, 3, 21)
    assert (resolved_end_date - resolved_start_date).days == 1
