from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Column, Date, DateTime, Float, MetaData, String, Table, create_engine, inspect, text

from app.ingestion.clients.base import IngestionClient
from app.ingestion.runners.job_runner import IngestionJobRunner
from app.ingestion.runners.registry import IngestionPipelineDefinition, IngestionPipelineRegistry
from app.ingestion.services.repository import IngestionRepository
from app.ingestion.services.pipeline import IngestionPipelineService
from app.ingestion.transformers.json_records import JSONRecordTransformer
from app.ingestion.types import NormalizedRecord, PersistResult, RawPayloadEnvelope
from app.ingestion.validators.required_fields import RequiredFieldsValidator
from app.models.enums import DataSourceType, IngestionPayloadType, IngestionRunStatus, IngestionRunType
from app.models.ingestion import DataSource, IngestionRun, RawIngestionPayload


REPO_ROOT = Path(__file__).resolve().parents[1]


class _StaticClient(IngestionClient):
    def __init__(self, payloads: list[RawPayloadEnvelope]) -> None:
        self.payloads = payloads

    def fetch(self, data_source: DataSource, *, run_type: IngestionRunType) -> list[RawPayloadEnvelope]:
        _ = (data_source, run_type)
        return list(self.payloads)


class _CollectingWriter:
    def __init__(self) -> None:
        self.records: list[NormalizedRecord] = []

    def write(
        self,
        db,
        records,
        *,
        data_source: DataSource,
        ingestion_run: IngestionRun,
    ) -> PersistResult:
        _ = (db, data_source, ingestion_run)
        self.records.extend(records)
        return PersistResult(
            records_inserted=len(records),
            records_skipped=0,
            metadata_json={"writer_name": "collecting_writer"},
        )


class _FailingWriter:
    def write(
        self,
        db,
        records,
        *,
        data_source: DataSource,
        ingestion_run: IngestionRun,
    ) -> PersistResult:
        _ = (db, records, data_source, ingestion_run)
        raise RuntimeError("sink unavailable")


def _make_alembic_config(database_url: str) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_ingestion_pipeline_service_persists_raw_payloads_and_finalizes_partial_run(db):
    data_source = DataSource(
        source_name="NOAA Weather Feed",
        source_type=DataSourceType.API,
        base_url="https://example.test/weather",
        is_active=True,
    )
    db.add(data_source)
    db.commit()
    db.refresh(data_source)

    client = _StaticClient(
        [
            RawPayloadEnvelope(
                payload_type=IngestionPayloadType.BATCH,
                source_identifier="batch-1",
                raw_json=[
                    {"external_id": "wx-1", "station_name": "North Station", "rainfall_mm": 21.4},
                    {"external_id": "wx-2", "rainfall_mm": 8.1},
                ],
            )
        ]
    )
    writer = _CollectingWriter()
    pipeline = IngestionPipelineService(
        db,
        client=client,
        transformer=JSONRecordTransformer(
            record_type="weather_observation",
            source_identifier_field="external_id",
        ),
        writer=writer,
        validators=[RequiredFieldsValidator("external_id", "station_name")],
    )

    result = pipeline.run(data_source.id, run_type=IngestionRunType.INCREMENTAL)

    assert result.status == IngestionRunStatus.PARTIAL
    assert result.records_fetched == 1
    assert result.records_inserted == 1
    assert result.records_skipped == 1
    assert result.error_message is None
    assert result.metadata_json["validation_error_count"] == 1
    assert result.metadata_json["skipped_record_count"] == 1
    assert result.metadata_json["skip_reason_counts"] == {"missing_required_field": 1}
    assert result.metadata_json["skip_record_samples"][0]["stage"] == "validation"
    assert result.metadata_json["writer_name"] == "collecting_writer"

    run = db.get(IngestionRun, result.ingestion_run_id)
    assert run is not None
    assert run.status == IngestionRunStatus.PARTIAL
    assert run.run_type == IngestionRunType.INCREMENTAL
    assert run.finished_at is not None

    raw_payloads = (
        db.query(RawIngestionPayload)
        .filter(RawIngestionPayload.ingestion_run_id == run.id)
        .order_by(RawIngestionPayload.id.asc())
        .all()
    )
    assert len(raw_payloads) == 1
    assert raw_payloads[0].source_identifier == "batch-1"
    assert writer.records[0].source_identifier == "wx-1"
    assert writer.records[0].values["station_name"] == "North Station"


def test_ingestion_repository_auto_creates_missing_ingestion_tables(db):
    bind = db.get_bind()
    assert bind is not None

    RawIngestionPayload.__table__.drop(bind=bind, checkfirst=True)
    IngestionRun.__table__.drop(bind=bind, checkfirst=True)
    DataSource.__table__.drop(bind=bind, checkfirst=True)

    repository = IngestionRepository(db)

    repository.ensure_ingestion_tables_ready()

    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    assert "data_sources" in tables
    assert "ingestion_runs" in tables
    assert "raw_ingestion_payloads" in tables


def test_ingestion_pipeline_service_marks_run_failed_when_writer_raises(db):
    data_source = DataSource(
        source_name="Regional Soil Feed",
        source_type=DataSourceType.API,
        base_url="https://example.test/soils",
        is_active=True,
    )
    db.add(data_source)
    db.commit()
    db.refresh(data_source)

    pipeline = IngestionPipelineService(
        db,
        client=_StaticClient(
            [
                RawPayloadEnvelope(
                    payload_type=IngestionPayloadType.RECORD,
                    source_identifier="soil-1",
                    raw_json={"external_id": "soil-1", "ph": 6.7, "field_name": "Parcel A"},
                )
            ]
        ),
        transformer=JSONRecordTransformer(record_type="soil_observation"),
        writer=_FailingWriter(),
        validators=[RequiredFieldsValidator("external_id", "field_name")],
    )

    with pytest.raises(RuntimeError, match="sink unavailable"):
        pipeline.run(data_source.id)

    run = db.query(IngestionRun).order_by(IngestionRun.id.desc()).first()
    assert run is not None
    assert run.status == IngestionRunStatus.FAILED
    assert run.records_fetched == 1
    assert run.records_inserted == 0
    assert run.records_skipped == 0
    assert "sink unavailable" in (run.error_message or "")

    raw_payload_count = (
        db.query(RawIngestionPayload)
        .filter(RawIngestionPayload.ingestion_run_id == run.id)
        .count()
    )
    assert raw_payload_count == 1


def test_ingestion_job_runner_runs_registered_pipeline_for_active_sources(db):
    active_source = DataSource(
        source_name="Weather API",
        source_type=DataSourceType.API,
        base_url="https://example.test/weather",
        is_active=True,
    )
    inactive_source = DataSource(
        source_name="Old Weather API",
        source_type=DataSourceType.API,
        base_url="https://example.test/old-weather",
        is_active=False,
    )
    db.add_all([active_source, inactive_source])
    db.commit()

    registry = IngestionPipelineRegistry()
    writer = _CollectingWriter()
    registry.register(
        DataSourceType.API,
        IngestionPipelineDefinition(
            client=_StaticClient(
                [
                    RawPayloadEnvelope(
                        payload_type=IngestionPayloadType.RECORD,
                        source_identifier="weather-1",
                        raw_json={"external_id": "weather-1", "station_name": "North Station"},
                    )
                ]
            ),
            transformer=JSONRecordTransformer(record_type="weather_observation"),
            writer=writer,
            validators=(RequiredFieldsValidator("external_id", "station_name"),),
        ),
    )

    runner = IngestionJobRunner(db, registry=registry)
    results = runner.run_active_sources(run_type=IngestionRunType.BACKFILL)

    assert len(results) == 1
    assert results[0].status == IngestionRunStatus.SUCCEEDED
    assert results[0].records_inserted == 1
    assert writer.records[0].values["station_name"] == "North Station"


def test_ingestion_foundation_migration_upgrade_and_downgrade(tmp_path):
    database_path = tmp_path / "ingestion_foundation.sqlite"
    database_url = f"sqlite:///{database_path.as_posix()}"
    config = _make_alembic_config(database_url)

    command.upgrade(config, "010")
    command.upgrade(config, "011")

    engine = create_engine(database_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert "data_sources" in tables
    assert "ingestion_runs" in tables
    assert "raw_ingestion_payloads" in tables

    metadata = MetaData()
    metadata.reflect(bind=engine, only=("data_sources", "ingestion_runs", "raw_ingestion_payloads"))
    data_sources_table = metadata.tables["data_sources"]
    ingestion_runs_table = metadata.tables["ingestion_runs"]
    raw_payloads_table = metadata.tables["raw_ingestion_payloads"]
    data_source_id = uuid4().hex

    with engine.begin() as connection:
        connection.execute(
            data_sources_table.insert().values(
                id=data_source_id,
                source_name="Weather API",
                source_type="api",
                base_url="https://example.test/weather",
                is_active=True,
                created_at=text("CURRENT_TIMESTAMP"),
                updated_at=text("CURRENT_TIMESTAMP"),
            )
        )
        connection.execute(
            ingestion_runs_table.insert().values(
                id=1,
                data_source_id=data_source_id,
                run_type="full",
                status="succeeded",
                started_at=text("CURRENT_TIMESTAMP"),
                finished_at=text("CURRENT_TIMESTAMP"),
                records_fetched=12,
                records_inserted=10,
                records_skipped=2,
                error_message=None,
                metadata_json={},
            )
        )
        connection.execute(
            raw_payloads_table.insert().values(
                id=1,
                ingestion_run_id=1,
                payload_type="json",
                source_identifier="weather-1",
                raw_json={"rainfall_mm": 12.4},
                created_at=text("CURRENT_TIMESTAMP"),
            )
        )

    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT records_fetched, records_inserted, records_skipped
                FROM ingestion_runs
                WHERE id = 1
                """
            )
        ).mappings().one()

    assert row["records_fetched"] == 12
    assert row["records_inserted"] == 10
    assert row["records_skipped"] == 2

    command.downgrade(config, "010")

    downgraded_tables = set(inspect(engine).get_table_names())
    assert "data_sources" not in downgraded_tables
    assert "ingestion_runs" not in downgraded_tables
    assert "raw_ingestion_payloads" not in downgraded_tables


def test_alembic_upgrade_head_bootstraps_unversioned_legacy_schema(tmp_path):
    database_path = tmp_path / "legacy_bootstrap.sqlite"
    database_url = f"sqlite:///{database_path.as_posix()}"
    engine = create_engine(database_url)
    metadata = MetaData()
    Table(
        "fields",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("name", String(255), nullable=False),
        Column("latitude", Float, nullable=True),
        Column("longitude", Float, nullable=True),
    )
    Table(
        "soil_tests",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("field_id", String(36), nullable=False),
        Column("sample_date", Date, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
    )
    Table(
        "crop_profiles",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("crop_name", String(255), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )
    Table(
        "weather_history",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("field_id", String(36), nullable=False),
        Column("weather_date", Date, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
    )
    metadata.create_all(engine)

    config = _make_alembic_config(database_url)
    command.upgrade(config, "head")

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert "alembic_version" in tables
    assert "data_sources" in tables
    assert "ingestion_runs" in tables
    assert "raw_ingestion_payloads" in tables

    expected_head = ScriptDirectory.from_config(config).get_current_head()
    with engine.connect() as connection:
        version_num = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()

    assert version_num == expected_head
