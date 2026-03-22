from logging.config import fileConfig
import logging

from sqlalchemy import Column, MetaData, String, Table, engine_from_config, inspect, pool
from alembic import context
from alembic.script import ScriptDirectory
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.config import settings
from app.database import Base
from app.models import crop_economic_profile, crop_price, crop_profile, external_crop_statistics, feedback, field, field_crop_cycle, ingestion, input_cost, recommendation, soil_test, weather_history  # noqa

config = context.config
legacy_database_url = "postgresql://postgres:postgres@localhost:5432/agrimind"
logger = logging.getLogger("alembic.env")
LEGACY_CORE_TABLES = frozenset(
    {
        "fields",
        "soil_tests",
        "crop_profiles",
        "weather_history",
    }
)
INGESTION_FOUNDATION_TABLES = (
    ingestion.DataSource.__table__,
    ingestion.IngestionRun.__table__,
    ingestion.RawIngestionPayload.__table__,
)


def _escape_config_value(value: str) -> str:
    """Escape percent signs before storing values in the Alembic config parser."""

    return value.replace("%", "%%")


configured_database_url = config.get_main_option("sqlalchemy.url")
if configured_database_url in {"", legacy_database_url}:
    config.set_main_option("sqlalchemy.url", _escape_config_value(settings.DATABASE_URL))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _bootstrap_unversioned_legacy_schema(connection) -> None:
    inspector = inspect(connection)
    available_tables = set(inspector.get_table_names())
    if "alembic_version" in available_tables:
        return
    if not LEGACY_CORE_TABLES.issubset(available_tables):
        return

    for table in INGESTION_FOUNDATION_TABLES:
        table.create(bind=connection, checkfirst=True)

    script = ScriptDirectory.from_config(config)
    head_revision = script.get_current_head()
    if head_revision is None:
        return

    version_table = Table(
        "alembic_version",
        MetaData(),
        Column("version_num", String(32), primary_key=True, nullable=False),
    )
    version_table.create(bind=connection, checkfirst=True)
    existing_version = connection.execute(version_table.select()).scalar_one_or_none()
    if existing_version is not None:
        return

    logger.warning(
        "Detected an unversioned existing AgriMind schema; bootstrapping ingestion tables and stamping Alembic head %s",
        head_revision,
    )
    connection.execute(version_table.insert().values(version_num=head_revision))


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        _bootstrap_unversioned_legacy_schema(connection)
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
        if connection.in_transaction():
            connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
