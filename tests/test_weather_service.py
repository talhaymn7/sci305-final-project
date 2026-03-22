from datetime import date

from app.models.enums import WaterSourceType
from app.models.field import Field
from app.models.weather_history import WeatherHistory
from app.services.weather_service import WeatherService


def _create_field(db, **overrides) -> Field:
    values = {
        "name": "Weather Field",
        "location_name": "Climate Valley",
        "latitude": 39.1,
        "longitude": -94.5,
        "area_hectares": 10.0,
        "elevation_meters": 220.0,
        "slope_percent": 3.0,
        "irrigation_available": True,
        "water_source_type": WaterSourceType.WELL,
        "infrastructure_score": 70,
        "drainage_quality": "good",
    }
    values.update(overrides)
    field = Field(**values)
    db.add(field)
    db.flush()
    return field


def _add_weather_record(db, field_id: int, **overrides) -> WeatherHistory:
    values = {
        "field_id": field_id,
        "date": date(2024, 1, 1),
        "min_temp": 5.0,
        "max_temp": 14.0,
        "avg_temp": 9.0,
        "rainfall_mm": 2.0,
        "humidity": 60.0,
        "wind_speed": 10.0,
        "solar_radiation": 16.0,
        "et0": 3.0,
    }
    values.update(overrides)
    record = WeatherHistory(**values)
    db.add(record)
    return record


def test_get_recent_weather_returns_newest_first_with_latest_date_anchor(db):
    field = _create_field(db)
    _add_weather_record(db, field.id, date=date(2024, 1, 1), min_temp=0.0, max_temp=4.0, avg_temp=2.0)
    _add_weather_record(db, field.id, date=date(2024, 1, 20), min_temp=4.0, max_temp=10.0, avg_temp=7.0)
    _add_weather_record(db, field.id, date=date(2024, 2, 15), min_temp=8.0, max_temp=14.0, avg_temp=11.0)
    db.commit()

    service = WeatherService(db)
    recent_weather = service.get_recent_weather(field.id, days=30)

    assert [record.date for record in recent_weather] == [date(2024, 2, 15), date(2024, 1, 20)]


def test_get_recent_weather_returns_empty_list_when_no_data_exists(db):
    field = _create_field(db)

    service = WeatherService(db)

    assert service.get_recent_weather(field.id, days=30) == []


def test_get_climate_summary_computes_expected_metrics(db):
    field = _create_field(db)
    _add_weather_record(
        db,
        field.id,
        date=date(2024, 1, 2),
        min_temp=-2.0,
        max_temp=5.0,
        avg_temp=1.0,
        rainfall_mm=10.0,
    )
    _add_weather_record(
        db,
        field.id,
        date=date(2024, 6, 15),
        min_temp=18.0,
        max_temp=37.0,
        avg_temp=27.0,
        rainfall_mm=2.0,
    )
    _add_weather_record(
        db,
        field.id,
        date=date(2024, 12, 31),
        min_temp=5.0,
        max_temp=34.0,
        avg_temp=20.0,
        rainfall_mm=0.5,
    )
    db.commit()

    service = WeatherService(db)
    summary = service.get_climate_summary(field.id, days=365)

    assert summary is not None
    assert summary.avg_temp == 16.0
    assert summary.total_rainfall == 12.5
    assert summary.frost_days == 1
    assert summary.heat_days == 2


def test_get_climate_summary_exposes_extended_window_metrics(db):
    field = _create_field(db)
    _add_weather_record(
        db,
        field.id,
        date=date(2024, 1, 8),
        min_temp=2.0,
        max_temp=14.0,
        avg_temp=8.0,
        rainfall_mm=3.0,
        humidity=55.0,
        wind_speed=7.0,
        solar_radiation=15.0,
        et0=2.5,
    )
    _add_weather_record(
        db,
        field.id,
        date=date(2024, 1, 10),
        min_temp=-1.0,
        max_temp=38.0,
        avg_temp=18.5,
        rainfall_mm=5.0,
        humidity=65.0,
        wind_speed=9.0,
        solar_radiation=17.0,
        et0=3.5,
    )
    db.commit()

    summary = WeatherService(db).get_climate_summary(field.id, days=7)

    assert summary is not None
    assert summary.avg_min_temp == 0.5
    assert summary.avg_max_temp == 26.0
    assert summary.total_et0 == 6.0
    assert summary.observation_days_count == 2
    assert summary.missing_days_count == 5
    assert summary.heat_threshold_c == 30.0
    assert summary.coverage_ratio == 0.2857


def test_get_climate_summaries_returns_bulk_field_lookup(db):
    field_a = _create_field(db, name="Weather Field A")
    field_b = _create_field(db, name="Weather Field B")
    _add_weather_record(db, field_a.id, date=date(2024, 2, 1), avg_temp=12.0, rainfall_mm=4.0)
    _add_weather_record(db, field_b.id, date=date(2024, 2, 1), max_temp=34.0, avg_temp=24.0, rainfall_mm=0.0)
    db.commit()

    summaries = WeatherService(db).get_climate_summaries([field_a.id, field_b.id, 9999], days=7)

    assert summaries[field_a.id] is not None
    assert summaries[field_b.id] is not None
    assert summaries[9999] is None
    assert summaries[field_a.id].avg_temp == 12.0
    assert summaries[field_b.id].avg_temp == 24.0


def test_get_climate_summary_honors_heat_threshold_override(db):
    field = _create_field(db)
    _add_weather_record(
        db,
        field.id,
        date=date(2024, 6, 15),
        min_temp=18.0,
        max_temp=37.0,
        avg_temp=27.0,
    )
    _add_weather_record(
        db,
        field.id,
        date=date(2024, 6, 16),
        min_temp=17.0,
        max_temp=34.0,
        avg_temp=25.0,
    )
    db.commit()

    service = WeatherService(db)
    summary = service.get_climate_summary(field.id, heat_threshold_c=33.0)

    assert summary is not None
    assert summary.heat_days == 2


def test_get_climate_summary_returns_none_when_no_data_exists(db):
    field = _create_field(db)

    service = WeatherService(db)

    assert service.get_climate_summary(field.id) is None
