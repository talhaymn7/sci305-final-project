"""Compatibility façade for provider-based field ranking."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.ai.orchestration.ranking import (
    ClimateLookup,
    EconomicLookup,
    RankedFieldResultInternal,
    RankingOrchestrator,
    RankingResult,
    SoilLookup,
    rank_fields_for_crop,
)
from app.models.field import Field
from app.services.crop_service import get_crop
from app.services.economic_service import EconomicService
from app.services.field_service import get_field
from app.services.soil_service import get_latest_soil_test_for_field
from app.services.weather_service import WeatherService


def rank_fields(
    db: Session,
    field_ids: list[int],
    crop_id: int,
    top_n: int = 5,
) -> RankingResult:
    """DB-backed adapter that preserves the existing ranking API contract."""

    crop_profile = get_crop(db, crop_id)
    if not crop_profile:
        raise ValueError(f"Crop with id {crop_id} not found")

    fields: list[Field] = []
    for field_id in field_ids:
        field_obj = get_field(db, field_id)
        if field_obj is not None:
            fields.append(field_obj)

    soil_lookup = {
        field_obj.id: get_latest_soil_test_for_field(db, field_obj.id)
        for field_obj in fields
    }
    weather_service = WeatherService(db)
    climate_lookup = weather_service.get_climate_summaries([field_obj.id for field_obj in fields])
    economic_service = EconomicService(db)
    economic_lookup = {
        field_obj.id: economic_service.calculate_profit(
            field_obj,
            crop_profile,
            soil_test=soil_lookup[field_obj.id],
            climate_summary=climate_lookup[field_obj.id],
        )
        for field_obj in fields
    }
    return RankingOrchestrator().rank_fields_for_crop(
        fields,
        crop_profile,
        soil_lookup,
        top_n=top_n,
        climate_summaries=climate_lookup,
        economic_assessments=economic_lookup,
    )


__all__ = [
    "ClimateLookup",
    "EconomicLookup",
    "RankedFieldResultInternal",
    "RankingOrchestrator",
    "RankingResult",
    "SoilLookup",
    "rank_fields",
    "rank_fields_for_crop",
]
