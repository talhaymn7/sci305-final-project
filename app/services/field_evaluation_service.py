"""Field-scoped climate evaluation services used by API handlers."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.orchestration.recommendation import RecommendationOrchestrator
from app.schemas.recommendation import RankedFieldResult, RecommendationBlockerRead
from app.schemas.weather_history import ClimateSummary
from app.services import field_catalog_service
from app.services.crop_service import get_crop
from app.services.errors import NotFoundError
from app.services.field_service import get_field
from app.services.soil_service import get_latest_soil_test_for_field
from app.services.weather_service import WeatherService


def _dedupe_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for message in messages:
        normalized = message.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _serialize_blockers(blockers: list[Any]) -> list[RecommendationBlockerRead]:
    return [
        RecommendationBlockerRead(
            code=blocker.code,
            dimension=blocker.dimension,
            message=blocker.message,
        )
        for blocker in blockers
    ]


def get_field_climate_summary(
    db: Session,
    field_id: int | str | UUID,
    *,
    days: int = 30,
    heat_threshold_c: float | None = None,
) -> ClimateSummary:
    """Return a recent field climate summary or raise a not-found error."""

    field_record = field_catalog_service.get_field(db, field_id)
    summary = WeatherService(db).get_climate_summary(
        field_record["id"],
        days=days,
        heat_threshold_c=heat_threshold_c,
    )
    if summary is None:
        raise NotFoundError("Climate summary not found for field")
    return summary


def get_field_recommendation(
    db: Session,
    *,
    field_id: int,
    crop_id: int,
    climate_days: int | None = None,
    heat_threshold_c: float | None = None,
) -> RankedFieldResult:
    """Return a single-field deterministic recommendation with climate context."""

    field_obj = get_field(db, field_id)
    if field_obj is None:
        raise NotFoundError("Field not found")

    crop = get_crop(db, crop_id)
    if crop is None:
        raise NotFoundError("Crop not found")

    soil_test = get_latest_soil_test_for_field(db, field_id)
    climate_summary = WeatherService(db).get_climate_summary(
        field_id,
        days=climate_days,
        heat_threshold_c=heat_threshold_c,
    )
    recommendation = RecommendationOrchestrator().generate(
        field_obj,
        crop,
        soil_test,
        climate_summary=climate_summary,
    )
    suitability = recommendation.suitability
    explanation = recommendation.explanation

    return RankedFieldResult(
        rank=1,
        field_id=field_id,
        crop_id=crop_id,
        score=suitability.total_score,
        total_score=suitability.total_score,
        agronomic_score=suitability.agronomic_score,
        climate_score=suitability.climate_score,
        explanation=explanation.detailed_explanation,
        reasons=_dedupe_messages(
            [
                *suitability.reasons,
                *suitability.climate_reasons,
                *suitability.climate_warnings,
            ]
        ),
        climate_reasons=list(suitability.climate_reasons),
        climate_warnings=list(suitability.climate_warnings),
        climate_strengths=list(suitability.climate_strengths),
        climate_weaknesses=list(suitability.climate_weaknesses),
        climate_risks=list(suitability.climate_risks),
        strengths=list(explanation.strengths),
        weaknesses=list(explanation.weaknesses),
        risks=list(explanation.risks),
        blockers=_serialize_blockers(suitability.blockers),
    )
