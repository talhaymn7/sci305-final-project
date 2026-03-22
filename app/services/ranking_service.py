"""Application service for ranking fields for a selected crop."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.contracts.metadata import AITraceMetadata
from app.ai.contracts.explanation import ExplanationOutput, adapt_explanation_output
from app.ai.features.explanation import build_explanation_input
from app.ai.orchestration.ranking import RankedFieldResultInternal, RankingOrchestrator
from app.ai.registry import AIProviderRegistry, get_ai_provider_registry
from app.db.reflection import reflect_tables, table_has_columns, tables_exist
from app.schemas.ai_metadata import AITraceMetadataRead
from app.schemas.ranking import (
    CropSummary,
    RankFieldsResponse,
    RankedFieldProviderMetadata,
    RankedFieldRecommendation,
    ScoreBlockerRead,
    ScoreComponentRead,
)
from app.services.crop_service import get_crop
from app.services.economic_service import EconomicService
from app.services.errors import NotFoundError, ServiceValidationError
from app.services.field_service import get_all_fields, get_fields_by_ids
from app.services.soil_service import get_latest_soil_test_for_field
from app.services.weather_service import WeatherService
from app.services.yield_prediction_service import YieldPredictionService


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(round(float(value)))


def _normalize_reflected_id(column, identifier: int | str | UUID) -> int | str | UUID:
    """Coerce a string identifier to the reflected column's Python type when possible."""

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


def _supports_provider_backed_ranking(db: Session) -> bool:
    """Return whether the full ORM-backed ranking stack is available on this schema."""

    return (
        tables_exist(db, "fields", "crop_profiles", "soil_tests", "weather_history")
        and table_has_columns(db, "fields", "drainage_quality")
        and table_has_columns(
            db,
            "crop_profiles",
            "optimal_temp_min_c",
            "optimal_temp_max_c",
            "rainfall_requirement_mm",
        )
        and table_has_columns(db, "weather_history", "date")
    )


def _validate_fallback_crop_row(crop_row: Mapping[str, Any], crop_id: int | str) -> None:
    required_columns = (
        "crop_name",
        "ideal_ph_min",
        "ideal_ph_max",
        "tolerable_ph_min",
        "tolerable_ph_max",
        "water_requirement_level",
        "drainage_requirement",
        "frost_sensitivity",
        "heat_sensitivity",
    )
    missing_columns = [column for column in required_columns if crop_row.get(column) is None]
    if missing_columns:
        missing_display = ", ".join(missing_columns)
        raise ServiceValidationError(
            f"Crop with id {crop_id} is missing required ranking fields: {missing_display}"
        )


def _build_fallback_crop_proxy(crop_row: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id=crop_row["id"],
        crop_name=crop_row["crop_name"],
        scientific_name=crop_row.get("scientific_name"),
        ideal_ph_min=float(crop_row["ideal_ph_min"]),
        ideal_ph_max=float(crop_row["ideal_ph_max"]),
        tolerable_ph_min=float(crop_row["tolerable_ph_min"]),
        tolerable_ph_max=float(crop_row["tolerable_ph_max"]),
        water_requirement_level=crop_row["water_requirement_level"],
        drainage_requirement=crop_row["drainage_requirement"],
        frost_sensitivity=crop_row["frost_sensitivity"],
        heat_sensitivity=crop_row["heat_sensitivity"],
        salinity_tolerance=crop_row.get("salinity_tolerance"),
        rooting_depth_cm=_coerce_float(crop_row.get("rooting_depth_cm")),
        slope_tolerance=_coerce_float(crop_row.get("slope_tolerance")),
        optimal_temp_min_c=_coerce_float(crop_row.get("optimal_temp_min_c")),
        optimal_temp_max_c=_coerce_float(crop_row.get("optimal_temp_max_c")),
        tolerable_temp_min_c=_coerce_float(crop_row.get("tolerable_temp_min_c")),
        tolerable_temp_max_c=_coerce_float(crop_row.get("tolerable_temp_max_c")),
        rainfall_requirement_mm=_coerce_float(crop_row.get("rainfall_requirement_mm")),
        preferred_rainfall_min_mm=_coerce_float(crop_row.get("preferred_rainfall_min_mm")),
        preferred_rainfall_max_mm=_coerce_float(crop_row.get("preferred_rainfall_max_mm")),
        frost_tolerance_days=_coerce_int(crop_row.get("frost_tolerance_days")),
        heat_tolerance_days=_coerce_int(crop_row.get("heat_tolerance_days")),
        organic_matter_preference=crop_row.get("organic_matter_preference"),
    )


def _build_fallback_field_proxy(
    field_row: Mapping[str, Any],
    soil_row: Mapping[str, Any] | None,
) -> SimpleNamespace:
    drainage_quality = (
        field_row.get("drainage_quality")
        or (soil_row.get("drainage_class") if soil_row is not None else None)
        or "moderate"
    )
    return SimpleNamespace(
        id=field_row["id"],
        name=field_row["name"],
        location_name=field_row.get("location_name"),
        latitude=_coerce_float(field_row.get("latitude")),
        longitude=_coerce_float(field_row.get("longitude")),
        area_hectares=_coerce_float(field_row.get("area_hectares")) or 0.0,
        elevation_meters=_coerce_float(field_row.get("elevation_meters")),
        slope_percent=_coerce_float(field_row.get("slope_percent")) or 0.0,
        aspect=field_row.get("aspect"),
        irrigation_available=bool(field_row.get("irrigation_available")),
        water_source_type=field_row.get("water_source_type"),
        infrastructure_score=_coerce_int(field_row.get("infrastructure_score")),
        drainage_quality=drainage_quality,
        notes=field_row.get("notes"),
    )


def _build_fallback_soil_proxy(soil_row: Mapping[str, Any] | None) -> SimpleNamespace | None:
    if soil_row is None:
        return None
    if soil_row.get("ph") is None or soil_row.get("organic_matter_percent") is None:
        return None
    return SimpleNamespace(
        id=soil_row["id"],
        field_id=soil_row["field_id"],
        sample_date=soil_row.get("sample_date"),
        ph=float(soil_row["ph"]),
        ec=_coerce_float(soil_row.get("ec")),
        organic_matter_percent=float(soil_row["organic_matter_percent"]),
        nitrogen_ppm=_coerce_float(soil_row.get("nitrogen_ppm")),
        phosphorus_ppm=_coerce_float(soil_row.get("phosphorus_ppm")),
        potassium_ppm=_coerce_float(soil_row.get("potassium_ppm")),
        calcium_ppm=_coerce_float(soil_row.get("calcium_ppm")),
        magnesium_ppm=_coerce_float(soil_row.get("magnesium_ppm")),
        texture_class=soil_row.get("texture_class"),
        drainage_class=soil_row.get("drainage_class"),
        depth_cm=_coerce_float(soil_row.get("depth_cm")),
        water_holding_capacity=_coerce_float(soil_row.get("water_holding_capacity")),
        notes=soil_row.get("notes"),
    )


def _load_fallback_field_rows(
    db: Session,
    *,
    field_ids: list[int | str | UUID] | None,
) -> list[Mapping[str, Any]]:
    fields_table = reflect_tables(db, "fields")["fields"]
    if field_ids is None:
        query = select(fields_table)
        if "created_at" in fields_table.c:
            query = query.order_by(fields_table.c.created_at.asc(), fields_table.c.name.asc())
        else:
            query = query.order_by(fields_table.c.name.asc())
        return db.execute(query).mappings().all()

    requested_ids = [
        _normalize_reflected_id(fields_table.c.id, field_id)
        for field_id in dict.fromkeys(field_ids)
    ]
    if not requested_ids:
        return []
    rows = db.execute(
        select(fields_table).where(fields_table.c.id.in_(requested_ids))
    ).mappings().all()
    rows_by_id = {row["id"]: row for row in rows}
    return [rows_by_id[field_id] for field_id in requested_ids if field_id in rows_by_id]


def _load_latest_fallback_soils(
    db: Session,
    *,
    field_ids: list[int | str | UUID],
) -> dict[int | str | UUID, Mapping[str, Any]]:
    if not field_ids:
        return {}

    soil_tests_table = reflect_tables(db, "soil_tests")["soil_tests"]
    query = select(soil_tests_table).where(soil_tests_table.c.field_id.in_(field_ids))
    if "sample_date" in soil_tests_table.c and "created_at" in soil_tests_table.c:
        query = query.order_by(
            soil_tests_table.c.field_id.asc(),
            soil_tests_table.c.sample_date.desc(),
            soil_tests_table.c.created_at.desc(),
        )
    rows = db.execute(query).mappings().all()
    latest_by_field_id: dict[int | str | UUID, Mapping[str, Any]] = {}
    for row in rows:
        field_id = row["field_id"]
        if field_id not in latest_by_field_id:
            latest_by_field_id[field_id] = row
    return latest_by_field_id


def _get_ranked_fields_response_fallback(
    db: Session,
    *,
    crop_id: int | str | UUID,
    top_n: int | None,
    field_ids: list[int | str | UUID] | None,
) -> RankFieldsResponse:
    crop_profiles_table = reflect_tables(db, "crop_profiles")["crop_profiles"]
    normalized_crop_id = _normalize_reflected_id(crop_profiles_table.c.id, crop_id)
    crop_row = db.execute(
        select(crop_profiles_table).where(crop_profiles_table.c.id == normalized_crop_id)
    ).mappings().one_or_none()
    if crop_row is None:
        raise NotFoundError(f"Crop with id {crop_id} not found")

    _validate_fallback_crop_row(crop_row, crop_id)
    field_rows = _load_fallback_field_rows(db, field_ids=field_ids)
    if not field_rows:
        if field_ids is None:
            raise NotFoundError("No fields found for ranking")
        raise NotFoundError("No fields found for the provided field filter")

    latest_soil_rows = _load_latest_fallback_soils(
        db,
        field_ids=[row["id"] for row in field_rows],
    )
    crop = _build_fallback_crop_proxy(crop_row)
    field_proxies = [
        _build_fallback_field_proxy(field_row, latest_soil_rows.get(field_row["id"]))
        for field_row in field_rows
    ]
    soil_lookup = {
        field_row["id"]: _build_fallback_soil_proxy(latest_soil_rows.get(field_row["id"]))
        for field_row in field_rows
    }
    climate_lookup = None
    if tables_exist(db, "weather_history"):
        weather_service = WeatherService(db)
        climate_lookup = weather_service.get_climate_summaries([field_row["id"] for field_row in field_rows])
    yield_service = YieldPredictionService(db)
    yield_lookup = {
        field_row["id"]: yield_service.predict_for_entities(
            field_proxies[index],
            crop,
            soil_test=soil_lookup[field_row["id"]],
            climate_summary=(climate_lookup or {}).get(field_row["id"]),
        )
        for index, field_row in enumerate(field_rows)
    }
    economic_service = EconomicService(db)
    economic_lookup = {
        field_row["id"]: economic_service.calculate_profit(
            field_proxies[index],
            crop,
            soil_test=soil_lookup[field_row["id"]],
            climate_summary=(climate_lookup or {}).get(field_row["id"]),
            yield_prediction=yield_lookup[field_row["id"]],
        )
        for index, field_row in enumerate(field_rows)
    }

    registry = get_ai_provider_registry()
    explanation_provider = registry.get_explanation_provider()
    ranking = RankingOrchestrator(
        suitability_provider=registry.get_suitability_provider(),
        ranking_augmentation_provider=registry.get_ranking_augmentation_provider(),
    ).rank_fields_for_crop(
        field_proxies,
        crop,
        soil_lookup,
        top_n=top_n,
        climate_summaries=climate_lookup,
        economic_assessments=economic_lookup,
        yield_predictions=yield_lookup,
    )

    return RankFieldsResponse(
        crop=CropSummary.model_validate(crop),
        total_fields_evaluated=len(field_proxies),
        ranked_results=[
            _serialize_ranked_result(
                entry,
                explanation_provider=explanation_provider,
                registry=registry,
            )
            for entry in ranking.ranked_fields
        ],
    )


def _serialize_breakdown(breakdown: dict[str, object]) -> dict[str, ScoreComponentRead]:
    return {
        key: ScoreComponentRead.model_validate(component)
        for key, component in breakdown.items()
    }


def _serialize_blockers(blockers: list[object]) -> list[ScoreBlockerRead]:
    return [ScoreBlockerRead.model_validate(blocker) for blocker in blockers]


def _serialize_trace_metadata(metadata: AITraceMetadata) -> AITraceMetadataRead:
    normalized = metadata.normalized()
    return AITraceMetadataRead(
        provider_name=normalized.provider_name,
        provider_version=normalized.provider_version,
        generated_at=normalized.generated_at,
        confidence=normalized.confidence,
        debug_info=normalized.debug_info,
    )


def _serialize_provider_metadata(
    entry: RankedFieldResultInternal,
    *,
    explanation_output: ExplanationOutput,
    registry: AIProviderRegistry,
) -> RankedFieldProviderMetadata:
    yield_prediction = entry.yield_prediction
    generated_at = explanation_output.generated_at
    risk_metadata_payload = (explanation_output.debug_info or {}).get("risk_provider_metadata")
    risk_provider_metadata = None
    if isinstance(risk_metadata_payload, dict):
        risk_generated_at = risk_metadata_payload.get("generated_at")
        if isinstance(risk_generated_at, str):
            try:
                parsed_risk_generated_at = datetime.fromisoformat(risk_generated_at)
            except ValueError:
                parsed_risk_generated_at = generated_at
        else:
            parsed_risk_generated_at = generated_at
        risk_provider_metadata = _serialize_trace_metadata(
            AITraceMetadata(
                provider_name=str(risk_metadata_payload.get("provider_name", registry.settings.AI_RISK_PROVIDER)),
                provider_version=(
                    str(risk_metadata_payload["provider_version"])
                    if risk_metadata_payload.get("provider_version") is not None
                    else None
                ),
                generated_at=parsed_risk_generated_at,
                confidence=(
                    float(risk_metadata_payload["confidence"])
                    if risk_metadata_payload.get("confidence") is not None
                    else None
                ),
                debug_info=(
                    risk_metadata_payload.get("debug_info")
                    if isinstance(risk_metadata_payload.get("debug_info"), dict)
                    else None
                ),
            )
        )
    return RankedFieldProviderMetadata(
        agronomic_provider=_serialize_trace_metadata(
            AITraceMetadata(
                provider_name=registry.settings.AI_SUITABILITY_PROVIDER,
                provider_version=None,
                generated_at=generated_at,
                confidence=entry.result.confidence_score,
                debug_info={
                    "agronomic_score": entry.agronomic_score,
                    "climate_score": entry.climate_score,
                    "economic_score": entry.economic_score,
                    "total_score": entry.total_score,
                },
            )
        ),
        ranking_provider=_serialize_trace_metadata(
            AITraceMetadata(
                provider_name=registry.settings.AI_RANKING_AUGMENTATION_PROVIDER,
                provider_version=None,
                generated_at=generated_at,
                confidence=entry.confidence_score,
                debug_info={
                    "ranking_score": entry.ranking_score,
                    "economic_score": entry.economic_score,
                    "estimated_revenue": entry.estimated_revenue,
                    "estimated_cost": entry.estimated_cost,
                    "estimated_profit": entry.estimated_profit,
                },
            )
        ),
        explanation_provider=_serialize_trace_metadata(explanation_output.metadata),
        yield_provider=(
            _serialize_trace_metadata(
                AITraceMetadata(
                    provider_name=yield_prediction.metadata.provider_name,
                    provider_version=yield_prediction.model_version,
                    generated_at=yield_prediction.metadata.generated_at,
                    confidence=yield_prediction.confidence_score,
                    debug_info={
                        **(yield_prediction.metadata.debug_info or {}),
                        "training_source": yield_prediction.training_source,
                    },
                )
            )
            if yield_prediction is not None
            else None
        ),
        risk_provider=risk_provider_metadata
        or _serialize_trace_metadata(
            AITraceMetadata(
                provider_name=registry.settings.AI_RISK_PROVIDER,
                provider_version=None,
                generated_at=generated_at,
                confidence=explanation_output.confidence,
                debug_info={"risk_count": len(explanation_output.risks)},
            )
        ),
    )


def _serialize_ranking_metadata(
    entry: RankedFieldResultInternal,
    *,
    explanation_output: ExplanationOutput,
    registry: AIProviderRegistry,
) -> AITraceMetadataRead:
    return _serialize_trace_metadata(
        AITraceMetadata(
            provider_name=registry.settings.AI_RANKING_AUGMENTATION_PROVIDER,
            provider_version=None,
            generated_at=datetime.now(timezone.utc),
            confidence=entry.confidence_score or explanation_output.confidence,
            debug_info={
                "agronomic_provider": registry.settings.AI_SUITABILITY_PROVIDER,
                "explanation_provider": explanation_output.provider_name,
                "has_yield_prediction": entry.yield_prediction is not None,
                "climate_score": entry.climate_score,
                "economic_score": entry.economic_score,
            },
        )
    )


def _serialize_ranked_result(
    entry: RankedFieldResultInternal,
    *,
    explanation_provider,
    registry: AIProviderRegistry,
) -> RankedFieldRecommendation:
    explanation_output = explanation_provider.explain(build_explanation_input(entry))
    return RankedFieldRecommendation(
        rank=entry.rank,
        field_id=entry.field_id,
        field_name=entry.field_name,
        total_score=entry.total_score,
        agronomic_score=entry.agronomic_score,
        climate_score=entry.climate_score,
        economic_score=entry.economic_score,
        risk_score=entry.risk_score,
        confidence_score=entry.confidence_score,
        estimated_revenue=entry.estimated_revenue,
        estimated_cost=entry.estimated_cost,
        estimated_profit=entry.estimated_profit,
        predicted_yield=entry.predicted_yield,
        predicted_yield_min=entry.predicted_yield_min,
        predicted_yield_max=entry.predicted_yield_max,
        predicted_yield_range=entry.predicted_yield_range,
        ranking_score=entry.ranking_score,
        strengths=explanation_output.strengths,
        weaknesses=explanation_output.weaknesses,
        risks=explanation_output.risks,
        climate_reasons=entry.climate_reasons,
        climate_warnings=entry.climate_warnings,
        climate_strengths=entry.climate_strengths,
        climate_weaknesses=entry.climate_weaknesses,
        climate_risks=entry.climate_risks,
        breakdown=_serialize_breakdown(entry.breakdown),
        blockers=_serialize_blockers(entry.blockers),
        reasons=entry.reasons,
        metadata=_serialize_ranking_metadata(
            entry,
            explanation_output=explanation_output,
            registry=registry,
        ),
        provider_metadata=_serialize_provider_metadata(
            entry,
            explanation_output=explanation_output,
            registry=registry,
        ),
        explanation=adapt_explanation_output(explanation_output),
    )


def _load_fields_for_ranking(db: Session, field_ids: list[int] | None) -> list[object]:
    if field_ids is None:
        return get_all_fields(db)
    return get_fields_by_ids(db, field_ids)


def get_ranked_fields_response(
    db: Session,
    crop_id: int | str | UUID,
    top_n: int | None = 5,
    field_ids: list[int | str | UUID] | None = None,
) -> RankFieldsResponse:
    """Build the ranking response payload for the POST /rank-fields endpoint."""

    if not _supports_provider_backed_ranking(db):
        return _get_ranked_fields_response_fallback(
            db,
            crop_id=crop_id,
            top_n=top_n,
            field_ids=field_ids,
        )

    crop = get_crop(db, crop_id)
    if crop is None:
        raise NotFoundError(f"Crop with id {crop_id} not found")

    fields = _load_fields_for_ranking(db, field_ids)
    if not fields:
        if field_ids is None:
            raise NotFoundError("No fields found for ranking")
        raise NotFoundError("No fields found for the provided field filter")

    soil_lookup = {
        field.id: get_latest_soil_test_for_field(db, field.id)
        for field in fields
    }
    weather_service = WeatherService(db)
    climate_lookup = weather_service.get_climate_summaries([field.id for field in fields])
    yield_service = YieldPredictionService(db)
    yield_lookup = {
        field.id: yield_service.predict_for_entities(
            field,
            crop,
            soil_test=soil_lookup[field.id],
            climate_summary=climate_lookup[field.id],
        )
        for field in fields
    }
    economic_service = EconomicService(db)
    economic_lookup = {
        field.id: economic_service.calculate_profit(
            field,
            crop,
            soil_test=soil_lookup[field.id],
            climate_summary=climate_lookup[field.id],
            yield_prediction=yield_lookup[field.id],
        )
        for field in fields
    }
    registry = get_ai_provider_registry()
    explanation_provider = registry.get_explanation_provider()
    ranking = RankingOrchestrator(
        suitability_provider=registry.get_suitability_provider(),
        ranking_augmentation_provider=registry.get_ranking_augmentation_provider(),
    ).rank_fields_for_crop(
        fields,
        crop,
        soil_lookup,
        top_n=top_n,
        climate_summaries=climate_lookup,
        economic_assessments=economic_lookup,
        yield_predictions=yield_lookup,
    )

    return RankFieldsResponse(
        crop=CropSummary.model_validate(crop),
        total_fields_evaluated=len(fields),
        ranked_results=[
            _serialize_ranked_result(
                entry,
                explanation_provider=explanation_provider,
                registry=registry,
            )
            for entry in ranking.ranked_fields
        ],
    )
