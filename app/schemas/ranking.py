"""Schemas for the field ranking API."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from app.engines.scoring_types import ScoreStatus
from app.schemas.ai_metadata import AITraceMetadataRead
from app.schemas.explanation import FieldExplanation
from app.schemas.yield_prediction import YieldPredictionRange


class RankFieldsRequest(BaseModel):
    """Request payload for ranking fields against a selected crop."""

    model_config = ConfigDict(extra="forbid")

    crop_id: PositiveInt | str
    top_n: PositiveInt | None = 5
    field_ids: list[PositiveInt | str] | None = None


class CropSummary(BaseModel):
    """Compact crop profile summary returned alongside ranking results."""

    model_config = ConfigDict(from_attributes=True)

    id: int | str | UUID
    crop_name: str
    scientific_name: str | None


class ScoreComponentRead(BaseModel):
    """Serialized scoring component included in ranking responses."""

    model_config = ConfigDict(from_attributes=True)

    key: str
    label: str
    weight: float
    awarded_points: float
    max_points: float
    status: ScoreStatus
    reasons: list[str]


class ScoreBlockerRead(BaseModel):
    """Serialized hard blocker included in ranking responses."""

    model_config = ConfigDict(from_attributes=True)

    code: str
    dimension: str
    message: str


class RankedFieldProviderMetadata(BaseModel):
    """Provider metadata attached to each ranked field result."""

    model_config = ConfigDict(extra="forbid")

    agronomic_provider: AITraceMetadataRead
    ranking_provider: AITraceMetadataRead
    explanation_provider: AITraceMetadataRead
    yield_provider: AITraceMetadataRead | None = None
    risk_provider: AITraceMetadataRead | None = None


class RankedFieldRecommendation(BaseModel):
    """Single ranked recommendation entry for a field."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    field_id: int | str | UUID
    field_name: str
    total_score: float
    agronomic_score: float
    climate_score: float | None
    economic_score: float
    risk_score: float | None
    confidence_score: float | None
    estimated_revenue: float | None = None
    estimated_cost: float | None = None
    estimated_profit: float | None = None
    predicted_yield: float | None
    predicted_yield_min: float | None = None
    predicted_yield_max: float | None = None
    predicted_yield_range: YieldPredictionRange | None
    ranking_score: float
    strengths: list[str]
    weaknesses: list[str]
    risks: list[str]
    climate_reasons: list[str] = Field(default_factory=list)
    climate_warnings: list[str] = Field(default_factory=list)
    climate_strengths: list[str] = Field(default_factory=list)
    climate_weaknesses: list[str] = Field(default_factory=list)
    climate_risks: list[str] = Field(default_factory=list)
    breakdown: dict[str, ScoreComponentRead]
    blockers: list[ScoreBlockerRead]
    reasons: list[str]
    metadata: AITraceMetadataRead
    provider_metadata: RankedFieldProviderMetadata
    explanation: FieldExplanation


class RankFieldsResponse(BaseModel):
    """Response payload returned by the field ranking endpoint."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "ranking.v2"
    crop: CropSummary
    total_fields_evaluated: int
    ranked_results: list[RankedFieldRecommendation]
