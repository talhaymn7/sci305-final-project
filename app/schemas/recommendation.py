"""Schemas for single-field recommendation and evaluation APIs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RecommendationRead(BaseModel):
    """Persisted recommendation record."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    field_id: int
    crop_id: int
    suitability_score: float
    rank: int
    explanation: str
    created_at: datetime


class RankFieldsRequest(BaseModel):
    """Legacy ranking request shape retained for compatibility."""

    model_config = ConfigDict(extra="forbid")

    field_ids: list[int]
    crop_id: int
    top_n: int = 5


class RecommendationBlockerRead(BaseModel):
    """Serialized blocker returned by a direct field recommendation."""

    model_config = ConfigDict(extra="forbid")

    code: str
    dimension: str
    message: str


class RankedFieldResult(BaseModel):
    """Single-field evaluation payload with climate-aware scoring fields."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    field_id: int | str | UUID
    crop_id: int | str | UUID
    score: float
    total_score: float
    agronomic_score: float
    climate_score: float | None = None
    explanation: str
    reasons: list[str] = Field(default_factory=list)
    climate_reasons: list[str] = Field(default_factory=list)
    climate_warnings: list[str] = Field(default_factory=list)
    climate_strengths: list[str] = Field(default_factory=list)
    climate_weaknesses: list[str] = Field(default_factory=list)
    climate_risks: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    blockers: list[RecommendationBlockerRead] = Field(default_factory=list)
