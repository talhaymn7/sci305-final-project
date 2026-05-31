"""Schemas for hemp cycle history and the adaptive prescription request."""
from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.hemp_prescription import HempPrescriptionRequest


class CycleRecord(BaseModel):
    """One completed crop cycle -- the atomic unit for historical learning.

    Stores what was recommended, what was actually applied, and what
    yield was achieved. The engine uses a list of these to adjust the
    cold-start recommendation for the next cycle.
    """
    model_config = ConfigDict(extra="forbid")

    cycle_id: str = Field(..., description="Unique cycle ID (e.g. FieldCropCycle PK as string)")

    # What the cold-start model recommended at cycle start
    predicted_yield_ton_dekar: float = Field(..., ge=0.0)
    rec_nitrogen_kg_dekar:     float = Field(..., ge=0.0)
    rec_phosphorus_kg_dekar:   float = Field(..., ge=0.0)
    rec_potassium_kg_dekar:    float = Field(..., ge=0.0)
    rec_irrigation_mm_week:    float = Field(..., ge=0.0)

    # What was actually applied
    applied_nitrogen_kg_dekar:    float = Field(..., ge=0.0)
    applied_phosphorus_kg_dekar:  float = Field(..., ge=0.0)
    applied_potassium_kg_dekar:   float = Field(..., ge=0.0)
    applied_irrigation_mm_season: float = Field(..., ge=0.0,
                                                description="Total irrigation applied over the full season (mm)")

    # Outcomes recorded at harvest
    actual_yield_ton_dekar:  float = Field(..., ge=0.0)
    actual_moisture_percent: float = Field(..., ge=0.0, le=100.0)
    cycle_days:              int   = Field(..., gt=0)

    # Agronomic context (used to weight relevance across varieties/systems)
    variety:         str = "other"
    previous_crop:   str = "other"
    irrigation_type: str = "none"


class AdaptivePrescriptionRequest(BaseModel):
    """Wrapper that bundles the current field request with cycle history."""
    model_config = ConfigDict(extra="forbid")

    request:       HempPrescriptionRequest
    cycle_history: list[CycleRecord] = Field(default_factory=list)
