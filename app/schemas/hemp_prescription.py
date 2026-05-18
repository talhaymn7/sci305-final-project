"""Request and response schemas for the hemp prescription endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DrainageClass = Literal["poor", "moderate", "good", "excellent"]
TextureClass = Literal["sandy loam", "loam", "silt loam", "clay loam", "silty clay loam"]


class HempPrescriptionRequest(BaseModel):
    """All agronomic inputs required to generate a hemp prescription."""

    model_config = ConfigDict(extra="forbid")

    # Soil chemistry
    ph: float = Field(..., ge=3.0, le=10.0)
    nitrogen_ppm: float = Field(..., ge=0.0)
    phosphorus_ppm: float = Field(..., ge=0.0)
    potassium_ppm: float = Field(..., ge=0.0)
    organic_matter_percent: float = Field(..., ge=0.0, le=100.0)
    ec: float = Field(..., ge=0.0, description="Electrical conductivity (dS/m)")

    # Soil physical
    drainage_class: DrainageClass
    texture_class: TextureClass

    # Field
    area_hectares: float = Field(..., gt=0.0)
    slope_percent: float = Field(..., ge=0.0)
    irrigation_available: bool
    elevation_meters: float = Field(..., ge=0.0)

    # Climate (growing-season averages)
    avg_temp: float = Field(..., description="Mean growing-season temperature (°C)")
    seasonal_rainfall_mm: float = Field(..., ge=0.0)
    avg_humidity: float = Field(..., ge=0.0, le=100.0)
    avg_solar_radiation: float = Field(..., ge=0.0, description="MJ/m²/day")


class HempPrescriptionResult(BaseModel):
    """Prescription output returned for a hemp field scenario."""

    model_config = ConfigDict(extra="forbid")

    rec_nitrogen_kg_ha: float = Field(..., description="Recommended N application (kg/ha)")
    rec_phosphorus_kg_ha: float = Field(..., description="Recommended P application (kg/ha)")
    rec_potassium_kg_ha: float = Field(..., description="Recommended K application (kg/ha)")
    rec_irrigation_mm_week: float = Field(..., description="Weekly irrigation target (mm/week)")
    expected_yield_ton_ha: float = Field(..., description="Forecast fiber yield (ton/ha)")

    suitable: bool = Field(..., description="False if a blocking constraint eliminates this field")
    blockers: list[str] = Field(default_factory=list, description="Hard constraints that fired")
    notes: list[str] = Field(default_factory=list, description="Agronomic observations")
    provider: str = Field(..., description="Inference backend used")
