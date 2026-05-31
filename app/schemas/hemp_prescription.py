"""Request and response schemas for the hemp prescription endpoint."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

DrainageClass  = Literal["poor", "moderate", "good", "excellent"]
TextureClass   = Literal["sandy loam", "loam", "silt loam", "clay loam", "silty clay loam"]
HempVariety    = Literal["narli", "vezir", "other"]
IrrigationType = Literal["drip", "sprinkler", "flood", "none"]
WaterSource    = Literal["well", "river", "municipal", "rain"]
PreviousCrop   = Literal["legume", "cereal", "sunflower", "hemp", "fallow", "other"]


class HempPrescriptionRequest(BaseModel):
    """All agronomic inputs required to generate a hemp prescription.

    Soil chemistry and climate fields are optional; the engine imputes
    variety-specific defaults and lowers the confidence score when they
    are absent.
    """
    model_config = ConfigDict(extra="forbid")

    # Variety and agronomic context
    variety: HempVariety = "other"
    first_hemp_season: bool = False
    previous_crop: PreviousCrop = "other"

    # Soil chemistry (optional -- allow no-lab cold start)
    ph:                      float | None = Field(None, ge=3.0,   le=10.0)
    nitrogen_ppm:            float | None = Field(None, ge=0.0)
    phosphorus_ppm:          float | None = Field(None, ge=0.0)
    potassium_ppm:           float | None = Field(None, ge=0.0)
    organic_matter_percent:  float | None = Field(None, ge=0.0, le=100.0)
    ec:                      float | None = Field(None, ge=0.0, description="Electrical conductivity (dS/m)")

    # Soil physical
    drainage_class: DrainageClass = "moderate"
    texture_class:  TextureClass  = "loam"

    # Field (area in dekar; 1 ha = 10 dekar)
    area_dekar:        float = Field(..., gt=0.0, description="Field area (dekar)")
    slope_percent:     float = Field(..., ge=0.0)
    irrigation_type:   IrrigationType = "none"
    water_source:      WaterSource    = "rain"
    elevation_meters:  float = Field(default=100.0, ge=0.0)

    # Climate (optional -- fetched from integrations when absent)
    avg_temp:              float | None = Field(None, description="Mean growing-season temperature (degC)")
    seasonal_rainfall_mm:  float | None = Field(None, ge=0.0)
    avg_humidity:          float | None = Field(None, ge=0.0, le=100.0)
    avg_solar_radiation:   float | None = Field(None, ge=0.0, description="MJ/m2/day")


class HempPrescriptionResult(BaseModel):
    """Prescription output returned for a hemp field scenario."""
    model_config = ConfigDict(extra="forbid")

    # Recommendations (dekar units)
    rec_nitrogen_kg_dekar:    float = Field(..., description="Recommended N application (kg/dekar)")
    rec_phosphorus_kg_dekar:  float = Field(..., description="Recommended P application (kg/dekar)")
    rec_potassium_kg_dekar:   float = Field(..., description="Recommended K application (kg/dekar)")
    rec_irrigation_mm_week:   float = Field(..., description="Weekly gross irrigation target (mm/week)")
    expected_yield_ton_dekar: float = Field(..., description="Forecast fiber yield (ton/dekar)")

    suitable:   bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0,
                              description="Prediction confidence: 1.0=full lab data+history, 0.4=no data")
    blockers:   list[str] = Field(default_factory=list)
    notes:      list[str] = Field(default_factory=list)
    provider:   str       = Field(..., description="Inference backend: xgboost | rule_based | historical | none")
