# app/engines/hemp_prescription_engine.py
"""Hemp prescription engine.

Validates inputs, fires blocking constraints, routes to cold-start or
historical provider, and attaches agronomic notes to the result.
"""
from __future__ import annotations

from pathlib import Path

from app.ai.providers.ml.hemp_historical import HistoricalHempProvider
from app.ai.providers.ml.hemp_prescription import HempPrescriptionProvider
from app.schemas.hemp_cycle import CycleRecord
from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult

_cold_start_provider: HempPrescriptionProvider | None = None
_historical_provider: HistoricalHempProvider | None = None


def _get_cold_start(model_dir: str | Path | None = None) -> HempPrescriptionProvider:
    global _cold_start_provider
    if _cold_start_provider is None:
        _cold_start_provider = HempPrescriptionProvider(model_dir=model_dir)
    return _cold_start_provider


def _get_historical() -> HistoricalHempProvider:
    global _historical_provider
    if _historical_provider is None:
        _historical_provider = HistoricalHempProvider()
    return _historical_provider


# ── Hard blockers (deterministic -- run before any ML) ─────────────────────────

_BLOCKERS: list[tuple] = [
    (lambda r: r.ph is not None and r.ph < 5.0,
     "ph_too_acidic",    "Soil pH below 5.0 is outside hemp's tolerable range."),
    (lambda r: r.ph is not None and r.ph > 8.0,
     "ph_too_alkaline",  "Soil pH above 8.0 is outside hemp's tolerable range."),
    (lambda r: r.slope_percent > 20.0,
     "slope_excessive",  "Slope above 20% makes mechanical hemp cultivation impractical."),
    (lambda r: r.avg_temp is not None and r.avg_temp < 5.0,
     "temp_too_cold",    "Mean growing-season temperature below 5 degC prevents hemp establishment."),
    (lambda r: r.avg_temp is not None and r.avg_temp > 35.0,
     "temp_too_hot",     "Mean growing-season temperature above 35 degC causes heat stress."),
    (lambda r: r.ec is not None and r.ec > 4.0,
     "salinity_high",    "EC above 4.0 dS/m exceeds hemp's salinity tolerance."),
]


def _check_blockers(request: HempPrescriptionRequest) -> list[str]:
    return [message for check, _code, message in _BLOCKERS if check(request)]


# ── Agronomic notes ────────────────────────────────────────────────────────────

def _build_notes(request: HempPrescriptionRequest, result: HempPrescriptionResult) -> list[str]:
    notes: list[str] = []
    if request.ph is not None and not (6.0 <= request.ph <= 7.0):
        notes.append(f"pH {request.ph} is outside the ideal 6.0-7.0 range -- consider lime or sulfur amendment.")
    if request.drainage_class == "poor":
        notes.append("Poor drainage raises waterlogging risk -- raised beds or tile drainage recommended.")
    if result.rec_nitrogen_kg_dekar > 8.0:
        notes.append(f"High N deficit ({result.rec_nitrogen_kg_dekar:.1f} kg/dekar) -- split application across two dressings.")
    if request.irrigation_type == "none" and (request.seasonal_rainfall_mm or 999) < 300:
        notes.append("Seasonal rainfall below 300 mm with no irrigation -- drought stress likely during vegetative stage.")
    if request.slope_percent > 12:
        notes.append(f"Slope {request.slope_percent}% increases erosion risk -- contour cultivation advised.")
    if result.expected_yield_ton_dekar < 0.30:
        notes.append("Forecast yield below 0.30 t/dekar -- field conditions are suboptimal for fiber hemp.")
    if request.ph is None:
        notes.append("No soil analysis provided -- prescription based on variety defaults. Lab analysis recommended before planting.")
    return notes


# ── Public interface ───────────────────────────────────────────────────────────

def compute_hemp_prescription(
    request: HempPrescriptionRequest,
    *,
    model_dir: str | Path | None = None,
    cycle_history: list[CycleRecord] | None = None,
) -> HempPrescriptionResult:
    """Validate inputs, route to cold-start or historical provider, return annotated prescription."""
    blockers = _check_blockers(request)
    if blockers:
        return HempPrescriptionResult(
            rec_nitrogen_kg_dekar=0.0, rec_phosphorus_kg_dekar=0.0,
            rec_potassium_kg_dekar=0.0, rec_irrigation_mm_week=0.0,
            expected_yield_ton_dekar=0.0, suitable=False,
            confidence=1.0, blockers=blockers, notes=[], provider="none",
        )

    cold_start = _get_cold_start(model_dir).predict(request)

    if not cold_start.suitable:
        return HempPrescriptionResult(
            rec_nitrogen_kg_dekar=0.0, rec_phosphorus_kg_dekar=0.0,
            rec_potassium_kg_dekar=0.0, rec_irrigation_mm_week=0.0,
            expected_yield_ton_dekar=0.0, suitable=False,
            confidence=cold_start.confidence, blockers=[],
            notes=["Field conditions do not meet the minimum suitability threshold for hemp cultivation."],
            provider=cold_start.provider,
        )

    if cycle_history:
        result = _get_historical().predict(request, cycle_history, cold_start)
    else:
        result = cold_start

    agronomic_notes = _build_notes(request, result)
    return result.model_copy(update={"notes": agronomic_notes + result.notes})
