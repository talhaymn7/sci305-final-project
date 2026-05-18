"""Hemp prescription engine.

Validates inputs, fires blocking constraints, calls the ML provider,
and attaches human-readable agronomic notes to the result.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.ai.providers.ml.hemp_prescription import HempPrescriptionProvider
from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult

# Module-level singleton; avoids reloading model files on every request.
_provider: HempPrescriptionProvider | None = None


def _get_provider(model_dir: str | Path | None = None) -> HempPrescriptionProvider:
    global _provider
    if _provider is None:
        _provider = HempPrescriptionProvider(model_dir=model_dir)
    return _provider


# ── Blocking constraints ───────────────────────────────────────────────────────

_BLOCKERS = [
    (lambda r: r.ph < 5.0,         "ph_too_acidic",  "Soil pH below 5.0 is outside hemp's tolerable range."),
    (lambda r: r.ph > 8.0,         "ph_too_alkaline", "Soil pH above 8.0 is outside hemp's tolerable range."),
    (lambda r: r.slope_percent > 20.0, "slope_excessive", "Slope above 20% makes mechanical hemp cultivation impractical."),
    (lambda r: r.avg_temp < 5.0,   "temp_too_cold",  "Mean growing-season temperature below 5°C prevents hemp establishment."),
    (lambda r: r.avg_temp > 35.0,  "temp_too_hot",   "Mean growing-season temperature above 35°C causes heat stress."),
    (lambda r: r.ec > 4.0,         "salinity_high",  "EC above 4.0 dS/m exceeds hemp's salinity tolerance."),
]


def _check_blockers(request: HempPrescriptionRequest) -> list[str]:
    return [message for check, _code, message in _BLOCKERS if check(request)]


# ── Agronomic notes ────────────────────────────────────────────────────────────

def _build_notes(request: HempPrescriptionRequest, result: HempPrescriptionResult) -> list[str]:
    notes: list[str] = []

    if not (6.0 <= request.ph <= 7.0):
        notes.append(f"pH {request.ph} is outside the ideal 6.0–7.0 range — consider lime or sulfur amendment.")
    if request.drainage_class == "poor":
        notes.append("Poor drainage raises waterlogging risk during wet spells — raised beds or tile drainage recommended.")
    if result.rec_nitrogen_kg_ha > 80:
        notes.append(f"High N deficit ({result.rec_nitrogen_kg_ha:.0f} kg/ha) — split application across two dressings.")
    if not request.irrigation_available and request.seasonal_rainfall_mm < 300:
        notes.append("Seasonal rainfall below 300 mm with no irrigation — drought stress likely during vegetative stage.")
    if request.slope_percent > 12:
        notes.append(f"Slope {request.slope_percent}% increases erosion risk — contour cultivation or cover strips advised.")
    if result.expected_yield_ton_ha < 3.0:
        notes.append("Forecast yield below 3 t/ha — agronomic conditions are suboptimal for fiber hemp.")

    return notes


# ── Public interface ───────────────────────────────────────────────────────────

def compute_hemp_prescription(
    request: HempPrescriptionRequest,
    *,
    model_dir: str | Path | None = None,
) -> HempPrescriptionResult:
    """Validate inputs, run ML inference, and return an annotated prescription."""

    blockers = _check_blockers(request)
    if blockers:
        return HempPrescriptionResult(
            rec_nitrogen_kg_ha=0.0,
            rec_phosphorus_kg_ha=0.0,
            rec_potassium_kg_ha=0.0,
            rec_irrigation_mm_week=0.0,
            expected_yield_ton_ha=0.0,
            suitable=False,
            blockers=blockers,
            notes=[],
            provider="none",
        )

    provider = _get_provider(model_dir)
    raw = provider.predict(request)
    notes = _build_notes(request, raw)

    return HempPrescriptionResult(
        rec_nitrogen_kg_ha=raw.rec_nitrogen_kg_ha,
        rec_phosphorus_kg_ha=raw.rec_phosphorus_kg_ha,
        rec_potassium_kg_ha=raw.rec_potassium_kg_ha,
        rec_irrigation_mm_week=raw.rec_irrigation_mm_week,
        expected_yield_ton_ha=raw.expected_yield_ton_ha,
        suitable=True,
        blockers=[],
        notes=notes,
        provider=raw.provider,
    )
