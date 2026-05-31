# app/api/hemp.py
"""Hemp prescription API endpoints."""
from fastapi import APIRouter, HTTPException

from app.engines.hemp_prescription_engine import compute_hemp_prescription
from app.schemas.hemp_cycle import AdaptivePrescriptionRequest
from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult

router = APIRouter(prefix="/hemp", tags=["hemp"])


@router.post("/prescription", response_model=HempPrescriptionResult)
def hemp_prescription_endpoint(request: HempPrescriptionRequest) -> HempPrescriptionResult:
    """Cold-start prescription from soil, field, and climate inputs only."""
    try:
        return compute_hemp_prescription(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/prescription/adaptive", response_model=HempPrescriptionResult)
def hemp_adaptive_prescription_endpoint(body: AdaptivePrescriptionRequest) -> HempPrescriptionResult:
    """Adaptive prescription: cold-start corrected by completed cycle history.

    Pass cycles sorted oldest-first in ``cycle_history``.
    When ``cycle_history`` is empty, behaves identically to POST /prescription.
    """
    try:
        return compute_hemp_prescription(
            body.request,
            cycle_history=body.cycle_history or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
