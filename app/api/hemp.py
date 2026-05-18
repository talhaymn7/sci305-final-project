"""Hemp prescription API endpoint."""

from fastapi import APIRouter, HTTPException

from app.engines.hemp_prescription_engine import compute_hemp_prescription
from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult

router = APIRouter(prefix="/hemp", tags=["hemp"])


@router.post("/prescription", response_model=HempPrescriptionResult)
def hemp_prescription_endpoint(request: HempPrescriptionRequest) -> HempPrescriptionResult:
    """Generate a site-specific hemp prescription from soil, field, and climate inputs."""
    try:
        return compute_hemp_prescription(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
