from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.service_errors import raise_http_exception_for_service_error
from app.database import get_db
from app.schemas.recommendation import RankedFieldResult
from app.services.field_evaluation_service import get_field_recommendation

router = APIRouter(prefix="/recommendation", tags=["recommendations"])


@router.get("/{field_id}/{crop_id}", response_model=RankedFieldResult)
def get_recommendation(
    field_id: int,
    crop_id: int,
    climate_days: int | None = Query(default=None, ge=1, le=366),
    heat_threshold_c: float | None = Query(default=None, ge=-50, le=70),
    db: Session = Depends(get_db),
):
    try:
        return get_field_recommendation(
            db,
            field_id=field_id,
            crop_id=crop_id,
            climate_days=climate_days,
            heat_threshold_c=heat_threshold_c,
        )
    except Exception as exc:
        raise_http_exception_for_service_error(exc)
