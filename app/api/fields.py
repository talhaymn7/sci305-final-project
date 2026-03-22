from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.service_errors import raise_http_exception_for_service_error
from app.database import get_db
from app.schemas.field import FieldCreate, FieldRead, FieldUpdate
from app.schemas.management import ManagementPlanRead
from app.schemas.weather_history import ClimateSummary
from app.services import field_catalog_service
from app.services.field_evaluation_service import (
    get_field_climate_summary as get_field_climate_summary_service,
)
from app.services.management_service import (
    MAX_PLAN_WEEKS,
    ManagementPlanConflictError,
    ManagementPlanNotFoundError,
    ManagementService,
)

router = APIRouter(prefix="/fields", tags=["fields"])


@router.post("/", response_model=FieldRead, status_code=201)
def create_field(field_data: FieldCreate, db: Session = Depends(get_db)):
    try:
        return field_catalog_service.create_field(db, field_data)
    except Exception as exc:
        raise_http_exception_for_service_error(exc)


@router.get("/", response_model=list[FieldRead])
def list_fields(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return field_catalog_service.list_fields(db, skip=skip, limit=limit)


@router.get("/{field_id}", response_model=FieldRead)
def get_field(field_id: str, db: Session = Depends(get_db)):
    try:
        return field_catalog_service.get_field(db, field_id)
    except Exception as exc:
        raise_http_exception_for_service_error(exc)


@router.get("/{field_id}/climate-summary", response_model=ClimateSummary)
def get_field_climate_summary(
    field_id: str,
    days: int = Query(default=30, ge=1, le=366),
    heat_threshold_c: float | None = Query(default=None, ge=-50, le=70),
    db: Session = Depends(get_db),
):
    try:
        return get_field_climate_summary_service(
            db,
            field_id,
            days=days,
            heat_threshold_c=heat_threshold_c,
        )
    except Exception as exc:
        raise_http_exception_for_service_error(exc)


@router.get("/{field_id}/management-plan", response_model=ManagementPlanRead)
def get_management_plan(
    field_id: int,
    target_date: date | None = Query(default=None),
    weeks: int = Query(default=8, ge=1, le=MAX_PLAN_WEEKS),
    db: Session = Depends(get_db),
):
    service = ManagementService(db)
    try:
        return service.get_management_plan(
            field_id,
            target_date=target_date or date.today(),
            weeks=weeks,
        )
    except ManagementPlanNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ManagementPlanConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/{field_id}", response_model=FieldRead)
def update_field(field_id: str, field_data: FieldUpdate, db: Session = Depends(get_db)):
    try:
        return field_catalog_service.update_field(db, field_id, field_data)
    except Exception as exc:
        raise_http_exception_for_service_error(exc)


@router.delete("/{field_id}", status_code=204)
def delete_field(field_id: str, db: Session = Depends(get_db)):
    try:
        field_catalog_service.delete_field(db, field_id)
    except Exception as exc:
        raise_http_exception_for_service_error(exc)
