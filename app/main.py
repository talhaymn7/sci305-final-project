from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai.registry import get_ai_provider_registry
from app.api import agri_assistant, crop_profiles, crops, dashboards, fields, hemp, ranking, recommendations, soil_tests
from app.config import settings
from app.db import check_database_connection, dispose_database_engine
from app import models  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate external dependencies once when the application starts."""

    _ = app
    get_ai_provider_registry().validate_configuration()
    check_database_connection()
    yield
    dispose_database_engine()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(fields.router, prefix=settings.API_V1_PREFIX)
    app.include_router(soil_tests.router, prefix=settings.API_V1_PREFIX)
    app.include_router(crops.router, prefix=settings.API_V1_PREFIX)
    app.include_router(crop_profiles.router, prefix=settings.API_V1_PREFIX)
    app.include_router(dashboards.router, prefix=settings.API_V1_PREFIX)
    app.include_router(ranking.router, prefix=settings.API_V1_PREFIX)
    app.include_router(recommendations.router, prefix=settings.API_V1_PREFIX)
    app.include_router(agri_assistant.router, prefix=settings.API_V1_PREFIX)
    app.include_router(hemp.router, prefix=settings.API_V1_PREFIX)

    return app


app = create_app()
