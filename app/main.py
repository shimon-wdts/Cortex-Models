from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dishka import make_container
from dishka.integrations.fastapi import DishkaRoute, setup_dishka
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router as alerts_router
from app.core.config import settings
from app.core.logging import init_logging
from app.di import AppProvider


logger = logging.getLogger(__name__)

init_logging()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info( "CORTEX_ENV=%s ", os.getenv("CORTEX_ENV", ""))
    logger.info("loaded_files=%s", getattr(settings, "_loaded_files", []))
    logger.info("app=%s, version=%s",
        settings.app.name,
        settings.app.version,
    )    
    logger.info("cors_allow_origins=%s", settings.cors_allow_origins)    

    yield

app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description="Predicted chip-fill decisions, ROI, and rationale (v10).",
    route_class=DishkaRoute,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(alerts_router)

container = make_container(AppProvider())
setup_dishka(container, app)

# Expose Prometheus metrics at /metrics for cluster scraping.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
