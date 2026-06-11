from contextlib import asynccontextmanager
import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dishka import make_container
from dishka.integrations.fastapi import DishkaRoute, setup_dishka

from app.api.routes import router as alerts_router
from app.core.config import settings
from app.core.logging import init_logging
from app.di import AppProvider
from app.metrics import APP_EXCEPTIONS_TOTAL, monitor_event_loop_lag, route_label, setup_metrics


logger = logging.getLogger(__name__)

init_logging()

def log_banner():
    banner_path = Path("app/resources/banner.txt")

    try:
        banner = banner_path.read_text(encoding="utf-8")
        logger.info("\n%s", banner)
    except FileNotFoundError:
        logger.warning("Banner file not found: %s", banner_path)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log_banner()
    logger.info( "CORTEX_ENV=%s ", os.getenv("CORTEX_ENV", ""))
    logger.info("loaded_files=%s", getattr(settings, "_loaded_files", []))
    logger.info("app=%s, version=%s",
        settings.app.name,
        settings.app.version,
    )    
    logger.info("cors_allow_origins=%s", settings.cors_allow_origins)    

    stop_event = asyncio.Event()
    lag_task = asyncio.create_task(monitor_event_loop_lag(stop_event))

    yield

    stop_event.set()
    lag_task.cancel()

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

@app.middleware("http")
async def exception_counter_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        route_path = route_label(request.scope.get("route"))
        APP_EXCEPTIONS_TOTAL.labels(type(exc).__name__, route_path).inc()
        raise

app.include_router(alerts_router)

container = make_container(AppProvider())
setup_dishka(container, app)

setup_metrics(app)
