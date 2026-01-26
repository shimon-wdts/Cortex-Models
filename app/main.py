from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dishka import make_container
from dishka.integrations.fastapi import DishkaRoute, setup_dishka

from app.api.routes import router as alerts_router
from app.core.config import settings
from app.core.logging import init_logging
from app.di import AppProvider


init_logging()

app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description="Predicted chip-fill decisions, ROI, and rationale (v10).",
    route_class=DishkaRoute,
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
