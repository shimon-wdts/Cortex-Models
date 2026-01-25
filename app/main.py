from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dishka import make_container
from dishka.integrations.fastapi import DishkaRoute, setup_dishka

from app.api.routes import router as alerts_router
from app.core import config
from app.di import AppProvider


app = FastAPI(
    title="Casino Intervention API",
    version="v10",
    description="Predicted chip-fill decisions, ROI, and rationale (v10).",
    route_class=DishkaRoute,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(alerts_router)

container = make_container(AppProvider())
setup_dishka(container, app)
