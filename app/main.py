from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as alerts_router
from app.core import config


app = FastAPI(
    title="Casino Intervention API",
    version="v10",
    description="Predicted chip-fill decisions, ROI, and rationale (v10).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(alerts_router)
