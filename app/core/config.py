import os


def _split_csv(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]


DATA_DIR = os.getenv("DATA_DIR", "data")
ALERTS_CSV_V10 = os.path.join(DATA_DIR, "predictive_fill_top_alerts_v10.csv")
ACK_FILE = os.path.join(DATA_DIR, "acknowledged_alerts_v10.csv")

PAS_BASE_URL = os.getenv("PAS_BASE_URL", "https://YOUR-PAS-ENDPOINT")
PAS_PARTNER_ID = os.getenv("PAS_PARTNER_ID", "your-partner-id")
PAS_API_KEY = os.getenv("PAS_API_KEY", "your-api-key")
PAS_DECISION_TOPIC = os.getenv("PAS_DECISION_TOPIC", "inspection-decisions")

CORS_ALLOW_ORIGINS = _split_csv(
    os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
)
