import os

from dynaconf import Dynaconf


settings = Dynaconf(
    settings_files=[
        "config/settings.yaml",
        "config/settings.dev.yaml",
        "config/settings.uat.yaml",
        "config/settings.prod.yaml",
        "config/settings.local.yaml",
    ],
    environments=False,
    env_switcher="CORTEX_ENV",
    envvar_prefix="CORTEX",
    load_dotenv=True,
)

DATA_DIR = settings.data_dir
ALERTS_CSV_V10 = os.path.join(settings.data_dir, settings.alerts_csv_name)
ACK_FILE = os.path.join(settings.data_dir, settings.ack_file_name)

PAS_BASE_URL = settings.pas_base_url
PAS_PARTNER_ID = settings.pas_partner_id
PAS_API_KEY = settings.pas_api_key
PAS_DECISION_TOPIC = settings.pas_decision_topic

CORS_ALLOW_ORIGINS = settings.cors_allow_origins
