import os

from dynaconf import Dynaconf

env = os.getenv("CORTEX_ENV", "local").strip().lower() or "local"
settings_files = [
    "config/settings.yaml",
    "config/models.yaml",
    f"config/settings.{env}.yaml",
    # f"config/models.{env}.yaml",
]

settings = Dynaconf(
    settings_files=settings_files,
    environments=False,
    env_switcher="CORTEX_ENV",
    envvar_prefix="CORTEX",
    load_dotenv=True,
)

