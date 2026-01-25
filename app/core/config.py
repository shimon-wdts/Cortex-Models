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
