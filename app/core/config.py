import os
from pathlib import Path

from dynaconf import Dynaconf

env = os.getenv("CORTEX_ENV", "local").strip().lower() or "local"
config_dir = Path(__file__).resolve().parents[2] / "config"
_settings_file_candidates = [
    config_dir / "settings.yaml",
    config_dir / "models.yaml",
    config_dir / f"settings.{env}.yaml",
    config_dir / f"models.{env}.yaml",
]
settings_files = [str(path) for path in _settings_file_candidates if path.exists()]

settings = Dynaconf(
    settings_files=settings_files,
    environments=False,
    env_switcher="CORTEX_ENV",
    envvar_prefix="CORTEX",
    load_dotenv=True,
    merge_enabled=True,
)
