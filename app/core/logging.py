import logging
import logging.config

from app.core.config import settings


_configured = False


def init_logging() -> None:
    global _configured
    if _configured:
        return

    level_name = str(settings.get("log.level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    disable_existing_loggers = bool(
        settings.get("log.disable_existing_loggers", False)
    )
    try:
        import colorlog  # noqa: F401
    except ModuleNotFoundError:
        formatter = {
            "format": "%(asctime)s %(levelname)s: %(message)s",
        }
    else:
        formatter = {
            "()": "colorlog.ColoredFormatter",
            "format": "%(log_color)s%(asctime)s %(levelname)s:%(reset)s %(message)s",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": disable_existing_loggers,
            "formatters": {
                "standard": formatter
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": level,
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": level, "handlers": ["console"]},
            "loggers": {
                "app": {"level": level, "propagate": True},
                "uvicorn.error": {"level": level, "handlers": ["console"], "propagate": False},
                "uvicorn.access": {"level": level,  "handlers": ["console"], "propagate": False},
            },
        }
    )

    _configured = True
