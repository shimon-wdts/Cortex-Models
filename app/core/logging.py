import logging
import logging.config

from app.core.config import settings


_configured = False


def init_logging() -> None:
    global _configured
    if _configured:
        return

    level_name = str(settings.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
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
                "": {  # root logger
                    "handlers": ["console"],
                    "level": level,
                },
                "uvicorn.error": {"level": level, "handlers": ["console"]},
                "uvicorn.access": {"level": level, "handlers": ["console"]},
            },
        }
    )

    _configured = True
