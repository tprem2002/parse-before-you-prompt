"""Application logging configuration."""

import logging
from logging.config import dictConfig


def configure_logging(log_level: str | None = None) -> None:
    """Configure concise process-wide console logging."""

    if log_level is None:
        from app.core.config import get_settings

        log_level = get_settings().log_level

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": log_level,
                }
            },
            "loggers": {
                "httpx": {"level": "WARNING", "propagate": True},
                "httpx2": {"level": "WARNING", "propagate": True},
                "httpcore": {"level": "WARNING", "propagate": True},
                "openai": {"level": "WARNING", "propagate": True},
            },
            "root": {"handlers": ["console"], "level": log_level},
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named application logger."""

    return logging.getLogger(name)
