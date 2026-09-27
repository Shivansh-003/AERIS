import logging
import sys

from backend.app.core.config import get_settings


def setup_logging() -> logging.Logger:
    settings = get_settings()
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger("aeris")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger
