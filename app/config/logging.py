from logging import Formatter, Logger, StreamHandler, getLogger

from app.config import settings


def get_logger() -> Logger:
    """Setup application logger."""
    logger = getLogger("app")

    if not logger.handlers:
        handler = StreamHandler()
        handler.setFormatter(Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(settings.LOGGING_LEVEL)
    return logger


logger = get_logger()
