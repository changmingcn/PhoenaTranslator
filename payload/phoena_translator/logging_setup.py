"""Explicit logging lifecycle for the translator service."""

from __future__ import annotations

import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOCK = threading.Lock()
_HANDLER_MARKER = "_phoena_translator_handler"
# The service has no external logrotate; bound the on-disk footprint here.
_LOG_MAX_BYTES = 50 * 1024 * 1024
_LOG_BACKUP_COUNT = 5


def configure_logging(
    log_file: Path,
    *,
    logger_name: str = "translator",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure service logging once, when runtime initialization is explicit."""
    with _LOCK:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False
        for handler in list(logger.handlers):
            if getattr(handler, _HANDLER_MARKER, False):
                logger.removeHandler(handler)
                handler.close()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handlers: list[logging.Handler] = [
            logging.StreamHandler(),
            RotatingFileHandler(
                log_file,
                maxBytes=_LOG_MAX_BYTES,
                backupCount=_LOG_BACKUP_COUNT,
                encoding="utf-8",
            ),
        ]
        for handler in handlers:
            setattr(handler, _HANDLER_MARKER, True)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logging.getLogger("fontTools").setLevel(logging.WARNING)
        return logger

