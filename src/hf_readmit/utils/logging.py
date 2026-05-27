from __future__ import annotations

import json
import logging
from logging import Logger, StreamHandler


class JsonLogFormatter(logging.Formatter):
    """Formatter for structured JSON log output."""

    def format(self, record: logging.LogRecord) -> str:
        message = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            message["exception"] = self.formatException(record.exc_info)
        return json.dumps(message)


def setup_logging(level: int = logging.INFO) -> Logger:
    """Configure root logger for JSON-formatted console output."""
    logger = logging.getLogger()
    logger.setLevel(level)

    handler = StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(JsonLogFormatter())

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
