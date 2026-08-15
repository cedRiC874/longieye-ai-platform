"""Structured, privacy-safe telemetry helpers for LongiEye."""

from __future__ import annotations

import json
import logging
import os
import re
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from uuid import uuid4


_REQUEST_ID: ContextVar[str] = ContextVar("longieye_request_id", default="-")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_EXTRA_FIELDS = (
    "event",
    "http_method",
    "http_path",
    "status_code",
    "duration_ms",
    "error_code",
    "model_id",
)


def normalize_request_id(candidate: str | None) -> str:
    """Accept a short trace ID or replace it with a generated UUID.

    Restricting the character set prevents log injection through request
    headers while still allowing common distributed-tracing identifiers.
    """

    if candidate and _SAFE_REQUEST_ID.fullmatch(candidate):
        return candidate
    return str(uuid4())


def set_request_id(value: str) -> Token[str]:
    return _REQUEST_ID.set(value)


def reset_request_id(token: Token[str]) -> None:
    _REQUEST_ID.reset(token)


def current_request_id() -> str:
    return _REQUEST_ID.get()


class JsonFormatter(logging.Formatter):
    """Emit one compact JSON object per log line without request bodies."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": current_request_id(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure and return the isolated LongiEye application logger."""

    logger = logging.getLogger("longieye")
    configured_level = (level or os.getenv("LONGIEYE_LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, configured_level, logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)
    return logger
