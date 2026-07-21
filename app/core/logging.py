"""Structured JSON logging with request-scoped scientific identifiers."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

_LOG_CONTEXT: ContextVar[dict[str, str] | None] = ContextVar(
    "nanoloop_log_context", default=None
)

_CONTEXT_KEYS = frozenset(
    {
        "request_id",
        "job_id",
        "image_id",
        "run_id",
        "model_id",
        "tenant_id",
        "principal_id",
        "credential_id",
    }
)
_EXTRA_KEYS = frozenset(
    {
        "component",
        "detail",
        "duration_ms",
        "event",
        "method",
        "outcome",
        "path",
        "status_code",
        "auth_mode",
        "auth_outcome",
        "auth_reason",
    }
)
_COUNT_KEYS = frozenset({"deferred_count", "error_count"})
_HANDLER_MARKER = "_nanoloop_json_handler"
_FILE_TOKEN_PATH_SEGMENT = re.compile(r"(?<=/files/)[^/?\s\"]+")


def get_log_context() -> dict[str, str]:
    """Return a copy so callers cannot mutate another task's context."""

    return dict(_LOG_CONTEXT.get() or {})


def bind_log_context(**values: str | None) -> Token[dict[str, str] | None]:
    """Add whitelisted non-empty values to the current async context."""

    current = get_log_context()
    current.update(
        {
            key: value
            for key, value in values.items()
            if key in _CONTEXT_KEYS and isinstance(value, str) and value
        }
    )
    return _LOG_CONTEXT.set(current)


def reset_log_context(token: Token[dict[str, str] | None]) -> None:
    _LOG_CONTEXT.reset(token)


@contextmanager
def log_context(**values: str | None) -> Iterator[None]:
    token = bind_log_context(**values)
    try:
        yield
    finally:
        reset_log_context(token)


def current_request_id() -> str | None:
    return (_LOG_CONTEXT.get() or {}).get("request_id")


class JsonFormatter(logging.Formatter):
    """Serialize stable log fields without recording request bodies or secrets."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            # Uvicorn access messages include the raw request target. Signed file URLs carry a
            # bearer token in the path, so redaction must happen at the shared formatter boundary
            # rather than only in NanoLoop's request-completion middleware.
            "message": _redact_sensitive_log_text(record.getMessage()),
        }
        direct_context = {
            key: value
            for key in _CONTEXT_KEYS
            if isinstance(value := getattr(record, key, None), str) and value
        }
        # Bound request/route context is authoritative when lower layers also
        # provide an identifier through ``extra``.
        direct_context.update(get_log_context())
        payload.update(direct_context)
        for key in _EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        for key in _COUNT_KEYS:
            value = getattr(record, key, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _redact_sensitive_log_text(value: str) -> str:
    return _FILE_TOKEN_PATH_SEGMENT.sub("<redacted>", value)


def configure_logging(level: str = "INFO", *, stream: Any = None) -> None:
    """Install one JSON handler for the application and Uvicorn loggers."""

    output = stream if stream is not None else sys.stderr
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for existing in tuple(root.handlers):
        if getattr(existing, _HANDLER_MARKER, False):
            root.removeHandler(existing)
            existing.close()
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)
    root.setLevel(level.upper())

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
