from __future__ import annotations

import io
import json
import logging
from typing import cast

from app.core.logging import JsonFormatter, configure_logging, log_context


def _format_record(*, extra: dict[str, object]) -> dict[str, object]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("tests.unit.logging")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.info("unit_event", extra=extra)
    return cast(dict[str, object], json.loads(stream.getvalue()))


def test_json_formatter_preserves_direct_identifiers_and_safe_counts() -> None:
    record = _format_record(
        extra={
            "request_id": "req_direct",
            "job_id": "job_direct",
            "image_id": "image_direct",
            "run_id": "run_direct",
            "model_id": "model_direct",
            "deferred_count": 2,
            "error_count": 1,
            "secret": "must-not-leak",
        }
    )

    assert record["request_id"] == "req_direct"
    assert record["job_id"] == "job_direct"
    assert record["image_id"] == "image_direct"
    assert record["run_id"] == "run_direct"
    assert record["model_id"] == "model_direct"
    assert record["deferred_count"] == 2
    assert record["error_count"] == 1
    assert "secret" not in record


def test_bound_context_overrides_direct_extra_and_invalid_counts_are_omitted() -> None:
    with log_context(request_id="req_bound", run_id="run_bound"):
        record = _format_record(
            extra={
                "request_id": "req_direct",
                "run_id": "run_direct",
                "deferred_count": -1,
                "error_count": True,
            }
        )

    assert record["request_id"] == "req_bound"
    assert record["run_id"] == "run_bound"
    assert "deferred_count" not in record
    assert "error_count" not in record


def test_configure_logging_is_idempotent_and_preserves_external_handlers() -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    external = logging.StreamHandler(io.StringIO())
    try:
        root.addHandler(external)

        configure_logging("INFO", stream=io.StringIO())
        configure_logging("WARNING", stream=io.StringIO())

        assert external in root.handlers
        owned = [
            handler
            for handler in root.handlers
            if getattr(handler, "_nanoloop_json_handler", False)
        ]
        assert len(owned) == 1
        assert isinstance(owned[0].formatter, JsonFormatter)
        assert root.level == logging.WARNING
    finally:
        for handler in tuple(root.handlers):
            root.removeHandler(handler)
            if handler not in original_handlers:
                handler.close()
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)
