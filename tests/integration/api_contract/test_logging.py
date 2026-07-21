from __future__ import annotations

import io
import json
import logging

from app.core.logging import JsonFormatter, log_context


def test_json_logging_merges_safe_context_with_bound_values_taking_precedence() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("tests.contract.logging")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    with log_context(request_id="req_bound", job_id="job_bound", image_id="image_bound"):
        logger.info(
            "contract_event",
            extra={
                "request_id": "req_direct",
                "job_id": "job_direct",
                "image_id": "image_direct",
                "run_id": "run_direct",
                "model_id": "model_direct",
                "deferred_count": 3,
                "error_count": 1,
                "event": "tested",
                "status_code": 200,
                "secret": "must-not-leak",
            },
        )

    record = json.loads(stream.getvalue())
    assert record["request_id"] == "req_bound"
    assert record["job_id"] == "job_bound"
    assert record["image_id"] == "image_bound"
    assert record["run_id"] == "run_direct"
    assert record["model_id"] == "model_direct"
    assert record["deferred_count"] == 3
    assert record["error_count"] == 1
    assert record["event"] == "tested"
    assert record["status_code"] == 200
    assert "secret" not in record
